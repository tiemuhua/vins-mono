#include "estimator.h"

Estimator::Estimator() : feature_manager{Rs} {
    LOG_I("init begins");
    clearState();
}

void Estimator::setParameter() {
    for (int i = 0; i < NUM_OF_CAM; i++) {
        tic[i] = TIC[i];
        ric[i] = RIC[i];
    }
    feature_manager.setRic(ric);
    ProjectionFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionTdFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    td = TD;
}

void Estimator::clearState() {
    for (int i = 0; i < WINDOW_SIZE + 1; i++) {
        Rs[i].setIdentity();
        Ps[i].setZero();
        Vs[i].setZero();
        Bas[i].setZero();
        Bgs[i].setZero();
        dt_buf[i].clear();
        linear_acceleration_buf[i].clear();
        angular_velocity_buf[i].clear();

        if (pre_integrations[i] != nullptr)
            delete pre_integrations[i];
        pre_integrations[i] = nullptr;
    }

    for (int i = 0; i < NUM_OF_CAM; i++) {
        tic[i] = Vector3d::Zero();
        ric[i] = Matrix3d::Identity();
    }

    for (auto &it: all_image_frame) {
        if (it.second.pre_integration != nullptr) {
            delete it.second.pre_integration;
            it.second.pre_integration = nullptr;
        }
    }

    solver_flag = INITIAL;
    first_imu = false,
            sum_of_back = 0;
    sum_of_front = 0;
    frame_count = 0;
    solver_flag = INITIAL;
    initial_timestamp = 0;
    all_image_frame.clear();
    td = TD;


    delete tmp_pre_integration;
    delete last_marginalization_info;

    tmp_pre_integration = nullptr;
    last_marginalization_info = nullptr;
    last_marginal_param_blocks.clear();

    feature_manager.clearState();

    failure_occur = false;
    relocalization_info = false;

    drift_correct_r = Matrix3d::Identity();
}

void Estimator::processIMU(double dt, const Vector3d &linear_acceleration, const Vector3d &angular_velocity) {
    if (!first_imu) {
        first_imu = true;
        acc_0 = linear_acceleration;
        gyr_0 = angular_velocity;
    }

    if (!pre_integrations[frame_count]) {
        pre_integrations[frame_count] = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};
    }
    if (frame_count != 0) {
        pre_integrations[frame_count]->push_back(dt, linear_acceleration, angular_velocity);
        tmp_pre_integration->push_back(dt, linear_acceleration, angular_velocity);

        dt_buf[frame_count].push_back(dt);
        linear_acceleration_buf[frame_count].push_back(linear_acceleration);
        angular_velocity_buf[frame_count].push_back(angular_velocity);

        int j = frame_count;
        Vector3d un_acc_0 = Rs[j] * (acc_0 - Bas[j]) - g;
        Vector3d un_gyr = 0.5 * (gyr_0 + angular_velocity) - Bgs[j];
        Rs[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        Vector3d un_acc_1 = Rs[j] * (linear_acceleration - Bas[j]) - g;
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        Ps[j] += dt * Vs[j] + 0.5 * dt * dt * un_acc;
        Vs[j] += dt * un_acc;
    }
    acc_0 = linear_acceleration;
    gyr_0 = angular_velocity;
}

void Estimator::processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image,
                             const double &time_stamp) {
    LOG_D("new image coming ------------------------------------------");
    LOG_D("Adding feature points %lu", image.size());
    if (feature_manager.addFeatureCheckParallax(frame_count, image, td))
        marginalization_flag = MARGIN_OLD;
    else
        marginalization_flag = MARGIN_SECOND_NEW;

    LOG_D("this frame is--------------------%s", marginalization_flag ? "reject" : "accept");
    LOG_D("%s", marginalization_flag ? "Non-keyframe" : "Keyframe");
    LOG_D("Solving %d", frame_count);
    LOG_D("number of feature: %d", feature_manager.getFeatureCount());
    time_stamps[frame_count] = time_stamp;

    ImageFrame image_frame(image, time_stamp);
    image_frame.pre_integration = tmp_pre_integration;
    all_image_frame.insert(make_pair(time_stamp, image_frame));
    tmp_pre_integration = new IntegrationBase{acc_0, gyr_0, Bas[frame_count], Bgs[frame_count]};

    if (ESTIMATE_EXTRINSIC == 2) {
        LOG_I("calibrating extrinsic param, rotation movement is needed");
        if (frame_count != 0) {
            vector<pair<Vector3d, Vector3d>> corres = feature_manager.getCorresponding(frame_count - 1, frame_count);
            Matrix3d calib_ric;
            if (initial_ex_rotation.CalibrationExRotation(corres, pre_integrations[frame_count]->delta_q, calib_ric)) {
                LOG_W("initial extrinsic rotation calib success");
                ric[0] = calib_ric;
                RIC[0] = calib_ric;
                ESTIMATE_EXTRINSIC = 1;
            }
        }
    }

    if (solver_flag == INITIAL) {
        if (frame_count == WINDOW_SIZE) {
            bool result = false;
            if (ESTIMATE_EXTRINSIC != 2 && (time_stamp - initial_timestamp) > 0.1) {
                result = initialStructure();
                initial_timestamp = time_stamp;
            }
            if (result) {
                solver_flag = NON_LINEAR;
                solveOdometry();
                slideWindow();
                feature_manager.removeFailures();
                LOG_I("Initialization finish!");
                last_R = Rs[WINDOW_SIZE];
                last_P = Ps[WINDOW_SIZE];
                last_R0 = Rs[0];
                last_P0 = Ps[0];

            } else
                slideWindow();
        } else
            frame_count++;
    } else {
        TicToc t_solve;
        solveOdometry();
        LOG_D("solver costs: %fms", t_solve.toc());

        if (failureDetection()) {
            LOG_W("failure detection!");
            failure_occur = true;
            clearState();
            setParameter();
            LOG_W("system reboot!");
            return;
        }

        TicToc t_margin;
        slideWindow();
        feature_manager.removeFailures();
        LOG_D("marginalization costs: %fms", t_margin.toc());
        // prepare output of VINS
        key_poses.clear();
        for (int i = 0; i <= WINDOW_SIZE; i++)
            key_poses.push_back(Ps[i]);

        last_R = Rs[WINDOW_SIZE];
        last_P = Ps[WINDOW_SIZE];
        last_R0 = Rs[0];
        last_P0 = Ps[0];
    }
}

bool Estimator::initialStructure() {
    TicToc t_sfm;
    //check imu observability
    Vector3d sum_acc;
    // todo tiemuhuaguo 原始代码很奇怪，all_image_frame隔一个用一个，而且all_image_frame.size() - 1是什么意思？
    for (const pair<const double, ImageFrame> &frame: all_image_frame) {
        double dt = frame.second.pre_integration->sum_dt;
        Vector3d tmp_acc = frame.second.pre_integration->delta_v / dt;
        sum_acc += tmp_acc;
    }
    Vector3d aver_acc = sum_acc / (double )all_image_frame.size();
    double var = 0;
    for (const pair<const double, ImageFrame> &frame:all_image_frame) {
        double dt = frame.second.pre_integration->sum_dt;
        Vector3d tmp_acc = frame.second.pre_integration->delta_v / dt;
        var += (tmp_acc - aver_acc).transpose() * (tmp_acc - aver_acc);
    }
    var = sqrt(var / (double )all_image_frame.size());
    if (var < 0.25) {
        LOG_E("IMU excitation not enough!");
        //return false;
    }

    // global sfm
    map<int, Vector3d> sfm_tracked_points;
    vector<SFMFeature> sfm_f;
    for (FeaturePerId &it_per_id: feature_manager.feature) {
        int imu_j = it_per_id.start_frame - 1;
        SFMFeature tmp_feature;
        tmp_feature.state = false;
        tmp_feature.id = it_per_id.feature_id;
        for (FeaturePerFrame &it_per_frame: it_per_id.feature_per_frame) {
            imu_j++;
            Vector3d pts_j = it_per_frame.point;
            tmp_feature.observation.emplace_back(make_pair(imu_j, Eigen::Vector2d{pts_j.x(), pts_j.y()}));
        }
        sfm_f.push_back(tmp_feature);
    }
    Matrix3d relative_R;
    Vector3d relative_T;
    int l;
    if (!relativePose(relative_R, relative_T, l)) {
        LOG_I("Not enough features or parallax; Move device around");
        return false;
    }
    Quaterniond Q[frame_count + 1];
    Vector3d T[frame_count + 1];
    if (!GlobalSFM::construct(frame_count + 1, Q, T, l, relative_R, relative_T, sfm_f, sfm_tracked_points)) {
        LOG_D("global SFM failed!");
        marginalization_flag = MARGIN_OLD;
        return false;
    }

    //solve pnp for all frame
    int i = 0;
    for (pair<const double, ImageFrame> &frame:all_image_frame) {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        if ((frame.first) == time_stamps[i]) {
            frame.second.is_key_frame = true;
            frame.second.R = Q[i].toRotationMatrix() * RIC[0].transpose();
            frame.second.T = T[i];
            i++;
            continue;
        }
        if ((frame.first) > time_stamps[i]) {
            i++;
        }
        Matrix3d R_initial = (Q[i].inverse()).toRotationMatrix();
        Vector3d P_initial = -R_initial * T[i];
        cv::eigen2cv(R_initial, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_initial, t);

        frame.second.is_key_frame = false;
        vector<cv::Point3f> pts_3_vector;
        vector<cv::Point2f> pts_2_vector;
        for (auto &id_pts: frame.second.points) {
            int feature_id = id_pts.first;
            for (auto &i_p: id_pts.second) {
                auto it = sfm_tracked_points.find(feature_id);
                if (it != sfm_tracked_points.end()) {
                    Vector3d world_pts = it->second;
                    pts_3_vector.emplace_back(world_pts(0), world_pts(1), world_pts(2));
                    pts_2_vector.emplace_back(i_p.second(0), i_p.second(1));
                }
            }
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        if (pts_3_vector.size() < 6) {
            cout << "pts_3_vector size " << pts_3_vector.size() << endl;
            LOG_D("Not enough points for solve pnp !");
            return false;
        }
        if (!cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, false)) {
            LOG_D("solve pnp fail!");
            return false;
        }

        cv::Rodrigues(rvec, r);
        MatrixXd tmp_R_pnp;
        cv::cv2eigen(r, tmp_R_pnp);
        MatrixXd R_pnp = tmp_R_pnp.transpose();
        MatrixXd T_pnp;
        cv::cv2eigen(t, T_pnp);
        frame.second.T = R_pnp * (-T_pnp);
        frame.second.R = R_pnp * RIC[0].transpose();
    }
    if (visualInitialAlign())
        return true;
    else {
        LOG_I("misaligned visual structure with IMU");
        return false;
    }

}

bool Estimator::visualInitialAlign() {
    TicToc t_g;
    VectorXd x;
    //solve scale
    bool result = VisualIMUAlignment(all_image_frame, Bgs, g, x);
    if (!result) {
        LOG_D("solve g failed!");
        return false;
    }

    // change state
    for (int i = 0; i <= frame_count; i++) {
        Matrix3d Ri = all_image_frame[time_stamps[i]].R;
        Vector3d Pi = all_image_frame[time_stamps[i]].T;
        Ps[i] = Pi;
        Rs[i] = Ri;
        all_image_frame[time_stamps[i]].is_key_frame = true;
    }

    VectorXd dep = feature_manager.getDepthVector();
    for (int i = 0; i < dep.size(); i++)
        dep[i] = -1;
    feature_manager.clearDepth(dep);

    //triangulate on cam pose , no tic
    Vector3d TIC_TMP[NUM_OF_CAM];
    for (auto &i: TIC_TMP)
        i.setZero();
    ric[0] = RIC[0];
    feature_manager.setRic(ric);
    feature_manager.triangulate(Ps, &(TIC_TMP[0]), &(RIC[0]));

    double s = (x.tail<1>())(0);
    for (int i = 0; i <= WINDOW_SIZE; i++) {
        pre_integrations[i]->repropagate(Vector3d::Zero(), Bgs[i]);
    }
    for (int i = frame_count; i >= 0; i--)
        Ps[i] = s * Ps[i] - Rs[i] * TIC[0] - (s * Ps[0] - Rs[0] * TIC[0]);
    int kv = -1;
    for (const auto &frame:all_image_frame) {
        if (frame.second.is_key_frame) {
            kv++;
            Vs[kv] = frame.second.R * x.segment<3>(kv*3);
        }
    }
    for (FeaturePerId &it_per_id: feature_manager.feature) {
        if (!(it_per_id.feature_per_frame.size() >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth *= s;
    }

    Matrix3d R0 = Utility::g2R(g);
    double yaw = Utility::R2ypr(R0 * Rs[0]).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    g = R0 * g;
    //Matrix3d rot_diff = R0 * Rs[0].transpose();
    Matrix3d rot_diff = R0;
    for (int i = 0; i <= frame_count; i++) {
        Ps[i] = rot_diff * Ps[i];
        Rs[i] = rot_diff * Rs[i];
        Vs[i] = rot_diff * Vs[i];
    }

    return true;
}

bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l) {
    // find previous frame which contains enough correspondence and parallax with the newest frame
    for (int i = 0; i < WINDOW_SIZE; i++) {
        vector<pair<Vector3d, Vector3d>> correspondences = feature_manager.getCorresponding(i, WINDOW_SIZE);
        if (correspondences.size() > 20) {
            double sum_parallax = 0;
            for (auto &correspond: correspondences) {
                Vector2d pts_0(correspond.first(0), correspond.first(1));
                Vector2d pts_1(correspond.second(0), correspond.second(1));
                double parallax = (pts_0 - pts_1).norm();
                sum_parallax += parallax;
            }
            double average_parallax = 1.0 * sum_parallax / int(correspondences.size());
            if (average_parallax * 460 > 30 && m_estimator.solveRelativeRT(correspondences, relative_R, relative_T)) {
                l = i;
                LOG_D("average_parallax %f choose l %d and newest frame to triangulate the whole structure",
                      average_parallax * 460, l);
                return true;
            }
        }
    }
    return false;
}

void Estimator::solveOdometry() {
    if (frame_count < WINDOW_SIZE)
        return;
    if (solver_flag == NON_LINEAR) {
        TicToc t_tri;
        feature_manager.triangulate(Ps, tic, ric);
        LOG_D("triangulation costs %f", t_tri.toc());
        optimization();
    }
}

void Estimator::vector2double() {
    for (int i = 0; i <= WINDOW_SIZE; i++) {
        para_Pose[i][0] = Ps[i].x();
        para_Pose[i][1] = Ps[i].y();
        para_Pose[i][2] = Ps[i].z();
        Quaterniond q{Rs[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        para_SpeedBias[i][0] = Vs[i].x();
        para_SpeedBias[i][1] = Vs[i].y();
        para_SpeedBias[i][2] = Vs[i].z();

        para_SpeedBias[i][3] = Bas[i].x();
        para_SpeedBias[i][4] = Bas[i].y();
        para_SpeedBias[i][5] = Bas[i].z();

        para_SpeedBias[i][6] = Bgs[i].x();
        para_SpeedBias[i][7] = Bgs[i].y();
        para_SpeedBias[i][8] = Bgs[i].z();
    }
    for (int i = 0; i < NUM_OF_CAM; i++) {
        para_Ex_Pose[i][0] = tic[i].x();
        para_Ex_Pose[i][1] = tic[i].y();
        para_Ex_Pose[i][2] = tic[i].z();
        Quaterniond q{ric[i]};
        para_Ex_Pose[i][3] = q.x();
        para_Ex_Pose[i][4] = q.y();
        para_Ex_Pose[i][5] = q.z();
        para_Ex_Pose[i][6] = q.w();
    }

    VectorXd dep = feature_manager.getDepthVector();
    for (int i = 0; i < feature_manager.getFeatureCount(); i++)
        para_Feature[i][0] = dep(i);
    if (ESTIMATE_TD)
        para_Td[0][0] = td;
}

void Estimator::double2vector() {
    Vector3d origin_R0 = Utility::R2ypr(Rs[0]);
    Vector3d origin_P0 = Ps[0];

    if (failure_occur) {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = false;
    }
    Vector3d origin_R00 = Utility::R2ypr(Quaterniond(para_Pose[0][6],
                                                     para_Pose[0][3],
                                                     para_Pose[0][4],
                                                     para_Pose[0][5]).toRotationMatrix());
    double y_diff = origin_R0.x() - origin_R00.x();

    Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0));
    if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0) {
        LOG_D("euler singular point!");
        rot_diff = Rs[0] * Quaterniond(para_Pose[0][6],
                                       para_Pose[0][3],
                                       para_Pose[0][4],
                                       para_Pose[0][5]).toRotationMatrix().transpose();
    }

    for (int i = 0; i <= WINDOW_SIZE; i++) {

        Rs[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4],
                                       para_Pose[i][5]).normalized().toRotationMatrix();

        Ps[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                    para_Pose[i][1] - para_Pose[0][1],
                                    para_Pose[i][2] - para_Pose[0][2]) + origin_P0;

        Vs[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                    para_SpeedBias[i][1],
                                    para_SpeedBias[i][2]);

        Bas[i] = Vector3d(para_SpeedBias[i][3],
                          para_SpeedBias[i][4],
                          para_SpeedBias[i][5]);

        Bgs[i] = Vector3d(para_SpeedBias[i][6],
                          para_SpeedBias[i][7],
                          para_SpeedBias[i][8]);
    }

    for (int i = 0; i < NUM_OF_CAM; i++) {
        tic[i] = Vector3d(para_Ex_Pose[i][0],
                          para_Ex_Pose[i][1],
                          para_Ex_Pose[i][2]);
        ric[i] = Quaterniond(para_Ex_Pose[i][6],
                             para_Ex_Pose[i][3],
                             para_Ex_Pose[i][4],
                             para_Ex_Pose[i][5]).toRotationMatrix();
    }

    VectorXd dep = feature_manager.getDepthVector();
    for (int i = 0; i < feature_manager.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    feature_manager.setDepth(dep);
    if (ESTIMATE_TD)
        td = para_Td[0][0];

    // relative info between two loop frame
    if (relocalization_info) {
        Matrix3d relo_r;
        relo_r = rot_diff *
                 Quaterniond(relo_Pose[6], relo_Pose[3], relo_Pose[4], relo_Pose[5]).normalized().toRotationMatrix();
        double drift_correct_yaw;
        drift_correct_yaw = Utility::R2ypr(prev_relo_r).x() - Utility::R2ypr(relo_r).x();
        drift_correct_r = Utility::ypr2R(Vector3d(drift_correct_yaw, 0, 0));
        relocalization_info = false;
    }
}

bool Estimator::failureDetection() {
    if (feature_manager.last_track_num < 2) {
        LOG_I(" little feature %d", feature_manager.last_track_num);
        //return true;
    }
    if (Bas[WINDOW_SIZE].norm() > 2.5) {
        LOG_I(" big IMU acc bias estimation %f", Bas[WINDOW_SIZE].norm());
        return true;
    }
    if (Bgs[WINDOW_SIZE].norm() > 1.0) {
        LOG_I(" big IMU gyr bias estimation %f", Bgs[WINDOW_SIZE].norm());
        return true;
    }
    Vector3d tmp_P = Ps[WINDOW_SIZE];
    if ((tmp_P - last_P).norm() > 5) {
        LOG_I(" big translation");
        return true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 1) {
        LOG_I(" big z translation");
        return true;
    }
    Matrix3d tmp_R = Rs[WINDOW_SIZE];
    Matrix3d delta_R = tmp_R.transpose() * last_R;
    Quaterniond delta_Q(delta_R);
    double delta_angle;
    delta_angle = acos(delta_Q.w()) * 2.0 / 3.14 * 180.0;
    if (delta_angle > 50) {
        LOG_I(" big delta_angle ");
        //return true;
    }
    return false;
}

namespace ceres {
    typedef ProductManifold<EuclideanManifold<3>, QuaternionManifold> SE3Manifold;
}

void Estimator::optimization() {
    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::CauchyLoss(1.0);
    for (int i = 0; i < WINDOW_SIZE + 1; i++) {
        ceres::Manifold *local_parameterization = new ceres::SE3Manifold();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, local_parameterization);
        problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEED_BIAS);
    }
    for (auto &i: para_Ex_Pose) {
        ceres::Manifold *local_parameterization = new ceres::SE3Manifold();
        problem.AddParameterBlock(i, SIZE_POSE, local_parameterization);
        if (!ESTIMATE_EXTRINSIC) {
            LOG_D("fix extinsic param");
            problem.SetParameterBlockConstant(i);
        } else
            LOG_D("estimate extinsic param");
    }
    if (ESTIMATE_TD) {
        problem.AddParameterBlock(para_Td[0], 1);
    }

    TicToc t_whole, t_prepare;
    vector2double();

    if (last_marginalization_info) {
        auto *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
        problem.AddResidualBlock(marginalization_factor, nullptr,
                                 last_marginal_param_blocks);
    }

    for (int i = 0; i < WINDOW_SIZE; i++) {
        int j = i + 1;
        if (pre_integrations[j]->sum_dt > 10.0)
            continue;
        auto *imu_factor = new IMUFactor(pre_integrations[j]);
        problem.AddResidualBlock(imu_factor, nullptr, para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
    }
    int f_m_cnt = 0;
    int feature_index = -1;
    for (FeaturePerId &it_per_id: feature_manager.feature) {
        if (!(it_per_id.feature_per_frame.size() >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
            continue;

        ++feature_index;

        int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;

        Vector3d pts_i = it_per_id.feature_per_frame[0].point;

        for (auto &it_per_frame: it_per_id.feature_per_frame) {
            imu_j++;
            if (imu_i == imu_j) {
                continue;
            }
            Vector3d pts_j = it_per_frame.point;
            if (ESTIMATE_TD) {
                auto *f_td = new ProjectionTdFactor(pts_i, pts_j,
                                                    it_per_id.feature_per_frame[0].velocity, it_per_frame.velocity,
                                                    it_per_id.feature_per_frame[0].cur_td, it_per_frame.cur_td,
                                                    it_per_id.feature_per_frame[0].uv.y(), it_per_frame.uv.y());
                problem.AddResidualBlock(f_td, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0],
                                         para_Feature[feature_index], para_Td[0]);
            } else {
                auto *f = new ProjectionFactor(pts_i, pts_j);
                problem.AddResidualBlock(f, loss_function, para_Pose[imu_i], para_Pose[imu_j], para_Ex_Pose[0],
                                         para_Feature[feature_index]);
            }
            f_m_cnt++;
        }
    }

    LOG_D("visual measurement count: %d", f_m_cnt);
    LOG_D("prepare for ceres: %f", t_prepare.toc());

    if (relocalization_info) {
        ceres::Manifold *local_parameterization = new ceres::SE3Manifold();
        problem.AddParameterBlock(relo_Pose, SIZE_POSE, local_parameterization);
        int retrive_feature_index = 0;
        feature_index = -1;
        for (FeaturePerId &it_per_id: feature_manager.feature) {
            if (!(it_per_id.feature_per_frame.size() >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                continue;
            ++feature_index;
            int start = it_per_id.start_frame;
            if (start <= relo_frame_local_index) {
                while ((int) match_points[retrive_feature_index].z() < it_per_id.feature_id) {
                    retrive_feature_index++;
                }
                if ((int) match_points[retrive_feature_index].z() == it_per_id.feature_id) {
                    Vector3d pts_j = Vector3d(match_points[retrive_feature_index].x(),
                                              match_points[retrive_feature_index].y(), 1.0);
                    Vector3d pts_i = it_per_id.feature_per_frame[0].point;

                    auto *f = new ProjectionFactor(pts_i, pts_j);
                    problem.AddResidualBlock(f, loss_function, para_Pose[start], relo_Pose, para_Ex_Pose[0],
                                             para_Feature[feature_index]);
                    retrive_feature_index++;
                }
            }
        }
    }

    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS;
    if (marginalization_flag == MARGIN_OLD)
        options.max_solver_time_in_seconds = SOLVER_TIME * 4.0 / 5.0;
    else
        options.max_solver_time_in_seconds = SOLVER_TIME;
    TicToc t_solver;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    LOG_D("Iterations : %d", static_cast<int>(summary.iterations.size()));
    LOG_D("solver costs: %f", t_solver.toc());

    double2vector();

    TicToc t_whole_marginalization;
    if (marginalization_flag == MARGIN_OLD) {
        auto *marginalization_info = new MarginalizationInfo();
        vector2double();

        if (last_marginalization_info) {
            vector<int> drop_set;
            for (int i = 0; i < static_cast<int>(last_marginal_param_blocks.size()); i++) {
                if (last_marginal_param_blocks[i] == para_Pose[0] ||
                    last_marginal_param_blocks[i] == para_SpeedBias[0])
                    drop_set.push_back(i);
            }
            // construct new marginlization_factor
            auto *cost_function = new MarginalizationFactor(last_marginalization_info);

            ResidualBlockInfo residual_block_info(cost_function, nullptr,
                                                  last_marginal_param_blocks, drop_set);
            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        if (pre_integrations[1]->sum_dt < 10.0) {
            auto *imu_factor = new IMUFactor(pre_integrations[1]);
            vector<double *> parameter_blocks = {
                    para_Pose[0],
                    para_SpeedBias[0],
                    para_Pose[1],
                    para_SpeedBias[1]
            };
            vector<int> drop_set = {0, 1};
            ResidualBlockInfo residual_block_info(imu_factor, nullptr,
                                                              parameter_blocks, drop_set);
            marginalization_info->addResidualBlockInfo(residual_block_info);
        }

        feature_index = -1;
        for (FeaturePerId &it_per_id: feature_manager.feature) {
            if (!(it_per_id.feature_per_frame.size() >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                continue;

            ++feature_index;

            int imu_i = it_per_id.start_frame, imu_j = imu_i - 1;
            if (imu_i != 0)
                continue;

            Vector3d pts_i = it_per_id.feature_per_frame[0].point;

            for (auto &it_per_frame: it_per_id.feature_per_frame) {
                imu_j++;
                if (imu_i == imu_j)
                    continue;

                vector<int> drop_set = {0, 3};
                Vector3d pts_j = it_per_frame.point;
                if (ESTIMATE_TD) {
                    auto *f_td = new ProjectionTdFactor(pts_i, pts_j,
                                                        it_per_id.feature_per_frame[0].velocity,
                                                        it_per_frame.velocity,
                                                        it_per_id.feature_per_frame[0].cur_td,
                                                        it_per_frame.cur_td,
                                                        it_per_id.feature_per_frame[0].uv.y(),
                                                        it_per_frame.uv.y());
                    vector<double *> parameter_blocks = {
                            para_Pose[imu_i],
                            para_Pose[imu_j],
                            para_Ex_Pose[0],
                            para_Feature[feature_index],
                            para_Td[0]
                    };
                    ResidualBlockInfo residual_block_info(f_td, loss_function, parameter_blocks, drop_set);
                    marginalization_info->addResidualBlockInfo(residual_block_info);
                } else {
                    auto *f = new ProjectionFactor(pts_i, pts_j);
                    vector<double *> parameter_blocks = {
                            para_Pose[imu_i],
                            para_Pose[imu_j],
                            para_Ex_Pose[0],
                            para_Feature[feature_index],
                    };
                    ResidualBlockInfo residual_block_info(f, loss_function, parameter_blocks, drop_set);
                    marginalization_info->addResidualBlockInfo(residual_block_info);
                }
            }
        }

        TicToc t_pre_margin;
        marginalization_info->preMarginalize();
        LOG_D("pre marginalization %f ms", t_pre_margin.toc());

        TicToc t_margin;
        marginalization_info->marginalize();
        LOG_D("marginalization %f ms", t_margin.toc());

        std::unordered_map<double*, double *> addr_shift;
        for (int i = 1; i <= WINDOW_SIZE; i++) {
            addr_shift[para_Pose[i]] = para_Pose[i - 1];
            addr_shift[para_SpeedBias[i]] = para_SpeedBias[i - 1];
        }
        for (auto &i: para_Ex_Pose)
            addr_shift[i] = i;
        if (ESTIMATE_TD) {
            addr_shift[para_Td[0]] = para_Td[0];
        }
        vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);

        delete last_marginalization_info;
        last_marginalization_info = marginalization_info;
        last_marginal_param_blocks = parameter_blocks;

    } else {
        if (last_marginalization_info &&
            std::count(std::begin(last_marginal_param_blocks),
                       std::end(last_marginal_param_blocks), para_Pose[WINDOW_SIZE - 1])) {

            auto *marginalization_info = new MarginalizationInfo();
            vector2double();
            if (last_marginalization_info) {
                vector<int> drop_set;
                for (int i = 0; i < static_cast<int>(last_marginal_param_blocks.size()); i++) {
                    assert(last_marginal_param_blocks[i] != para_SpeedBias[WINDOW_SIZE - 1]);
                    if (last_marginal_param_blocks[i] == para_Pose[WINDOW_SIZE - 1])
                        drop_set.push_back(i);
                }
                // construct new marginalization_factor
                auto *marginalization_factor = new MarginalizationFactor(last_marginalization_info);
                ResidualBlockInfo residual_block_info(marginalization_factor, nullptr,
                                                      last_marginal_param_blocks, drop_set);

                marginalization_info->addResidualBlockInfo(residual_block_info);
            }

            TicToc t_pre_margin;
            LOG_D("begin marginalization");
            marginalization_info->preMarginalize();
            LOG_D("end pre marginalization, %f ms", t_pre_margin.toc());

            TicToc t_margin;
            LOG_D("begin marginalization");
            marginalization_info->marginalize();
            LOG_D("end marginalization, %f ms", t_margin.toc());

            std::unordered_map<double*, double *> addr_shift;
            for (int i = 0; i <= WINDOW_SIZE; i++) {
                if (i == WINDOW_SIZE - 1)
                    continue;
                else if (i == WINDOW_SIZE) {
                    addr_shift[para_Pose[i]] = para_Pose[i - 1];
                    addr_shift[para_SpeedBias[i]] = para_SpeedBias[i - 1];
                } else {
                    addr_shift[para_Pose[i]] = para_Pose[i];
                    addr_shift[para_SpeedBias[i]] = para_SpeedBias[i];
                }
            }
            for (Pose &i: para_Ex_Pose)
                addr_shift[i] = i;
            if (ESTIMATE_TD) {
                addr_shift[para_Td[0]] = para_Td[0];
            }

            vector<double *> parameter_blocks = marginalization_info->getParameterBlocks(addr_shift);
            delete last_marginalization_info;
            last_marginalization_info = marginalization_info;
            last_marginal_param_blocks = parameter_blocks;

        }
    }
    LOG_D("whole marginalization costs: %f", t_whole_marginalization.toc());

    LOG_D("whole time for ceres: %f", t_whole.toc());
}

void Estimator::slideWindow() {
    TicToc t_margin;
    if (marginalization_flag == MARGIN_OLD) {
        double t_0 = time_stamps[0];
        back_R0 = Rs[0];
        back_P0 = Ps[0];
        if (frame_count == WINDOW_SIZE) {
            for (int i = 0; i < WINDOW_SIZE; i++) {
                Rs[i].swap(Rs[i + 1]);

                std::swap(pre_integrations[i], pre_integrations[i + 1]);

                dt_buf[i].swap(dt_buf[i + 1]);
                linear_acceleration_buf[i].swap(linear_acceleration_buf[i + 1]);
                angular_velocity_buf[i].swap(angular_velocity_buf[i + 1]);

                time_stamps[i] = time_stamps[i + 1];
                Ps[i].swap(Ps[i + 1]);
                Vs[i].swap(Vs[i + 1]);
                Bas[i].swap(Bas[i + 1]);
                Bgs[i].swap(Bgs[i + 1]);
            }
            time_stamps[WINDOW_SIZE] = time_stamps[WINDOW_SIZE - 1];
            Ps[WINDOW_SIZE] = Ps[WINDOW_SIZE - 1];
            Vs[WINDOW_SIZE] = Vs[WINDOW_SIZE - 1];
            Rs[WINDOW_SIZE] = Rs[WINDOW_SIZE - 1];
            Bas[WINDOW_SIZE] = Bas[WINDOW_SIZE - 1];
            Bgs[WINDOW_SIZE] = Bgs[WINDOW_SIZE - 1];

            delete pre_integrations[WINDOW_SIZE];
            pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

            dt_buf[WINDOW_SIZE].clear();
            linear_acceleration_buf[WINDOW_SIZE].clear();
            angular_velocity_buf[WINDOW_SIZE].clear();

            map<double, ImageFrame>::iterator it_0;
            it_0 = all_image_frame.find(t_0);
            delete it_0->second.pre_integration;
            it_0->second.pre_integration = nullptr;
            for (auto it = all_image_frame.begin(); it != it_0; ++it) {
                delete it->second.pre_integration;
                it->second.pre_integration = nullptr;
            }
            all_image_frame.erase(all_image_frame.begin(), it_0);
            all_image_frame.erase(t_0);
            slideWindowOld();
        }
    } else if (frame_count == WINDOW_SIZE) {
        for (unsigned int i = 0; i < dt_buf[frame_count].size(); i++) {
            double tmp_dt = dt_buf[frame_count][i];
            Vector3d tmp_linear_acceleration = linear_acceleration_buf[frame_count][i];
            Vector3d tmp_angular_velocity = angular_velocity_buf[frame_count][i];

            pre_integrations[frame_count - 1]->push_back(tmp_dt, tmp_linear_acceleration, tmp_angular_velocity);

            dt_buf[frame_count - 1].push_back(tmp_dt);
            linear_acceleration_buf[frame_count - 1].push_back(tmp_linear_acceleration);
            angular_velocity_buf[frame_count - 1].push_back(tmp_angular_velocity);
        }

        time_stamps[frame_count - 1] = time_stamps[frame_count];
        Ps[frame_count - 1] = Ps[frame_count];
        Vs[frame_count - 1] = Vs[frame_count];
        Rs[frame_count - 1] = Rs[frame_count];
        Bas[frame_count - 1] = Bas[frame_count];
        Bgs[frame_count - 1] = Bgs[frame_count];

        delete pre_integrations[WINDOW_SIZE];
        pre_integrations[WINDOW_SIZE] = new IntegrationBase{acc_0, gyr_0, Bas[WINDOW_SIZE], Bgs[WINDOW_SIZE]};

        dt_buf[WINDOW_SIZE].clear();
        linear_acceleration_buf[WINDOW_SIZE].clear();
        angular_velocity_buf[WINDOW_SIZE].clear();

        slideWindowNew();
    }
}

// real marginalization is removed in solve_ceres()
void Estimator::slideWindowNew() {
    sum_of_front++;
    feature_manager.removeFront(frame_count);
}

// real marginalization is removed in solve_ceres()
void Estimator::slideWindowOld() {
    sum_of_back++;

    bool shift_depth = solver_flag == NON_LINEAR;
    Matrix3d R0, R1;
    Vector3d P0, P1;
    R0 = back_R0 * ric[0];
    R1 = Rs[0] * ric[0];
    P0 = back_P0 + back_R0 * tic[0];
    P1 = Ps[0] + Rs[0] * tic[0];
    feature_manager.removeBackShiftDepth(R0, P0, R1, P1);
}

void Estimator::setReloFrame(double _frame_stamp, int _frame_index, vector<Vector3d> &_match_points,
                             const Vector3d &_relo_t, const Matrix3d &_relo_r) {
    relo_frame_stamp = _frame_stamp;
    match_points.clear();
    match_points = _match_points;
    prev_relo_t = _relo_t;
    prev_relo_r = _relo_r;
    for (int i = 0; i < WINDOW_SIZE; i++) {
        if (relo_frame_stamp == time_stamps[i]) {
            relo_frame_local_index = i;
            relocalization_info = 1;
            for (int j = 0; j < SIZE_POSE; j++)
                relo_Pose[j] = para_Pose[i][j];
        }
    }
}

