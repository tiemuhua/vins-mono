#include "estimator.h"

Estimator::Estimator() {
    LOG_I("init begins");
    clearState();
    for (int i = 0; i <= WINDOW_SIZE; i++) {
        if (i == WINDOW_SIZE - 1)
            continue;
        else if (i == WINDOW_SIZE) {
            margin_2nd_new_addr_shift_[para_Pose[i]] = para_Pose[i - 1];
            margin_2nd_new_addr_shift_[para_SpeedBias[i]] = para_SpeedBias[i - 1];
        } else {
            margin_2nd_new_addr_shift_[para_Pose[i]] = para_Pose[i];
            margin_2nd_new_addr_shift_[para_SpeedBias[i]] = para_SpeedBias[i];
        }
    }
    margin_2nd_new_addr_shift_[para_Ex_Pose] = para_Ex_Pose;
    if (ESTIMATE_TD) {
        margin_2nd_new_addr_shift_[para_Td] = para_Td;
    }

    for (int i = 1; i <= WINDOW_SIZE; i++) {
        margin_old_addr_shift_[para_Pose[i]] = para_Pose[i - 1];
        margin_old_addr_shift_[para_SpeedBias[i]] = para_SpeedBias[i - 1];
    }
    margin_old_addr_shift_[para_Ex_Pose] = para_Ex_Pose;
    if (ESTIMATE_TD) {
        margin_old_addr_shift_[para_Td] = para_Td;
    }
}

void Estimator::setParameter() {
    ProjectionFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    ProjectionTdFactor::sqrt_info = FOCAL_LENGTH / 1.5 * Matrix2d::Identity();
    td = TD;
}

void Estimator::clearState() {
    for (int i = 0; i < WINDOW_SIZE + 1; i++) {
        rot_window[i].setIdentity();
        pos_window[i].setZero();
        vec_window[i].setZero();
        ba_window[i].setZero();
        bg_window[i].setZero();

        if (pre_integrate_window[i] != nullptr)
            delete pre_integrate_window[i];
        pre_integrate_window[i] = nullptr;
    }

    TIC = Vector3d::Zero();
    RIC = Matrix3d::Identity();

    first_imu = false,
    sum_of_back = 0;
    sum_of_front = 0;
    frame_count = 0;
    initial_timestamp = 0;
    all_image_frame.clear();
    td = TD;


    delete tmp_pre_integration;
    delete last_marginal_info_;

    tmp_pre_integration = nullptr;
    last_marginal_info_ = nullptr;
    last_marginal_param_blocks_.clear();

    feature_manager_.clearState();

    failure_occur = false;
    is_re_localization_ = false;
}

void Estimator::processIMU(double dt, const Vector3d &acc, const Vector3d &gyr) {
    if (!first_imu) {
        first_imu = true;
        acc_0 = acc;
        gyr_0 = gyr;
    }

    if (!pre_integrate_window[frame_count]) {
        pre_integrate_window[frame_count] = new PreIntegration{acc_0, gyr_0, ba_window[frame_count], bg_window[frame_count]};
    }
    if (frame_count != 0) {
        pre_integrate_window[frame_count]->predict(dt, acc, gyr);
        tmp_pre_integration->predict(dt, acc, gyr);

        int j = frame_count;
        Vector3d un_acc_0 = rot_window[j] * (acc_0 - ba_window[j]) - g;
        Vector3d un_gyr = 0.5 * (gyr_0 + gyr) - bg_window[j];
        rot_window[j] *= Utility::deltaQ(un_gyr * dt).toRotationMatrix();
        Vector3d un_acc_1 = rot_window[j] * (acc - ba_window[j]) - g;
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        pos_window[j] += dt * vec_window[j] + 0.5 * dt * dt * un_acc;
        vec_window[j] += dt * un_acc;
    }
    acc_0 = acc;
    gyr_0 = gyr;
}

void Estimator::processImage(const FeatureTracker::FeaturesPerImage &image,
                             const double &time_stamp) {
    LOG_D("new image coming ------------------------------------------");
    LOG_D("Adding feature points %lu", image.feature_ids.size());
    std::vector<FeaturePoint> feature_points;
    for (int i = 0; i < image.feature_ids.size(); ++i) {
        FeaturePoint point;
        point.unified_point = image.unified_points[i];
        point.point = image.points[i];
        point.point_velocity = image.points_velocity[i];
        point.feature_id = image.feature_ids[i];
        point.cur_td = td;
        feature_points.emplace_back(std::move(point));
    }

    bool is_key_frame = feature_manager_.addFeatureCheckParallax(frame_count, feature_points, td);

    LOG_D("is key frame:%d", is_key_frame);
    LOG_D("Solving %d", frame_count);
    LOG_D("number of feature: %d", feature_manager_.getFeatureCount());
    time_stamp_window[frame_count] = time_stamp;

    map<int, FeaturePoint> feature_id_2_points;
    for (const FeaturePoint& point: feature_points) {
        feature_id_2_points[point.feature_id] = point;
    }

    ImageFrame image_frame(feature_id_2_points, time_stamp);
    image_frame.pre_integration = tmp_pre_integration;
    all_image_frame.emplace_back(image_frame);
    tmp_pre_integration = new PreIntegration{acc_0, gyr_0, ba_window[frame_count], bg_window[frame_count]};

    if (estimate_extrinsic_state == EstimateExtrinsicInitiating) {
        LOG_I("calibrating extrinsic param, rotation movement is needed");
        if (frame_count != 0) {
            vector<pair<cv::Point2f, cv::Point2f>> correspondences =
                    feature_manager_.getCorresponding(frame_count - 1, frame_count);
            Matrix3d calib_ric;
            if (initial_ex_rotation.CalibrationExRotation(correspondences, pre_integrate_window[frame_count]->DeltaQuat(), calib_ric)) {
                LOG_W("initial extrinsic rotation calib success");
                RIC = calib_ric;
                estimate_extrinsic_state = EstimateExtrinsicInitiated;
            }
        }
    }

    if (frame_count < WINDOW_SIZE) {
        frame_count++;
        return;
    }

    if (!has_initiated_) {
        bool result = false;
        if (estimate_extrinsic_state != EstimateExtrinsicInitiating && (time_stamp - initial_timestamp) > 0.1) {
            result = initialStructure();
            initial_timestamp = time_stamp;
        }
        if (!result) {
            slideWindow(true);
            return;
        }
        has_initiated_ = true;
    }

    TicToc t_solve;
    feature_manager_.triangulate(pos_window, rot_window, TIC, RIC);
    vector2double();
    optimization();
    double2vector();

    int cnt = std::count(std::begin(last_marginal_param_blocks_),
                         std::end(last_marginal_param_blocks_), para_Pose[WINDOW_SIZE - 1]);
    if (is_key_frame) {
        marginOld();
    } else if (last_marginal_info_ && cnt) {
        margin2ndNew();
    }
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
    slideWindow(is_key_frame);
    feature_manager_.removeFailures();
    LOG_D("marginalization costs: %fms", t_margin.toc());
    // prepare output of VINS
    key_poses.clear();
    for (int i = 0; i <= WINDOW_SIZE; i++)
        key_poses.push_back(pos_window[i]);

    last_R = rot_window[WINDOW_SIZE];
    last_P = pos_window[WINDOW_SIZE];
    last_R0 = rot_window[0];
    last_P0 = pos_window[0];
}

bool Estimator::initialStructure() {
    TicToc t_sfm;
    //check imu observability
    Vector3d sum_acc;
    // todo tiemuhuaguo 原始代码很奇怪，all_image_frame隔一个用一个，而且all_image_frame.size() - 1是什么意思？
    for (const ImageFrame &frame: all_image_frame) {
        double dt = frame.pre_integration->sum_dt;
        Vector3d tmp_acc = frame.pre_integration->DeltaVel() / dt;
        sum_acc += tmp_acc;
    }
    Vector3d avg_acc = sum_acc / (double )all_image_frame.size();
    double var = 0;
    for (const ImageFrame &frame:all_image_frame) {
        double dt = frame.pre_integration->sum_dt;
        Vector3d tmp_acc = frame.pre_integration->DeltaVel() / dt;
        var += (tmp_acc - avg_acc).transpose() * (tmp_acc - avg_acc);
    }
    var = sqrt(var / (double )all_image_frame.size());
    if (var < 0.25) {
        LOG_E("IMU excitation not enough!");
        return false;
    }

    // global sfm
    map<int, Vector3d> sfm_tracked_points;
    vector<SFMFeature> sfm_features;
    for (const FeaturesOfId &features_of_id: feature_manager_.features_) {
        SFMFeature sfm_feature;
        sfm_feature.state = false;
        sfm_feature.id = features_of_id.feature_id_;
        for (int i = 0; i < features_of_id.feature_points_.size(); ++i) {
            sfm_feature.observation.emplace_back(
                    make_pair(features_of_id.start_frame_ + i, features_of_id.feature_points_[i].unified_point));
        }
        sfm_features.push_back(sfm_feature);
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
    if (!GlobalSFM::construct(frame_count + 1, Q, T, l, relative_R, relative_T, sfm_features, sfm_tracked_points)) {
        LOG_D("global SFM failed!");
        return false;
    }

    //solve pnp for all frame
    int i = 0;
    for (ImageFrame &frame:all_image_frame) {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        if (frame.t == time_stamp_window[i]) {
            frame.is_key_frame = true;
            frame.R = Q[i].toRotationMatrix() * RIC.transpose();
            frame.T = T[i];
            i++;
            continue;
        }
        if ((frame.t) > time_stamp_window[i]) {
            i++;
        }
        Matrix3d R_initial = (Q[i].inverse()).toRotationMatrix();
        Vector3d P_initial = -R_initial * T[i];
        cv::eigen2cv(R_initial, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(P_initial, t);

        frame.is_key_frame = false;
        vector<cv::Point3f> pts_3_vector;
        vector<cv::Point2f> pts_2_vector;
        for (auto &id_pts: frame.feature_id_2_point) {
            int feature_id = id_pts.first;
            auto it = sfm_tracked_points.find(feature_id);
            Vector3d world_pts = it->second;
            if (it != sfm_tracked_points.end()) {
                pts_3_vector.emplace_back(world_pts(0), world_pts(1), world_pts(2));
                pts_2_vector.emplace_back(id_pts.second.unified_point);
            }
        }
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        if (pts_3_vector.size() < 6) {
            LOG_D("pts_3_vector size:%lu, Not enough points for solve pnp !", pts_3_vector.size());
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
        frame.T = R_pnp * (-T_pnp);
        frame.R = R_pnp * RIC.transpose();
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
    bool result = VisualIMUAlignment(all_image_frame, bg_window, g, x);
    if (!result) {
        LOG_D("solve g failed!");
        return false;
    }

    // change state
    for (int i = 0; i <= frame_count; i++) {
        Matrix3d Ri = all_image_frame[i].R;
        Vector3d Pi = all_image_frame[i].T;
        pos_window[i] = Pi;
        rot_window[i] = Ri;
        all_image_frame[i].is_key_frame = true;
    }

    VectorXd dep = feature_manager_.getDepthVector();
    for (int i = 0; i < dep.size(); i++)
        dep[i] = -1;
    feature_manager_.clearDepth(dep);

    //triangulate on cam pose , no tic
    feature_manager_.triangulate(pos_window, rot_window, Vector3d::Zero(), RIC);

    double s = (x.tail<1>())(0);
    for (int i = 0; i <= WINDOW_SIZE; i++) {
        pre_integrate_window[i]->rePrediction(Vector3d::Zero(), bg_window[i]);
    }
    for (int i = frame_count; i >= 0; i--)
        pos_window[i] = s * pos_window[i] - rot_window[i] * TIC[0] - (s * pos_window[0] - rot_window[0] * TIC[0]);
    int kv = -1;
    for (const auto &frame:all_image_frame) {
        if (frame.is_key_frame) {
            kv++;
            vec_window[kv] = frame.R * x.segment<3>(kv * 3);
        }
    }
    for (FeaturesOfId &features_of_id: feature_manager_.features_) {
        if (!(features_of_id.feature_points_.size() >= 2 && features_of_id.start_frame_ < WINDOW_SIZE - 2))
            continue;
        features_of_id.estimated_depth *= s;
    }

    Matrix3d R0 = Utility::g2R(g);
    double yaw = Utility::R2ypr(R0 * rot_window[0]).x();
    R0 = Utility::ypr2R(Eigen::Vector3d{-yaw, 0, 0}) * R0;
    g = R0 * g;
    Matrix3d rot_diff = R0;
    for (int i = 0; i <= frame_count; i++) {
        pos_window[i] = rot_diff * pos_window[i];
        rot_window[i] = rot_diff * rot_window[i];
        vec_window[i] = rot_diff * vec_window[i];
    }

    return true;
}

bool Estimator::relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l) {
    // find previous frame which contains enough correspondence and parallax with the newest frame
    for (int i = 0; i < WINDOW_SIZE; i++) {
        vector<pair<cv::Point2f , cv::Point2f>> correspondences = feature_manager_.getCorresponding(i, WINDOW_SIZE);
        if (correspondences.size() <= 20) {
            continue;
        }
        double sum_parallax = 0;
        for (auto &correspond: correspondences) {
            double parallax = norm(correspond.first - correspond.second);
            sum_parallax += parallax;
        }
        double average_parallax = 1.0 * sum_parallax / int(correspondences.size());
        if (average_parallax * 460 > 30 && MotionEstimator::solveRelativeRT(correspondences, relative_R, relative_T)) {
            l = i;
            LOG_D("average_parallax %f choose l %d and newest frame to triangulate the whole structure",
                  average_parallax * 460, l);
            return true;
        }
    }
    return false;
}

void Estimator::solveOdometry() {
    if (frame_count < WINDOW_SIZE)
        return;
    if (has_initiated_) {
    }
}

void Estimator::vector2double() {
    for (int i = 0; i <= WINDOW_SIZE; i++) {
        para_Pose[i][0] = pos_window[i].x();
        para_Pose[i][1] = pos_window[i].y();
        para_Pose[i][2] = pos_window[i].z();
        Quaterniond q{rot_window[i]};
        para_Pose[i][3] = q.x();
        para_Pose[i][4] = q.y();
        para_Pose[i][5] = q.z();
        para_Pose[i][6] = q.w();

        para_SpeedBias[i][0] = vec_window[i].x();
        para_SpeedBias[i][1] = vec_window[i].y();
        para_SpeedBias[i][2] = vec_window[i].z();

        para_SpeedBias[i][3] = ba_window[i].x();
        para_SpeedBias[i][4] = ba_window[i].y();
        para_SpeedBias[i][5] = ba_window[i].z();

        para_SpeedBias[i][6] = bg_window[i].x();
        para_SpeedBias[i][7] = bg_window[i].y();
        para_SpeedBias[i][8] = bg_window[i].z();
    }
    para_Ex_Pose[0] = TIC.x();
    para_Ex_Pose[1] = TIC.y();
    para_Ex_Pose[2] = TIC.z();
    Quaterniond q{RIC};
    para_Ex_Pose[3] = q.x();
    para_Ex_Pose[4] = q.y();
    para_Ex_Pose[5] = q.z();
    para_Ex_Pose[6] = q.w();

    VectorXd dep = feature_manager_.getDepthVector();
    for (int i = 0; i < feature_manager_.getFeatureCount(); i++)
        para_Feature[i][0] = dep(i);
    if (ESTIMATE_TD)
        para_Td[0] = td;
}

void Estimator::double2vector() {
    Vector3d origin_R0 = Utility::R2ypr(rot_window[0]);
    Vector3d origin_P0 = pos_window[0];

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
        LOG_D("euler singular point_!");
        rot_diff = rot_window[0] * Quaterniond(para_Pose[0][6],
                                               para_Pose[0][3],
                                               para_Pose[0][4],
                                               para_Pose[0][5]).toRotationMatrix().transpose();
    }

    for (int i = 0; i <= WINDOW_SIZE; i++) {

        rot_window[i] = rot_diff * Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4],
                                               para_Pose[i][5]).normalized().toRotationMatrix();

        pos_window[i] = rot_diff * Vector3d(para_Pose[i][0] - para_Pose[0][0],
                                    para_Pose[i][1] - para_Pose[0][1],
                                    para_Pose[i][2] - para_Pose[0][2]) + origin_P0;

        vec_window[i] = rot_diff * Vector3d(para_SpeedBias[i][0],
                                            para_SpeedBias[i][1],
                                            para_SpeedBias[i][2]);

        ba_window[i] = Vector3d(para_SpeedBias[i][3],
                                para_SpeedBias[i][4],
                                para_SpeedBias[i][5]);

        bg_window[i] = Vector3d(para_SpeedBias[i][6],
                                para_SpeedBias[i][7],
                                para_SpeedBias[i][8]);
    }
    TIC = Vector3d(para_Ex_Pose[0],para_Ex_Pose[1],para_Ex_Pose[2]);
    RIC = Quaterniond(para_Ex_Pose[6],para_Ex_Pose[3],para_Ex_Pose[4],para_Ex_Pose[5]).toRotationMatrix();

    VectorXd dep = feature_manager_.getDepthVector();
    for (int i = 0; i < feature_manager_.getFeatureCount(); i++)
        dep(i) = para_Feature[i][0];
    feature_manager_.setDepth(dep);
    if (ESTIMATE_TD)
        td = para_Td[0];

    // relative info between two loop frame
    if (is_re_localization_) {
        Matrix3d re_local_r = rot_diff *
                 Quaterniond(re_local_Pose[6], re_local_Pose[3], re_local_Pose[4], re_local_Pose[5]).normalized().toRotationMatrix();
        is_re_localization_ = false;
    }
}

bool Estimator::failureDetection() {
    if (feature_manager_.last_track_num < 2) {
        LOG_I(" little feature %d", feature_manager_.last_track_num);
        //return true;
    }
    if (ba_window[WINDOW_SIZE].norm() > 2.5) {
        LOG_I(" big IMU acc bias estimation %f", ba_window[WINDOW_SIZE].norm());
        return true;
    }
    if (bg_window[WINDOW_SIZE].norm() > 1.0) {
        LOG_I(" big IMU gyr bias estimation %f", bg_window[WINDOW_SIZE].norm());
        return true;
    }
    Vector3d tmp_P = pos_window[WINDOW_SIZE];
    if ((tmp_P - last_P).norm() > 5) {
        LOG_I(" big translation");
        return true;
    }
    if (abs(tmp_P.z() - last_P.z()) > 1) {
        LOG_I(" big z translation");
        return true;
    }
    Matrix3d tmp_R = rot_window[WINDOW_SIZE];
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
    // todo tiemuhuaguo 四元数顺序待确认
    typedef ProductManifold<EuclideanManifold<3>, QuaternionManifold> SE3Manifold;
}

void Estimator::optimization() {
    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::CauchyLoss(1.0);
    for (int i = 0; i < WINDOW_SIZE + 1; i++) {
        ceres::Manifold *manifold = new ceres::SE3Manifold();
        problem.AddParameterBlock(para_Pose[i], SIZE_POSE, manifold);
        problem.AddParameterBlock(para_SpeedBias[i], SIZE_SPEED_BIAS);
    }
    ceres::Manifold *manifold = new ceres::SE3Manifold();
    problem.AddParameterBlock(para_Ex_Pose, SIZE_POSE, manifold);
    if (!estimate_extrinsic_state == EstimateExtrinsicFix) {
        LOG_D("fix extrinsic param");
        problem.SetParameterBlockConstant(para_Ex_Pose);
    } else {
        LOG_D("estimate extrinsic param");
    }
    if (ESTIMATE_TD) {
        problem.AddParameterBlock(para_Td, 1);
    }

    TicToc t_whole, t_prepare;

    if (last_marginal_info_) {
        auto *cost_function = new MarginalFactor(last_marginal_info_);
        problem.AddResidualBlock(cost_function, nullptr,last_marginal_param_blocks_);
    }

    for (int i = 0; i < WINDOW_SIZE; i++) {
        int j = i + 1;
        if (pre_integrate_window[j]->sum_dt > 10.0)
            continue;
        auto *cost_function = new IMUFactor(pre_integrate_window[j]);
        problem.AddResidualBlock(cost_function, nullptr,
                                 para_Pose[i], para_SpeedBias[i], para_Pose[j], para_SpeedBias[j]);
    }
    int f_m_cnt = 0;
    int feature_index = -1;
    for (FeaturesOfId &features_of_id: feature_manager_.features_) {
        if (features_of_id.feature_points_.size() < 2 || features_of_id.start_frame_ >= WINDOW_SIZE - 2)
            continue;

        ++feature_index;

        int start_frame = features_of_id.start_frame_;
        const FeaturePoint & point0 = features_of_id.feature_points_[0];
        for (int i = 1; i < features_of_id.feature_points_.size(); ++i) {
            FeaturePoint &point = features_of_id.feature_points_[start_frame + i];
            if (ESTIMATE_TD) {
                auto *cost_function = new ProjectionTdFactor(point0.unified_point, point.unified_point,
                                                             point0.point_velocity, point.point_velocity,
                                                             point0.cur_td, point.cur_td,
                                                             point0.point.y, point.point.y);
                problem.AddResidualBlock(cost_function, loss_function,
                                         para_Pose[start_frame], para_Pose[i], para_Ex_Pose, para_Feature[feature_index], para_Td);
            } else {
                auto *cost_function = new ProjectionFactor(point0.unified_point, point.unified_point);
                problem.AddResidualBlock(cost_function, loss_function,
                                         para_Pose[start_frame], para_Pose[i], para_Ex_Pose, para_Feature[feature_index]);
            }
            f_m_cnt++;
        }
    }

    LOG_D("visual measurement count: %d", f_m_cnt);
    LOG_D("prepare for ceres: %f", t_prepare.toc());

    if (is_re_localization_) {
        ceres::Manifold *local_parameterization = new ceres::SE3Manifold();
        problem.AddParameterBlock(re_local_Pose, SIZE_POSE, local_parameterization);
        int retrive_feature_index = 0;
        feature_index = -1;
        for (FeaturesOfId &features_of_id: feature_manager_.features_) {
            if (features_of_id.feature_points_.size() < 2 || features_of_id.start_frame_ >= WINDOW_SIZE - 2)
                continue;
            ++feature_index;
            int start = features_of_id.start_frame_;
            if (start <= re_local_frame_local_index) {
                while ((int) match_points_[retrive_feature_index].feature_id < features_of_id.feature_id_) {
                    retrive_feature_index++;
                }
                if (match_points_[retrive_feature_index].feature_id == features_of_id.feature_id_) {
                    auto *cost_function = new ProjectionFactor(match_points_[retrive_feature_index].point,
                                                               features_of_id.feature_points_[0].unified_point);
                    problem.AddResidualBlock(cost_function, loss_function,
                                             para_Pose[start], re_local_Pose, para_Ex_Pose, para_Feature[feature_index]);
                    retrive_feature_index++;
                }
            }
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = NUM_ITERATIONS;
    TicToc t_solver;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    LOG_D("Iterations : %d, solver costs: %f", static_cast<int>(summary.iterations.size()), t_solver.toc());
}

void Estimator::margin2ndNew() {
    auto *marginal_info = new MarginalInfo();
    assert(last_marginal_info_);
    vector<int> drop_set;
    for (int i = 0; i < static_cast<int>(last_marginal_param_blocks_.size()); i++) {
        assert(last_marginal_param_blocks_[i] != para_SpeedBias[WINDOW_SIZE - 1]);
        if (last_marginal_param_blocks_[i] == para_Pose[WINDOW_SIZE - 1]){
            drop_set.push_back(i);
        }
    }
    // construct new marginalization_factor
    auto *marginalization_factor = new MarginalFactor(last_marginal_info_);
    ResidualBlockInfo residual_block_info(marginalization_factor, nullptr,
                                          last_marginal_param_blocks_, drop_set);

    marginal_info->addResidualBlockInfo(residual_block_info);
    marginal_info->preMarginalize();
    marginal_info->marginalize();

    vector<double *> parameter_blocks = marginal_info->getParameterBlocks(margin_2nd_new_addr_shift_);
    delete last_marginal_info_;
    last_marginal_info_ = marginal_info;
    last_marginal_param_blocks_ = parameter_blocks;
}

void Estimator::marginOld() {
    auto *marginal_info = new MarginalInfo();

    // 原来的约束
    if (last_marginal_info_) {
        vector<int> drop_set;
        for (int i = 0; i < static_cast<int>(last_marginal_param_blocks_.size()); i++) {
            if (last_marginal_param_blocks_[i] == para_Pose[0] ||
                last_marginal_param_blocks_[i] == para_SpeedBias[0])
                drop_set.push_back(i);
        }
        // construct new marginal_factor
        auto *cost_function = new MarginalFactor(last_marginal_info_);

        ResidualBlockInfo residual_block_info(cost_function, nullptr,
                                              last_marginal_param_blocks_, drop_set);
        marginal_info->addResidualBlockInfo(residual_block_info);
    }

    // 新的陀螺仪约束
    if (pre_integrate_window[1]->sum_dt < 10.0) {// todo tiemuhuaguo 1这个硬编码是怎么来的？
        auto *cost_function = new IMUFactor(pre_integrate_window[1]);
        vector<double *> parameter_blocks = {
                para_Pose[0],
                para_SpeedBias[0],
                para_Pose[1],
                para_SpeedBias[1]
        };
        vector<int> drop_set = {0, 1};
        ResidualBlockInfo residual_block_info(cost_function, nullptr,
                                              parameter_blocks, drop_set);
        marginal_info->addResidualBlockInfo(residual_block_info);
    }

    int feature_index = -1;
    ceres::LossFunction *loss_function = new ceres::CauchyLoss(1.0);
    for (const FeaturesOfId &features_of_id: feature_manager_.features_) {
        if (features_of_id.feature_points_.size() < 2 || features_of_id.start_frame_ >= WINDOW_SIZE - 2)
            continue;

        ++feature_index;

        if (features_of_id.start_frame_ != 0) // todo tiemuhuaguo ????
            continue;

        const FeaturePoint & point0 = features_of_id.feature_points_[0];
        for (int i = 1; i < features_of_id.feature_points_.size(); ++i) {
            const FeaturePoint &point = features_of_id.feature_points_[i];
            vector<int> drop_set = {0, 3};
            if (ESTIMATE_TD) {
                auto *cost_function = new ProjectionTdFactor(point0.unified_point,point.unified_point,
                                                             point0.point_velocity,point.point_velocity,
                                                             point0.cur_td,point.cur_td,
                                                             point0.point.y,point.point.y);
                vector<double *> parameter_blocks = {
                        para_Pose[0],
                        para_Pose[i],
                        para_Ex_Pose,
                        para_Feature[feature_index],
                        para_Td
                };
                ResidualBlockInfo residual_block_info(cost_function, loss_function, parameter_blocks, drop_set);
                marginal_info->addResidualBlockInfo(residual_block_info);
            } else {
                auto *cost_function = new ProjectionFactor(point0.unified_point, point.unified_point);
                vector<double *> parameter_blocks = {
                        para_Pose[0],
                        para_Pose[i],
                        para_Ex_Pose,
                        para_Feature[feature_index],
                };
                ResidualBlockInfo residual_block_info(cost_function, loss_function, parameter_blocks, drop_set);
                marginal_info->addResidualBlockInfo(residual_block_info);
            }
        }
    }

    marginal_info->preMarginalize();
    marginal_info->marginalize();

    vector<double *> parameter_blocks = marginal_info->getParameterBlocks(margin_old_addr_shift_);
    delete last_marginal_info_;
    last_marginal_info_ = marginal_info;
    last_marginal_param_blocks_ = parameter_blocks;
}

void Estimator::slideWindow(bool is_key_frame) {
    if (frame_count != WINDOW_SIZE) {
        return;
    }
    if (is_key_frame) {
        double t_0 = time_stamp_window[0];
        for (int i = 0; i < WINDOW_SIZE; i++) {
            rot_window[i].swap(rot_window[i + 1]);

            std::swap(pre_integrate_window[i], pre_integrate_window[i + 1]);

            time_stamp_window[i] = time_stamp_window[i + 1];
            pos_window[i].swap(pos_window[i + 1]);
            vec_window[i].swap(vec_window[i + 1]);
            ba_window[i].swap(ba_window[i + 1]);
            bg_window[i].swap(bg_window[i + 1]);
        }

        auto it = std::find(all_image_frame.begin(), all_image_frame.end(), [t_0](const ImageFrame&it)->bool {
            return it.t == t_0;
        });
        vector<ImageFrame> tmp_all_image_frame(it++, all_image_frame.end());
        all_image_frame = std::move(tmp_all_image_frame);
        sum_of_back++;
        bool shift_depth = has_initiated_;
        feature_manager_.removeBackShiftDepth(); // todo 只有初始化成功后才remove吗
    } else {
        for (int i = 0; i < (int)pre_integrate_window[WINDOW_SIZE]->dt_buf.size(); ++i) {
            double dt = pre_integrate_window[WINDOW_SIZE]->dt_buf[i];
            Vector3d acc = pre_integrate_window[WINDOW_SIZE]->acc_buf[i];
            Vector3d gyr = pre_integrate_window[WINDOW_SIZE]->gyr_buf[i];
            pre_integrate_window[WINDOW_SIZE - 1]->predict(dt, acc, gyr);
        }
        sum_of_front++;
        feature_manager_.removeFront(frame_count);
    }

    time_stamp_window[WINDOW_SIZE] = time_stamp_window[WINDOW_SIZE - 1];
    pos_window[WINDOW_SIZE] = pos_window[WINDOW_SIZE - 1];
    vec_window[WINDOW_SIZE] = vec_window[WINDOW_SIZE - 1];
    rot_window[WINDOW_SIZE] = rot_window[WINDOW_SIZE - 1];
    ba_window[WINDOW_SIZE] = ba_window[WINDOW_SIZE - 1];
    bg_window[WINDOW_SIZE] = bg_window[WINDOW_SIZE - 1];

    delete pre_integrate_window[WINDOW_SIZE];
    pre_integrate_window[WINDOW_SIZE] = new PreIntegration{acc_0, gyr_0, ba_window[WINDOW_SIZE], bg_window[WINDOW_SIZE]};
}

void Estimator::setReLocalFrame(double _frame_stamp, int _frame_index, vector<MatchPoint> &_match_points,
                                const Vector3d &_re_local_t, const Matrix3d &_re_local_r) {
    match_points_ = _match_points;
    for (int i = 0; i < WINDOW_SIZE; i++) {
        if (_frame_stamp == time_stamp_window[i]) {
            re_local_frame_local_index = i;
            is_re_localization_ = true;
            for (int j = 0; j < SIZE_POSE; j++)
                re_local_Pose[j] = para_Pose[i][j];
        }
    }
}

