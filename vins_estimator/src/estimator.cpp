#include "estimator.h"

Estimator::Estimator() {
    LOG_I("init begins");
    clearState();
    for (int i = 0; i <= WINDOW_SIZE; i++) {
        if (i == WINDOW_SIZE - 1)
            continue;
        else if (i == WINDOW_SIZE) {
            margin_2nd_new_addr_shift_[para_Pose[i]] = para_Pose[i - 1];
            margin_2nd_new_addr_shift_[para_Velocity[i]] = para_Velocity[i - 1];
            margin_2nd_new_addr_shift_[para_AccBias[i]] = para_AccBias[i - 1];
            margin_2nd_new_addr_shift_[para_GyrBias[i]] = para_GyrBias[i - 1];
        } else {
            margin_2nd_new_addr_shift_[para_Pose[i]] = para_Pose[i];
            margin_2nd_new_addr_shift_[para_Velocity[i]] = para_Velocity[i];
            margin_2nd_new_addr_shift_[para_AccBias[i]] = para_AccBias[i];
            margin_2nd_new_addr_shift_[para_GyrBias[i]] = para_GyrBias[i];
        }
    }
    margin_2nd_new_addr_shift_[para_Ex_Pose] = para_Ex_Pose;
    if (ESTIMATE_TD) {
        margin_2nd_new_addr_shift_[para_Td] = para_Td;
    }

    for (int i = 1; i <= WINDOW_SIZE; i++) {
        margin_old_addr_shift_[para_Pose[i]] = para_Pose[i - 1];
        margin_old_addr_shift_[para_Velocity[i]] = para_Velocity[i - 1];
        margin_old_addr_shift_[para_AccBias[i]] = para_AccBias[i - 1];
        margin_old_addr_shift_[para_GyrBias[i]] = para_GyrBias[i - 1];
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
        vel_window[i].setZero();
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
    frame_count_ = 0;
    initial_timestamp_ = 0;
    all_image_frame_.clear();
    td = TD;

    delete last_marginal_info_;
    last_marginal_info_ = nullptr;
    last_marginal_param_blocks_.clear();

    feature_manager_.clearState();

    failure_occur = false;
    is_re_localization_ = false;
}

void Estimator::processIMU(double dt, const Vector3d &acc, const Vector3d &gyr) {
    first_imu = true;
    acc_0 = acc;
    gyr_0 = gyr;
    if (all_image_frame_.empty()) {
        return;
    }
    all_image_frame_.back().pre_integrate_.predict(dt, acc, gyr);
}

void Estimator::processImage(const FeatureTracker::FeaturesPerImage &image,
                             const double &time_stamp) {
    LOG_D("Adding feature points %lu", image.feature_ids.size());
    std::vector<FeaturePoint> feature_points;
    for (int i = 0; i < image.feature_ids.size(); ++i) {
        FeaturePoint point;
        point.unified_point = image.unified_points[i];
        point.point = image.points[i];
        point.point_velocity = image.points_velocity[i];
        point.feature_id = image.feature_ids[i];
        feature_points.emplace_back(std::move(point));
    }

    bool is_key_frame = feature_manager_.addFeatureCheckParallax(frame_count_, feature_points, td);

    LOG_D("is key frame:%d, Solving %d, number of feature: %d",
          is_key_frame, frame_count_, feature_manager_.getFeatureCount());
    time_stamp_window[frame_count_] = time_stamp;

    PreIntegration pre_int(time_stamp, acc_0, gyr_0, ba_window[frame_count_], bg_window[frame_count_]);
    ImageFrame image_frame(std::move(feature_points),
                           time_stamp,
                           std::move(pre_int),
                           is_key_frame);
    all_image_frame_.emplace_back(std::move(image_frame));

    if (estimate_extrinsic_state == EstimateExtrinsicInitiating) {
        LOG_I("calibrating extrinsic param, rotation movement is needed");
        if (frame_count_ != 0) {
            vector<pair<cv::Point2f, cv::Point2f>> correspondences =
                    feature_manager_.getCorresponding(frame_count_ - 1, frame_count_);
            Matrix3d calib_ric;
            if (initial_ex_rotation.calibrateRotationExtrinsic(correspondences,
                                                               pre_integrate_window[frame_count_]->DeltaQuat(),
                                                               calib_ric)) {
                LOG_I("initial extrinsic rotation calib success");
                RIC = calib_ric;
                estimate_extrinsic_state = EstimateExtrinsicInitiated;
            }
        }
    }

    if (frame_count_ < WINDOW_SIZE) {
        frame_count_++;
        return;
    }

    if (!has_initiated_) {
        bool result = false;
        if (estimate_extrinsic_state != EstimateExtrinsicInitiating && (time_stamp - initial_timestamp_) > 0.1) {
            result = initialStructure();
            if (result) {
                result = visualInitialAlign(all_image_frame_, gravity_, frame_count_,
                                            bg_window, pos_window, rot_window,vel_window, pre_integrate_window,
                                            feature_manager_);
            }
            initial_timestamp_ = time_stamp;
        }
        if (!result) {
            slideWindow(true);
            return;
        }
        has_initiated_ = true;
    }

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

    if (failureDetection()) {
        LOG_W("failure detection!, system reboot!");
        failure_occur = true;
        clearState();
        setParameter();
        return;
    }

    slideWindow(is_key_frame);
    feature_manager_.removeFailures();
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
    for (const ImageFrame &frame: all_image_frame_) {
        double dt = frame.pre_integrate_.sum_dt;
        Vector3d tmp_acc = frame.pre_integrate_.DeltaVel() / dt;
        sum_acc += tmp_acc;
    }
    Vector3d avg_acc = sum_acc / (double )all_image_frame_.size();
    double var = 0;
    for (const ImageFrame &frame:all_image_frame_) {
        double dt = frame.pre_integrate_.sum_dt;
        Vector3d tmp_acc = frame.pre_integrate_.DeltaVel() / dt;
        var += (tmp_acc - avg_acc).transpose() * (tmp_acc - avg_acc);
    }
    var = sqrt(var / (double )all_image_frame_.size());
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
    Quaterniond Q[frame_count_ + 1];
    Vector3d T[frame_count_ + 1];
    if (!GlobalSFM::construct(frame_count_ + 1, Q, T, l, relative_R, relative_T, sfm_features, sfm_tracked_points)) {
        LOG_D("global SFM failed!");
        return false;
    }

    //solve pnp for all frame
    int i = 0;
    for (ImageFrame &frame:all_image_frame_) {
        // provide initial guess
        cv::Mat r, rvec, t, D, tmp_r;
        if (frame.t == time_stamp_window[i]) {
            frame.is_key_frame_ = true;
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

        frame.is_key_frame_ = false;
        vector<cv::Point3f> pts_3_vector;
        vector<cv::Point2f> pts_2_vector;
        for (const FeaturePoint &point:frame.points) {
            int feature_id = point.feature_id;
            const auto it = sfm_tracked_points.find(feature_id);
            if (it != sfm_tracked_points.end()) {
                Vector3d world_pts = it->second;
                pts_3_vector.emplace_back(world_pts(0), world_pts(1), world_pts(2));
                pts_2_vector.emplace_back(point.unified_point);
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

void EigenPose2Double(const Vector3d& t, const Matrix3d& r, Pose pose) {
    pose[0] = t.x();
    pose[1] = t.y();
    pose[2] = t.z();
    Quaterniond q(r);
    pose[3] = q.x();
    pose[4] = q.y();
    pose[5] = q.z();
    pose[6] = q.w();
}
void EigenVector3d2Double(const Vector3d& t, double arr[3]) {
    arr[0] = t.x();
    arr[1] = t.y();
    arr[2] = t.z();
}

void Estimator::vector2double() {
    for (int i = 0; i <= WINDOW_SIZE; i++) {
        EigenPose2Double(pos_window[i], rot_window[i], para_Pose[i]);
        EigenVector3d2Double(vel_window[i], para_Velocity[i]);
        EigenVector3d2Double(ba_window[i], para_AccBias[i]);
        EigenVector3d2Double(bg_window[i], para_GyrBias[i]);
    }
    EigenPose2Double(TIC, RIC, para_Ex_Pose);

    VectorXd inv_depth = feature_manager_.getInvDepthVector();
    for (int i = 0; i < inv_depth.size(); i++)
        para_FeatureInvDepth[i][0] = inv_depth(i);
    if (ESTIMATE_TD)
        para_Td[0] = td;
}

Matrix3d Pose2Mat(Pose pose) {
    return Quaterniond(pose[6],pose[3],pose[4],pose[5]).normalized().toRotationMatrix();
}
Vector3d Pose2Vec3(double arr[3]) {
    return {arr[0], arr[1], arr[2]};
}
void Estimator::double2vector() {
    Vector3d origin_R0 = Utility::R2ypr(rot_window[0]);
    Vector3d origin_P0 = pos_window[0];

    if (failure_occur) {
        origin_R0 = Utility::R2ypr(last_R0);
        origin_P0 = last_P0;
        failure_occur = false;
    }
    Vector3d origin_R00 = Utility::R2ypr(Pose2Mat(para_Pose[0]));
    double y_diff = origin_R0.x() - origin_R00.x();

    Matrix3d rot_diff = Utility::ypr2R(Vector3d(y_diff, 0, 0)); // todo 假设只会有yaw的移动？？？
    if (abs(abs(origin_R0.y()) - 90) < 1.0 || abs(abs(origin_R00.y()) - 90) < 1.0) {
        LOG_D("euler singular point_!");
        rot_diff = rot_window[0] * Pose2Mat(para_Pose[0]).transpose();
    }

    for (int i = 0; i <= WINDOW_SIZE; i++) {
        rot_window[i] = rot_diff * Pose2Mat(para_Pose[i]);
        pos_window[i] = rot_diff * (Pose2Vec3(para_Pose[i]) - Pose2Vec3(para_Pose[0])) + origin_P0;
        vel_window[i] = rot_diff * Pose2Vec3(para_Velocity[i]);
        ba_window[i] = Pose2Vec3(para_AccBias[i]);
        bg_window[i] = Pose2Vec3(para_GyrBias[i]);
    }
    TIC = Vector3d(para_Ex_Pose[0],para_Ex_Pose[1],para_Ex_Pose[2]);
    RIC = Pose2Mat(para_Ex_Pose);

    VectorXd inv_dep(feature_manager_.getFeatureCount());
    for (int i = 0; i < inv_dep.size(); i++)
        inv_dep(i) = para_FeatureInvDepth[i][0];
    feature_manager_.setInvDepth(inv_dep);
    if (ESTIMATE_TD)
        td = para_Td[0];

    // relative info between two loop frame
    if (is_re_localization_) {
        Matrix3d re_local_r = rot_diff * Pose2Mat(re_local_Pose);
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
        problem.AddParameterBlock(para_Pose[i], 7, manifold);
        problem.AddParameterBlock(para_Velocity[i], 3);
        problem.AddParameterBlock(para_AccBias[i], 3);
        problem.AddParameterBlock(para_GyrBias[i], 3);
    }
    ceres::Manifold *manifold = new ceres::SE3Manifold();
    problem.AddParameterBlock(para_Ex_Pose, 7, manifold);
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

    /*************** 1:边缘化 **************************/
    if (last_marginal_info_) {
        auto *cost_function = new MarginalFactor(last_marginal_info_);
        problem.AddResidualBlock(cost_function, nullptr,last_marginal_param_blocks_);
    }

    /*************** 2:IMU **************************/
    for (int i = 0; i < WINDOW_SIZE; i++) {
        int j = i + 1;
        if (pre_integrate_window[j]->sum_dt > 10.0)// todo why???
            continue;
        auto *cost_function = new IMUFactor(pre_integrate_window[j]);
        problem.AddResidualBlock(cost_function, nullptr,
                                 para_Pose[i], para_Velocity[i], para_AccBias[i], para_GyrBias[i],
                                 para_Pose[j], para_Velocity[j], para_AccBias[j], para_GyrBias[j]);
    }

    /*************** 3:特征点 **************************/
    int f_m_cnt = 0;
    int feature_index = -1;
    for (FeaturesOfId &features_of_id: feature_manager_.features_) {
        if (features_of_id.feature_points_.size() < 2 || features_of_id.start_frame_ >= WINDOW_SIZE - 2)
            continue;

        ++feature_index;

        int start_frame = features_of_id.start_frame_;
        const FeaturePoint & point0 = features_of_id.feature_points_[0];
        for (int i = 1; i < features_of_id.feature_points_.size(); ++i) {
            FeaturePoint &point = features_of_id.feature_points_[i];
            if (ESTIMATE_TD) {
                auto *cost_function = new ProjectionTdFactor(point0, point);
                problem.AddResidualBlock(cost_function, loss_function,
                                         para_Pose[start_frame], para_Pose[start_frame + i], para_Ex_Pose, para_FeatureInvDepth[feature_index], para_Td);
            } else {
                auto *cost_function = new ProjectionFactor(point0.unified_point, point.unified_point);
                problem.AddResidualBlock(cost_function, loss_function,
                                         para_Pose[start_frame], para_Pose[start_frame + i], para_Ex_Pose, para_FeatureInvDepth[feature_index]);
            }
            f_m_cnt++;
        }
    }

    LOG_D("visual measurement count: %d, prepare for ceres: %f", f_m_cnt, t_prepare.toc());

    /*************** 4:回环 **************************/
    if (is_re_localization_) {
        ceres::Manifold *local_parameterization = new ceres::SE3Manifold();
        problem.AddParameterBlock(re_local_Pose, 7, local_parameterization);
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
                                             para_Pose[start], re_local_Pose, para_Ex_Pose, para_FeatureInvDepth[feature_index]);
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
                last_marginal_param_blocks_[i] == para_Velocity[0]) // todo what it means???
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
                para_Pose[0], para_Velocity[0], para_AccBias[0], para_GyrBias[0],
                para_Pose[1], para_Velocity[1], para_AccBias[1], para_GyrBias[1],
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
                auto *cost_function = new ProjectionTdFactor(point0, point);
                vector<double *> parameter_blocks = {
                        para_Pose[0],
                        para_Pose[i],
                        para_Ex_Pose,
                        para_FeatureInvDepth[feature_index],
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
                        para_FeatureInvDepth[feature_index],
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
    assert(frame_count_ == WINDOW_SIZE);
    if (is_key_frame) {
        double t_0 = time_stamp_window[0];
        for (int i = 0; i < WINDOW_SIZE; i++) {
            rot_window[i].swap(rot_window[i + 1]);
            std::swap(pre_integrate_window[i], pre_integrate_window[i + 1]);
            time_stamp_window[i] = time_stamp_window[i + 1];
            pos_window[i].swap(pos_window[i + 1]);
            vel_window[i].swap(vel_window[i + 1]);
            ba_window[i].swap(ba_window[i + 1]);
            bg_window[i].swap(bg_window[i + 1]);
        }

        auto it = std::find(all_image_frame_.begin(), all_image_frame_.end(), [t_0](const ImageFrame&it)->bool {
            return it.t == t_0;
        });
        vector<ImageFrame> tmp_all_image_frame(it++, all_image_frame_.end());
        all_image_frame_ = std::move(tmp_all_image_frame);
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
        feature_manager_.removeFront();
    }

    time_stamp_window[WINDOW_SIZE] = time_stamp_window[WINDOW_SIZE - 1];
    pos_window[WINDOW_SIZE] = pos_window[WINDOW_SIZE - 1];
    vel_window[WINDOW_SIZE] = vel_window[WINDOW_SIZE - 1];
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
            for (int j = 0; j < 7; j++)
                re_local_Pose[j] = para_Pose[i][j];
        }
    }
}

