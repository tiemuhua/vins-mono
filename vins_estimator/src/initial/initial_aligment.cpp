#include "initial_alignment.h"
#include "log.h"

Vector3d solveGyroscopeBias(const vector<ImageFrame> &all_image_frame) {
    Matrix3d A = Matrix3d::Zero();
    Vector3d b = Vector3d::Zero();
    for (auto frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++) {
        auto frame_j = next(frame_i);
        MatrixXd tmp_A(3, 3);
        tmp_A.setZero();
        VectorXd tmp_b(3);
        tmp_b.setZero();
        Eigen::Quaterniond q_ij(frame_i->R.transpose() * frame_j->R);
        tmp_A = frame_j->pre_integrate_.jacobian.template block<3, 3>(O_R, O_BG);
        tmp_b = 2 * (frame_j->pre_integrate_.DeltaQuat().inverse() * q_ij).vec();
        A += tmp_A.transpose() * tmp_A;
        b += tmp_A.transpose() * tmp_b;
    }
    Vector3d delta_bg = A.ldlt().solve(b);
    return delta_bg;
}

typedef Matrix<double, 6, 10> Matrix_6_10;
typedef Matrix<double, 10, 10> Matrix10d;
typedef Matrix<double, 10, 1> Vector10d;
typedef Matrix<double, 6, 9> Matrix_6_9;
typedef Matrix<double, 9, 9> Matrix9d;
typedef Matrix<double, 6, 1> Vector6d;
typedef Matrix<double, 9, 1> Vector9d;
typedef Matrix<double, 3, 2> Matrix_3_2;

Matrix_3_2 TangentBasis(const Vector3d &g0) {
    Vector3d a = g0.normalized();
    Vector3d tmp(0, 0, 1);
    if (a == tmp)
        tmp << 1, 0, 0;
    Vector3d b = (tmp - a * (a.transpose() * tmp)).normalized();
    Vector3d c = a.cross(b);
    Matrix_3_2 bc = Matrix_3_2::Zero();
    bc.block<3, 1>(0, 0) = b;
    bc.block<3, 1>(0, 1) = c;
    return bc;
}

/**
 * @param all_image_frame 所有图片
 * @param g 标定后的重力加速度
 * @param s 尺度
 * @param vel todo坐标系下的速度，长度为todo
 * */
void RefineGravity(const vector<ImageFrame> &all_image_frame,
                   Vector3d &g, double &s, std::vector<Vector3d> &vel) {
    g = g.normalized() * G.norm();
    int n_state = (int )all_image_frame.size() * 3 + 2 + 1;

    MatrixXd A = MatrixXd::Zero(n_state, n_state);
    VectorXd b = VectorXd::Zero(n_state);
    VectorXd x = VectorXd::Zero(n_state);
    vel.resize(all_image_frame.size());

    for (int iter = 0; iter < 4; iter++) {
        Matrix_3_2 tangent_basis = TangentBasis(g);
        for (int i = 0; i < (int) all_image_frame.size() - 1; ++i) {
            const ImageFrame& frame_i = all_image_frame[i];
            const ImageFrame& frame_j = all_image_frame[i + 1];

            Matrix_6_9 tmp_A = Matrix_6_9::Zero();
            Vector6d tmp_b = Vector6d::Zero();

            Matrix3d rot_i_inv = frame_i.R.transpose();
            double dt = frame_j.pre_integrate_.sum_dt;
            double dt2 = dt * dt;
            Vector3d delta_pos_j = frame_j.pre_integrate_.DeltaPos();
            Vector3d delta_vel_j = frame_j.pre_integrate_.DeltaVel();
            tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
            tmp_A.block<3, 2>(0, 6) = rot_i_inv * dt2 / 2 * tangent_basis;
            tmp_A.block<3, 1>(0, 8) = rot_i_inv * (frame_j.T - frame_i.T) / 100.0;
            tmp_b.block<3, 1>(0, 0) = delta_pos_j + rot_i_inv * frame_j.R * TIC - TIC - rot_i_inv * dt2 / 2 * g;

            tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
            tmp_A.block<3, 3>(3, 3) = rot_i_inv * frame_j.R;
            tmp_A.block<3, 2>(3, 6) = rot_i_inv * dt * tangent_basis;
            tmp_b.block<3, 1>(3, 0) = delta_vel_j - rot_i_inv * dt * Matrix3d::Identity() * g;

            Matrix9d r_A = tmp_A.transpose() * tmp_A;
            Vector9d r_b = tmp_A.transpose() * tmp_b;

            A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
            b.segment<6>(i * 3) += r_b.head<6>();

            A.bottomRightCorner<3, 3>() += r_A.bottomRightCorner<3, 3>();
            b.tail<3>() += r_b.tail<3>();

            A.block<6, 3>(i * 3, n_state - 3) += r_A.topRightCorner<6, 3>();
            A.block<3, 6>(n_state - 3, i * 3) += r_A.bottomLeftCorner<3, 6>();
        }
        A = A * 1000.0;
        b = b * 1000.0;
        x = A.ldlt().solve(b);
        Vector2d dg = x.segment<2>(n_state - 3);
        g = (g + tangent_basis * dg).normalized() * G.norm();
    }
    s = (x.tail<1>())(0) / 100.0;
    for (int i = 0; i < all_image_frame.size(); ++i) {
        vel[i] = all_image_frame[i].R * x.segment<3>(i * 3);
    }
}

bool LinearAlignment(const vector<ImageFrame> &all_image_frame, Vector3d &g) {
    int n_state = (int )all_image_frame.size() * 3 + 3 + 1;

    MatrixXd A = MatrixXd::Zero(n_state, n_state);
    VectorXd b = VectorXd::Zero(n_state);

    for (int i = 0; i < (int) all_image_frame.size() - 1; ++i) {
        const ImageFrame& frame_i = all_image_frame[i];
        const ImageFrame& frame_j = all_image_frame[i + 1];

        Matrix_6_10 tmp_A = Matrix_6_10::Zero();
        Vector6d tmp_b =  Vector6d::Zero();

        // todo tiemuhuaguo frame_i的预积分才有意义，这里的公式需要重新推导
        double dt = frame_j.pre_integrate_.sum_dt;
        Matrix3d R_i_inv = frame_i.R.transpose();
        Matrix3d R_j = frame_j.R;

        tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
        tmp_A.block<3, 3>(0, 6) = R_i_inv * dt * dt / 2;
        tmp_A.block<3, 1>(0, 9) = R_i_inv * (frame_j.T - frame_i.T) / 100.0;
        tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
        tmp_A.block<3, 3>(3, 3) = R_i_inv * R_j;
        tmp_A.block<3, 3>(3, 6) = R_i_inv * dt;
        tmp_b.block<3, 1>(0, 0) = frame_j.pre_integrate_.DeltaPos() + R_i_inv * R_j * TIC - TIC;
        tmp_b.block<3, 1>(3, 0) = frame_j.pre_integrate_.DeltaVel();

        Matrix10d r_A = tmp_A.transpose() * tmp_A;
        Vector10d r_b = tmp_A.transpose() * tmp_b;

        A.block<6, 6>(i * 3, i * 3) += r_A.topLeftCorner<6, 6>();
        b.segment<6>(i * 3) += r_b.head<6>();

        A.bottomRightCorner<4, 4>() += r_A.bottomRightCorner<4, 4>();
        b.tail<4>() += r_b.tail<4>();

        A.block<6, 4>(i * 3, n_state - 4) += r_A.topRightCorner<6, 4>();
        A.block<4, 6>(n_state - 4, i * 3) += r_A.bottomLeftCorner<4, 6>();
    }
    A = A * 1000.0;
    b = b * 1000.0;
    VectorXd x = A.ldlt().solve(b);
    double s = x(n_state - 1) / 100.0;
    LOG_I("estimated scale: %f", s);
    g = x.segment<3>(n_state - 4);
    if (fabs(g.norm() - G.norm()) > 1.0 || s < 1e-4) {
        LOG_E("fabs(g.norm() - G.norm()) > 1.0 || s < 0");
        return false;
    }
    return true;
}

bool visualInitialAlign(vector<ImageFrame> &all_image_frame_, Eigen::Vector3d& gravity_,
                        int frame_count_, BgWindow &bg_window, PosWindow& pos_window, RotWindow &rot_window,
                        VelWindow &vel_window, PreIntegrateWindow &pre_integrate_window,
                        FeatureManager &feature_manager_) {
    int frame_size = (int) all_image_frame_.size();

    // todo solveGyroscopeBias求出来的是bg的值还是bg的变化值？
    Vector3d delta_bg = solveGyroscopeBias(all_image_frame_);
    for (int i = 0; i <= WINDOW_SIZE; i++)
        bg_window[i] += delta_bg;

    for (int i = 0; i < frame_size - 1; ++i) {
        all_image_frame_[i].pre_integrate_.rePrediction(Vector3d::Zero(), delta_bg);
    }

    if (!LinearAlignment(all_image_frame_, gravity_)) {
        return false;
    }
    double s;
    std::vector<Eigen::Vector3d> velocities;
    RefineGravity(all_image_frame_, gravity_, s, velocities);
    if (s < 1e-4) {
        return false;
    }

    Matrix3d R0 = std::find(all_image_frame_.begin(),
                            all_image_frame_.end(),
                            [](const ImageFrame& it){
                                return it.is_key_frame_;
                            })->R;
    Matrix3d rot = Utility::g2R(gravity_);
    Matrix3d rot_diff = Utility::R2ypr(rot * R0).inverse() * rot;
    gravity_ = rot_diff * gravity_;
    for (int frame_id = 0, key_frame_id = 0; frame_id < frame_size; frame_id++) {
        if (!all_image_frame_[frame_id].is_key_frame_) {
            continue;
        }
        pos_window[key_frame_id] = rot_diff * all_image_frame_[frame_id].R;
        rot_window[key_frame_id] = rot_diff * all_image_frame_[frame_id].T;
        vel_window[key_frame_id] = rot_diff * velocities[frame_id];
        key_frame_id++;
    }
    for (int i = frame_count_; i >= 0; i--) {
        pos_window[i] = s * pos_window[i] - rot_window[i] * TIC - (s * pos_window[0] - rot_window[0] * TIC);
    }

    for (int i = 0; i <= WINDOW_SIZE; i++) {
        pre_integrate_window[i]->rePrediction(Vector3d::Zero(), bg_window[i]);
    }

    //triangulate on cam pose , no tic
    feature_manager_.clearDepth();
    feature_manager_.triangulate(pos_window, rot_window, Vector3d::Zero(), RIC);
    for (FeaturesOfId &features_of_id: feature_manager_.features_) {
        if (features_of_id.feature_points_.size() < 2 || features_of_id.start_frame_ >= WINDOW_SIZE - 2) {
            continue;
        }
        features_of_id.estimated_depth *= s;
    }

    return true;
}
