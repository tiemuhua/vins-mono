#include "initial_alignment.h"
#include "log.h"

void solveGyroscopeBias(vector<ImageFrame> &all_image_frame, Vector3d *Bgs) {
    Matrix3d A = Matrix3d::Zero();
    Vector3d b = Vector3d::Zero();
    for (auto frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++) {
        auto frame_j = next(frame_i);
        MatrixXd tmp_A(3, 3);
        tmp_A.setZero();
        VectorXd tmp_b(3);
        tmp_b.setZero();
        Eigen::Quaterniond q_ij(frame_i->R.transpose() * frame_j->R);
        tmp_A = frame_j->pre_integration->jacobian.template block<3, 3>(O_R, O_BG);
        tmp_b = 2 * (frame_j->pre_integration->DeltaQuat().inverse() * q_ij).vec();
        A += tmp_A.transpose() * tmp_A;
        b += tmp_A.transpose() * tmp_b;
    }
    Vector3d delta_bg = A.ldlt().solve(b);

    for (int i = 0; i <= WINDOW_SIZE; i++)
        Bgs[i] += delta_bg;

    for (auto frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++) {
        auto frame_j = next(frame_i);
        frame_j->pre_integration->rePrediction(Vector3d::Zero(), Bgs[0]);
    }
}

typedef Matrix<double, 6, 10> Matrix_6_10;
typedef Matrix<double, 10, 10> Matrix10d;
typedef Matrix<double, 10, 1> Vector10d;
typedef Matrix<double, 6, 9> Matrix_6_9;
typedef Matrix<double, 9, 9> Matrix9d;
typedef Matrix<double, 6, 1> Vector6d;
typedef Matrix<double, 9, 1> Vector9d;
typedef Matrix<double, 3, 2> Matrix_3_2;

Matrix_3_2 TangentBasis(Vector3d &g0) {
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

void RefineGravity(vector<ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x) {
    Vector3d g0 = g.normalized() * G.norm();
    Vector3d lx, ly;
    int n_state = (int )all_image_frame.size() * 3 + 2 + 1;

    MatrixXd A = MatrixXd::Zero(n_state, n_state);
    VectorXd b = VectorXd::Zero(n_state);

    for (int iter = 0; iter < 4; iter++) {
        Matrix_3_2 lxly = TangentBasis(g0);
        int i = 0;
        for (auto frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++) {
            auto frame_j = next(frame_i);

            Matrix_6_9 tmp_A = Matrix_6_9::Zero();
            Vector6d tmp_b = Vector6d::Zero();

            Matrix3d rot_i_inv = frame_i->R.transpose();
            double dt = frame_j->pre_integration->sum_dt;
            double dt2 = dt * dt;
            Vector3d delta_pos_j = frame_j->pre_integration->DeltaPos();
            Vector3d delta_vel_j = frame_j->pre_integration->DeltaVel();
            tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
            tmp_A.block<3, 2>(0, 6) = rot_i_inv * dt2 / 2 * Matrix3d::Identity() * lxly;
            tmp_A.block<3, 1>(0, 8) = rot_i_inv * (frame_j->T - frame_i->T) / 100.0;
            tmp_b.block<3, 1>(0, 0) = delta_pos_j + rot_i_inv * frame_j->R * TIC - TIC - rot_i_inv * dt2 / 2 * g0;

            tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
            tmp_A.block<3, 3>(3, 3) = rot_i_inv * frame_j->R;
            tmp_A.block<3, 2>(3, 6) = rot_i_inv * dt * Matrix3d::Identity() * lxly;
            tmp_b.block<3, 1>(3, 0) = delta_vel_j - rot_i_inv * dt * Matrix3d::Identity() * g0;

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
        g0 = (g0 + lxly * dg).normalized() * G.norm();
    }
    g = g0;
}

bool LinearAlignment(vector<ImageFrame> &all_image_frame, Vector3d &g, VectorXd &x) {
    int n_state = (int )all_image_frame.size() * 3 + 3 + 1;

    MatrixXd A = MatrixXd::Zero(n_state, n_state);
    VectorXd b = VectorXd::Zero(n_state);

    int i = 0;
    for (auto frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++, i++) {
        auto frame_j = next(frame_i);

        Matrix_6_10 tmp_A = Matrix_6_10::Zero();
        Vector6d tmp_b =  Vector6d::Zero();

        double dt = frame_j->pre_integration->sum_dt;
        Matrix3d R_i_inv = frame_i->R.transpose();
        Matrix3d R_j = frame_j->R;

        tmp_A.block<3, 3>(0, 0) = -dt * Matrix3d::Identity();
        tmp_A.block<3, 3>(0, 6) = R_i_inv * dt * dt / 2 * Matrix3d::Identity();
        tmp_A.block<3, 1>(0, 9) = R_i_inv * (frame_j->T - frame_i->T) / 100.0;
        tmp_A.block<3, 3>(3, 0) = -Matrix3d::Identity();
        tmp_A.block<3, 3>(3, 3) = R_i_inv * R_j;
        tmp_A.block<3, 3>(3, 6) = R_i_inv * dt * Matrix3d::Identity();
        tmp_b.block<3, 1>(0, 0) = frame_j->pre_integration->DeltaPos() + R_i_inv * R_j * TIC - TIC;
        tmp_b.block<3, 1>(3, 0) = frame_j->pre_integration->DeltaVel();

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
    x = A.ldlt().solve(b);
    double s = x(n_state - 1) / 100.0;
    LOG_D("estimated scale: %f", s);
    g = x.segment<3>(n_state - 4);
    if (fabs(g.norm() - G.norm()) > 1.0 || s < 0) {
        LOG_E("fabs(g.norm() - G.norm()) > 1.0 || s < 0");
        return false;
    }

    RefineGravity(all_image_frame, g, x);
    s = (x.tail<1>())(0) / 100.0;
    (x.tail<1>())(0) = s;
    LOG_E("s:%lf", s);
    return s > 0.0;
}

bool VisualIMUAlignment(vector<ImageFrame> &all_image_frame, Vector3d *Bgs, Vector3d &g, VectorXd &x) {
    solveGyroscopeBias(all_image_frame, Bgs);
    return LinearAlignment(all_image_frame, g, x);
}
