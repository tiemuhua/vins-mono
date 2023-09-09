//
// Created by gjt on 5/14/23.
//

#include "visual_inertial_aligner.h"
#include <cstdlib>
#include "param.h"
#include "log.h"
#include "vins/impl/vins_utils.h"

namespace vins {
    // 此时不知ba、bg、gravity，ba和gravity耦合，都和位移有关。而bg只和旋转有关，因此可以在不知道ba、gravity的情况下独立求解bg。
    // 认为视觉求出来的旋转是准确的，通过IMU和视觉的差求出bg
    Eigen::Vector3d solveGyroBias(const std::vector<Frame> &all_image_frame) {
        Eigen::Matrix3d A = Eigen::Matrix3d::Zero();
        Eigen::Vector3d b = Eigen::Vector3d::Zero();
        for (auto frame_i = all_image_frame.begin(); next(frame_i) != all_image_frame.end(); frame_i++) {
            auto frame_j = next(frame_i);
            Eigen::Quaterniond q_ij(frame_i->R.transpose() * frame_j->R);
            Eigen::Matrix3d tmp_A = frame_j->pre_integral_->getJacobian().template block<3, 3>(kOrderRot, kOrderBG);
            Eigen::Vector3d tmp_b = 2 * (frame_j->pre_integral_->deltaQuat().inverse() * q_ij).vec();
            A += tmp_A.transpose() * tmp_A;
            b += tmp_A.transpose() * tmp_b;
        }
        Eigen::Vector3d bg = A.ldlt().solve(b);
        return bg;
    }

    typedef Eigen::Matrix<double, 6, 10> Matrix_6_10;
    typedef Eigen::Matrix<double, 10, 10> Matrix10d;
    typedef Eigen::Matrix<double, 10, 1> Vector10d;
    typedef Eigen::Matrix<double, 6, 9> Matrix_6_9;
    typedef Eigen::Matrix<double, 9, 9> Matrix9d;
    typedef Eigen::Matrix<double, 6, 1> Vector6d;
    typedef Eigen::Matrix<double, 9, 1> Vector9d;
    typedef Eigen::Matrix<double, 3, 2> Matrix_3_2;

    static Matrix_3_2 TangentBasis(const Eigen::Vector3d &g0) {
        Eigen::Vector3d a = g0.normalized();
        Eigen::Vector3d tmp(0, 0, 1);
        if (a == tmp)
            tmp << 1, 0, 0;
        Eigen::Vector3d b = (tmp - a * (a.transpose() * tmp)).normalized();
        Eigen::Vector3d c = a.cross(b);
        Matrix_3_2 bc = Matrix_3_2::Zero();
        bc.block<3, 1>(0, 0) = b;
        bc.block<3, 1>(0, 1) = c;
        return bc;
    }

    /**
     * @param all_frames 所有图片
     * @param g 标定后的重力加速度
     * @param s 尺度
     * @param vel todo坐标系下的速度，长度为todo
     * */
    void refineGravity(const std::vector<Frame> &all_frames,
                       const double gravity_norm,
                       ConstVec3dRef TIC,
                       Eigen::Vector3d &g,
                       double &s,
                       std::vector<Eigen::Vector3d> &vel) {
        const int frames_size = (int )all_frames.size();
        g = g.normalized() * gravity_norm;
        int n_state = frames_size * 3 + 2 + 1;

        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n_state, n_state);
        Eigen::VectorXd b = Eigen::VectorXd::Zero(n_state);
        Eigen::VectorXd x = Eigen::VectorXd::Zero(n_state);
        vel.resize(frames_size);

        for (int iter = 0; iter < 4; iter++) {
            Matrix_3_2 tangent_basis = TangentBasis(g);
            for (int i = 0; i < frames_size - 1; ++i) {
                const Frame& frame_i = all_frames[i];
                const Frame& frame_j = all_frames[i + 1];

                Matrix_6_9 tmp_A = Matrix_6_9::Zero();
                Vector6d tmp_b = Vector6d::Zero();

                const Eigen::Vector3d &pos_i = frame_i.T;
                const Eigen::Matrix3d &rot_i = frame_i.R;
                const Eigen::Matrix3d &rot_i_inv = frame_i.R.transpose();
                const Eigen::Vector3d img_delta_pos = frame_j.T - frame_i.T;
                const Eigen::Vector3d img_delta_rot = rot_i_inv * frame_j.R;

                const ImuIntegrator &pre_integral_j = *frame_j.pre_integral_;
                const double dt = pre_integral_j.deltaTime();
                const double dt2 = dt * dt;
                const Eigen::Vector3d &imu_delta_pos = pre_integral_j.deltaPos();
                const Eigen::Vector3d &imu_delta_vel = pre_integral_j.deltaVel();

                //.位移方程.
                tmp_A.block<3, 3>(0, 0) = -dt * Eigen::Matrix3d::Identity();
                tmp_A.block<3, 2>(0, 6) = rot_i_inv * dt2 / 2 * tangent_basis;
                tmp_A.block<3, 1>(0, 8) = rot_i_inv * img_delta_pos / 100.0;
                tmp_b.block<3, 1>(0, 0) = imu_delta_pos + img_delta_rot * TIC - TIC - rot_i_inv * dt2 / 2 * g;

                //.速度方程.
                tmp_A.block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity();
                tmp_A.block<3, 3>(3, 3) = img_delta_rot;
                tmp_A.block<3, 2>(3, 6) = rot_i_inv * dt * tangent_basis;
                tmp_b.block<3, 1>(3, 0) = imu_delta_vel - rot_i_inv * dt * Eigen::Matrix3d::Identity() * g;

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
            Eigen::Vector2d dg = x.segment<2>(n_state - 3);
            g = (g + tangent_basis * dg).normalized() * gravity_norm;
        }
        s = x(n_state - 1) / 100.0;
        for (int i = 0; i < frames_size; ++i) {
            vel[i] = all_frames[i].R * x.segment<3>(i * 3);
        }
    }

    bool linearAlignment(const std::vector<Frame> &all_frames, ConstVec3dRef TIC,
                         const double gravity_norm, Eigen::Vector3d &g) {
        int n_state = (int )all_frames.size() * 3 + 3 + 1;

        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n_state, n_state);
        Eigen::VectorXd b = Eigen::VectorXd::Zero(n_state);

        for (int i = 0; i < (int) all_frames.size() - 1; ++i) {
            const Frame& frame_i = all_frames[i];
            const Frame& frame_j = all_frames[i + 1];

            Matrix_6_10 tmp_A = Matrix_6_10::Zero();
            Vector6d tmp_b =  Vector6d::Zero();

            const Eigen::Matrix3d &rot_i_inv = frame_i.R.transpose();
            const Eigen::Vector3d img_delta_pos = frame_j.T - frame_i.T;
            const Eigen::Vector3d img_delta_rot = rot_i_inv * frame_j.R;

            const ImuIntegrator &pre_integral_j = *frame_j.pre_integral_;
            const double dt = pre_integral_j.deltaTime();
            const double dt2 = dt * dt;
            const Eigen::Vector3d &imu_delta_pos = pre_integral_j.deltaPos();
            const Eigen::Vector3d &imu_delta_vel = pre_integral_j.deltaVel();

            //.位移方程.
            tmp_A.block<3, 3>(0, 0) = -dt * Eigen::Matrix3d::Identity();                //.速度->位移.
            tmp_A.block<3, 3>(0, 6) = rot_i_inv * dt2 / 2;                              //.重力->位移.
            tmp_A.block<3, 1>(0, 9) = rot_i_inv * (img_delta_pos) / 100.0;              //.尺度->位移.
            tmp_b.block<3, 1>(0, 0) = imu_delta_pos + img_delta_rot * TIC - TIC;

            //.速度方程.
            tmp_A.block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity();                     //.速度.
            tmp_A.block<3, 3>(3, 3) = img_delta_rot;                                    //.速度.
            tmp_A.block<3, 3>(3, 6) = rot_i_inv * dt;                                   //.重力.
            tmp_b.block<3, 1>(3, 0) = imu_delta_vel;

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
        Eigen::VectorXd x = A.ldlt().solve(b);
        double s = x(n_state - 1) / 100.0;
        LOG_I("estimated scale: %f", s);
        g = x.segment<3>(n_state - 4);
        if (abs(g.norm() - gravity_norm) > 1.0 || s < 1e-4) {
            LOG_E("fabs(g.norm() - G.norm()) > 1.0 || s < 0");
            return false;
        }
        return true;
    }

    Eigen::Matrix3d rotGravityToZAxis(ConstVec3dRef gravity, ConstMat3dRef R0) {
        Eigen::Vector3d ng1 = gravity.normalized();
        Eigen::Vector3d ng2{0, 0, 1.0};
        Eigen::Matrix3d rot_gravity_to_z_axis_in_R0_frame =
                Eigen::Quaterniond::FromTwoVectors(ng1, ng2).toRotationMatrix();
        double yaw1 = utils::rot2ypr(R0).x();
        Eigen::Matrix3d reverse_yaw_rot = Eigen::AngleAxisd(-yaw1, Eigen::Vector3d::UnitZ()).toRotationMatrix();
        rot_gravity_to_z_axis_in_R0_frame = reverse_yaw_rot * rot_gravity_to_z_axis_in_R0_frame;
        Eigen::Matrix3d rot_gravity_to_z_axis = rot_gravity_to_z_axis_in_R0_frame * R0;
        double yaw2 = utils::rot2ypr(rot_gravity_to_z_axis_in_R0_frame * R0).x();
        Eigen::Matrix3d reverse_yaw_rot2 = Eigen::AngleAxisd(-yaw2, Eigen::Vector3d::UnitZ()).toRotationMatrix();
        return reverse_yaw_rot2 * rot_gravity_to_z_axis_in_R0_frame;
    }
}