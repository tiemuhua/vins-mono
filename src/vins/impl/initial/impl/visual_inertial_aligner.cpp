//
// Created by gjt on 5/14/23.
//

#include "visual_inertial_aligner.h"
#include <cstdlib>
#include "log.h"
#include "vins/impl/vins_utils.h"

namespace vins {
    // 此时不知ba、bg、gravity，ba和gravity耦合，都和位移有关。而bg只和旋转有关，因此可以在不知道ba、gravity的情况下独立求解bg。
    // 认为视觉求出来的旋转是准确的，通过IMU和视觉的差求出bg
    Eigen::Vector3d solveGyroBias(const std::vector<Eigen::Matrix3d> &imu_delta_rots,
                                  const std::vector<Eigen::Matrix3d> &img_delta_rots,
                                  const std::vector<Eigen::Matrix3d> &jacobians_bg_2_rot) {
        int interval_size = (int)imu_delta_rots.size();
        Eigen::MatrixXd A = Eigen::Matrix3d::Zero(interval_size, 3);
        Eigen::VectorXd b = Eigen::Vector3d::Zero(interval_size);
        for (int i = 0; i < interval_size; ++i) {
            A.block<3, 3>(i * 3, 0) = jacobians_bg_2_rot[i];
            // todo 下面这行感觉好像写反了
            Eigen::Quaterniond img_imu_diff(imu_delta_rots[i].transpose() * img_delta_rots[i]);
            b.block<3, 1>(i * 3, 0) = 2 * img_imu_diff.vec();
        }
        Eigen::Vector3d bg = (A.transpose() * A).ldlt().solve(A.transpose() * b);
        return bg;
    }

    bool solveGravityScaleVelocity(const std::vector<Frame> &all_frames,
                                   ConstVec3dRef TIC,
                                   Eigen::Vector3d &gravity,
                                   double &scale,
                                   std::vector<Eigen::Vector3d> &vel) {
        int frame_size = (int )all_frames.size();
        int n_state = frame_size * 3 + 3 + 1;
        int gravity_idx = frame_size * 3;
        int scale_idx = frame_size * 3 + 3;
        int n_equation = frame_size * 6;

        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n_equation, n_state);
        Eigen::VectorXd b = Eigen::VectorXd::Zero(n_state);

        for (int i = 0; i < (int) all_frames.size() - 1; ++i) {
            const Frame& frame_i = all_frames[i];
            const Frame& frame_j = all_frames[i + 1];

            const Eigen::Vector3d img_delta_pos = frame_j.imu_pos - frame_i.imu_pos;

            const ImuIntegrator &pre_integral_j = *frame_j.pre_integral_;
            const double dt = pre_integral_j.deltaTime();
            const double dt2 = dt * dt;
            const Eigen::Vector3d &imu_delta_pos = pre_integral_j.deltaPos();
            const Eigen::Vector3d &imu_delta_vel = pre_integral_j.deltaVel();

            //.在世界坐标系中写出位移方程.
            // img_delta_pos * scale - rot_i * img_delta_rot * TIC + rot_i * TIC == rot_i * imu_delta_pos - 0.5 * gravity * dt^2 + dt * velocity_avg
            //.整理为.
            // img_delta_pos * scale + 0.5 * dt^2 * gravity - dt * 0.5 * (velocity[i] + velocity[j]) == rot_i * (imu_delta_pos + img_delta_rot * TIC - TIC)
            A.block<3, 1>(i * 6, scale_idx) = img_delta_pos;
            A.block<3, 3>(i * 6, gravity_idx) = 0.5 * dt2 * Eigen::Matrix3d::Identity();
            A.block<3, 3>(i * 6, i * 3) = -dt * 0.5 * Eigen::Matrix3d::Identity();
            A.block<3, 3>(i * 6, i * 3 + 3) = -dt * 0.5 * Eigen::Matrix3d::Identity();
            b.block<3, 1>(i * 6, 0) = frame_i.imu_rot * imu_delta_pos + frame_j.imu_rot * TIC - frame_i.imu_rot * TIC;

            //.在世界坐标系中写出速度方程.
            // velocity[j] - velocity[i] = frame_i.R * imu_delta_vel - dt * gravity
            //.整理为.
            // dt * gravity - velocity[i] + velocity[j] = frame_i.R * imu_delta_vel
            A.block<3, 3>(i * 6 + 3, gravity_idx) = dt * Eigen::Matrix3d::Identity();
            A.block<3, 3>(i * 6 + 3, i * 3) = -Eigen::Matrix3d::Identity();
            A.block<3, 3>(i * 6 + 3, i * 3 + 3) = Eigen::Matrix3d::Identity();
            b.block<3, 1>(i * 6 + 3, 0) = frame_i.imu_rot * imu_delta_vel;
        }
        Eigen::VectorXd x = (A.transpose() * A).ldlt().solve(A.transpose() * b);
        scale = x(n_state - 1);
        gravity = x.segment<3>(n_state - 4);
        vel.clear();
        for (int i = 0; i < frame_size; ++i) {
            vel.emplace_back(x.block<3, 1>(i * 3, 0));
        }

        LOG_I("estimated scale: %f", scale);
        constexpr double standard_gravity_norm = 9.8;
        if (abs(gravity.norm() - standard_gravity_norm) > 1.0 || scale < 1e-4) {
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