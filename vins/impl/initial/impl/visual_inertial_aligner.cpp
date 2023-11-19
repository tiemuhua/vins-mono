//
// Created by gjt on 5/14/23.
//

#include "visual_inertial_aligner.h"
#include <cstdlib>
#include <glog/logging.h>
#include "impl/vins_utils.h"

namespace vins {
    // 此时不知ba、bg、gravity，ba和gravity耦合，都和位移有关。而bg只和旋转有关，因此可以在不知道ba、gravity的情况下独立求解bg。
    // 认为视觉求出来的旋转是准确的，通过IMU和视觉的差求出bg
    Eigen::Vector3d estimateGyroBias(const std::vector<Eigen::Matrix3d> &imu_delta_rots,
                                     const std::vector<Eigen::Matrix3d> &img_delta_rots,
                                     const std::vector<Eigen::Matrix3d> &jacobians_bg_2_rot) {
        int interval_size = (int) imu_delta_rots.size();
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(interval_size * 3, 3);
        Eigen::VectorXd b = Eigen::VectorXd::Zero(interval_size * 3);
        for (int i = 0; i < interval_size; ++i) {
            A.block<3, 3>(i * 3, 0) = jacobians_bg_2_rot[i];
            // todo 下面这行感觉好像写反了
            Eigen::Quaterniond img_imu_diff(imu_delta_rots[i].transpose() * img_delta_rots[i]);
            b.block<3, 1>(i * 3, 0) = 2 * img_imu_diff.vec();
        }
        Eigen::Vector3d bg = (A.transpose() * A).ldlt().solve(A.transpose() * b);
        return bg;
    }

    bool estimateRIC(const std::vector<Eigen::Matrix3d> &img_rots,
                     const std::vector<Eigen::Matrix3d> &imu_rots,
                     Eigen::Matrix3d &calib_ric_result) {
        PRINT_FUNCTION_TIME_COST
        int frame_count = (int) img_rots.size();

        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(frame_count * 4, 4);
        for (int i = 0; i < frame_count; i++) {
            Eigen::Quaterniond rot_img(img_rots[i]);
            Eigen::Quaterniond rot_imu(imu_rots[i]);

            Eigen::Vector3d img_q = rot_img.vec();
            Eigen::Matrix4d L = rot_img.w() * Eigen::Matrix4d::Identity();
            L.block<3, 3>(0, 0) += utils::skewSymmetric(img_q);
            L.block<3, 1>(0, 3) = img_q;
            L.block<1, 3>(3, 0) = -img_q.transpose();

            Eigen::Vector3d imu_q = rot_imu.vec();
            Eigen::Matrix4d R = rot_imu.w() * Eigen::Matrix4d::Identity();
            R.block<3, 3>(0, 0) -= utils::skewSymmetric(imu_q);
            R.block<3, 1>(0, 3) = imu_q;
            R.block<1, 3>(3, 0) = -imu_q.transpose();

            A.block<4, 4>(i * 4, 0) = L - R;
            LOG(INFO) << "L:\n" << L << "R:\n" << R << "\n";
        }

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Quaterniond estimated_R((Eigen::Vector4d)svd.matrixV().col(3));
        LOG(INFO) << "svd singular values:" << svd.singularValues().transpose();
        LOG(INFO) << "estimeated RIC Quaterniond:" << estimated_R;
        if (svd.singularValues()(3) > 1e-4) {
            LOG(ERROR) << "no ric solution";
            return false;
        }
        if (svd.singularValues()(2) < 1e-2) {
            LOG(ERROR) << "init ric failed";
            return false;
        }
        calib_ric_result = estimated_R.toRotationMatrix().inverse();
        return true;
    }

    bool estimateTICGravityScaleVelocity(const std::vector<Eigen::Matrix3d> &frames_img_rot,
                                         const std::vector<Eigen::Vector3d> &img_delta_poses,
                                         const std::vector<Eigen::Vector3d> &imu_delta_poses,
                                         const std::vector<Eigen::Vector3d> &imu_delta_velocities,
                                         const std::vector<double> &delta_times,
                                         const Eigen::Matrix3d &RIC,
                                         Eigen::Vector3d &TIC,
                                         Eigen::Vector3d &gravity,
                                         double &scale,
                                         std::vector<Eigen::Vector3d> &vel) {
        PRINT_FUNCTION_TIME_COST
        int frame_size = (int) frames_img_rot.size();
        int n_state = frame_size * 3 + 3 + 1 + 3;
        int gravity_idx = frame_size * 3;
        int scale_idx = gravity_idx + 3;
        int tic_idx = scale_idx + 1;
        int n_equation = frame_size * 6;

        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n_equation, n_state);
        Eigen::VectorXd b = Eigen::VectorXd::Zero(n_equation);

        for (int i = 0; i < frame_size - 1; ++i) {
            const double dt = delta_times[i];
            const double dt2 = dt * dt;

            //.在l坐标系中写出位移方程，l就是visual_initiator.cpp里面的big_parallax_frame_id这帧.
            //.等式两边均为imu在l坐标系的位移.
            // img_T * scale - rot_j * RIC.inv * TIC + rot_i * RIC.inv * TIC == rot_i * RIC.inv * imu_T - 0.5 * gravity * dt^2 + dt * vel_avg
            //.整理为.
            // 0.5 * dt^2 * gravity + img_T * scale + (rot_i - rot_j) * RIC.inv * TIC - dt * 0.5 * (vel[i] + vel[j]) == rot_i * RIC.inv * imu_T
            A.block<3, 3>(i * 6, gravity_idx) = 0.5 * dt2 * Eigen::Matrix3d::Identity();
            A.block<3, 1>(i * 6, scale_idx) = img_delta_poses[i];
            A.block<3, 3>(i * 6, tic_idx) = (frames_img_rot[i] - frames_img_rot[i + 1]) * RIC.transpose();
            A.block<3, 3>(i * 6, i * 3) = -dt * 0.5 * Eigen::Matrix3d::Identity();
            A.block<3, 3>(i * 6, i * 3 + 3) = -dt * 0.5 * Eigen::Matrix3d::Identity();
            b.block<3, 1>(i * 6, 0) = frames_img_rot[i] * RIC.transpose() * imu_delta_poses[i];

            //.在l坐标系中写出位移方程，l就是visual_initiator.cpp里面的big_parallax_frame_id这帧.
            //.等式两边均为imu在l系中的速度变化.
            // vel[j] - vel[i] = rot_i * RIC.inv * imu_delta_vel - dt * gravity
            //.整理为.
            // dt * gravity - velocity[i] + velocity[j] = rot_i * RIC.inv * imu_delta_vel
            A.block<3, 3>(i * 6 + 3, gravity_idx) = dt * Eigen::Matrix3d::Identity();
            A.block<3, 3>(i * 6 + 3, i * 3) = -Eigen::Matrix3d::Identity();
            A.block<3, 3>(i * 6 + 3, i * 3 + 3) = Eigen::Matrix3d::Identity();
            b.block<3, 1>(i * 6 + 3, 0) = frames_img_rot[i] * RIC.transpose() * imu_delta_velocities[i];
        }
        Eigen::VectorXd x = (A.transpose() * A).ldlt().solve(A.transpose() * b);
        vel.clear();
        for (int i = 0; i < frame_size; ++i) {
            vel.emplace_back(x.block<3, 1>(i * 3, 0));
        }
        gravity = x.segment<3>(gravity_idx);
        scale = x(scale_idx);
        TIC = x.segment<3>(tic_idx);
        assert(x.norm() < 1e5);

        LOG(INFO) << "gravity.norm():" << gravity.norm() << "\tscale:" << scale;
        constexpr double standard_gravity_norm = 9.8;
        if (abs(gravity.norm() - standard_gravity_norm) > 1.0 || scale < 1e-4) {
            LOG(ERROR) << "abs(gravity.norm() - standard_gravity_norm) > 1.0 || scale < 1e-4";
            return false;
        }
        return true;
    }

    Eigen::Matrix3d rotGravityToZAxis(const Eigen::Vector3d& gravity, const Eigen::Matrix3d& R0) {
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
