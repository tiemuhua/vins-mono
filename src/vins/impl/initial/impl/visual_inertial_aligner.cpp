//
// Created by gjt on 5/14/23.
//

#include "visual_inertial_aligner.h"
#include <cstdlib>
#include "param.h"
#include "log.h"
#include "vins/impl/vins_utils.h"

namespace vins {
    static Eigen::Vector3d solveGyroBias(const std::vector<Frame> &all_image_frame) {
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
        Eigen::Vector3d delta_bg = A.ldlt().solve(b);
        return delta_bg;
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
    static void refineGravity(const std::vector<Frame> &all_frames, const double gravity_norm, ConstVec3dRef TIC,
                              Eigen::Vector3d &g, double &s, std::vector<Eigen::Vector3d> &vel) {
        const int frames_size = (int )all_frames.size();
        g = g.normalized() * gravity_norm;
        int n_state = frames_size * 3 + 2 + 1;

        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n_state, n_state);
        Eigen::Vector3d b = Eigen::Vector3d::Zero(n_state);
        Eigen::Vector3d x = Eigen::Vector3d::Zero(n_state);
        vel.resize(frames_size);

        for (int iter = 0; iter < 4; iter++) {
            Matrix_3_2 tangent_basis = TangentBasis(g);
            for (int i = 0; i < frames_size - 1; ++i) {
                const Frame& frame_i = all_frames[i];
                const Frame& frame_j = all_frames[i + 1];

                Matrix_6_9 tmp_A = Matrix_6_9::Zero();
                Vector6d tmp_b = Vector6d::Zero();

                Eigen::Matrix3d rot_i_inv = frame_i.R.transpose();
                ImuIntegrator & pre_integral_j = *frame_j.pre_integral_;
                double dt = pre_integral_j.deltaTime();
                double dt2 = dt * dt;
                Eigen::Vector3d delta_pos_j = pre_integral_j.deltaPos();
                Eigen::Vector3d delta_vel_j = pre_integral_j.deltaVel();
                tmp_A.block<3, 3>(0, 0) = -dt * Eigen::Matrix3d::Identity();
                tmp_A.block<3, 2>(0, 6) = rot_i_inv * dt2 / 2 * tangent_basis;
                tmp_A.block<3, 1>(0, 8) = rot_i_inv * (frame_j.T - frame_i.T) / 100.0;
                tmp_b.block<3, 1>(0, 0) = delta_pos_j + rot_i_inv * frame_j.R * TIC - TIC - rot_i_inv * dt2 / 2 * g;

                tmp_A.block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity();
                tmp_A.block<3, 3>(3, 3) = rot_i_inv * frame_j.R;
                tmp_A.block<3, 2>(3, 6) = rot_i_inv * dt * tangent_basis;
                tmp_b.block<3, 1>(3, 0) = delta_vel_j - rot_i_inv * dt * Eigen::Matrix3d::Identity() * g;

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
        s = (x.tail<1>())(0) / 100.0;
        for (int i = 0; i < frames_size; ++i) {
            vel[i] = all_frames[i].R * x.segment<3>(i * 3);
        }
    }

    static bool linearAlignment(const std::vector<Frame> &all_frames, ConstVec3dRef TIC,
                                const double gravity_norm, Eigen::Vector3d &g) {
        int n_state = (int )all_frames.size() * 3 + 3 + 1;

        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n_state, n_state);
        Eigen::VectorXd b = Eigen::VectorXd::Zero(n_state);

        for (int i = 0; i < (int) all_frames.size() - 1; ++i) {
            const Frame& frame_i = all_frames[i];
            const Frame& frame_j = all_frames[i + 1];

            Matrix_6_10 tmp_A = Matrix_6_10::Zero();
            Vector6d tmp_b =  Vector6d::Zero();

            // todo tiemuhuaguo frame_i的预积分才有意义，这里的公式需要重新推导
            ImuIntegrator& pre_integral_j = *frame_j.pre_integral_;
            double dt = pre_integral_j.deltaTime();
            Eigen::Matrix3d R_i_inv = frame_i.R.transpose();
            Eigen::Matrix3d R_j = frame_j.R;

            tmp_A.block<3, 3>(0, 0) = -dt * Eigen::Matrix3d::Identity();
            tmp_A.block<3, 3>(0, 6) = R_i_inv * dt * dt / 2;
            tmp_A.block<3, 1>(0, 9) = R_i_inv * (frame_j.T - frame_i.T) / 100.0;
            tmp_A.block<3, 3>(3, 0) = -Eigen::Matrix3d::Identity();
            tmp_A.block<3, 3>(3, 3) = R_i_inv * R_j;
            tmp_A.block<3, 3>(3, 6) = R_i_inv * dt;
            tmp_b.block<3, 1>(0, 0) = pre_integral_j.deltaPos() + R_i_inv * R_j * TIC - TIC;
            tmp_b.block<3, 1>(3, 0) = pre_integral_j.deltaVel();

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

    static Eigen::Matrix3d rotGravityToZAxis(ConstVec3dRef gravity, ConstMat3dRef R0) {
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

    bool alignVisualAndInertial(const double gravity_norm,
                                ConstVec3dRef TIC,
                                ConstMat3dRef RIC,
                                std::vector<Frame> &all_frames,
                                Eigen::Vector3d& gravity,
                                Eigen::Vector3d& delta_bg,
                                Eigen::Matrix3d& rot_diff,
                                std::vector<Eigen::Vector3d> &velocities,
                                double &scale) {
        int frame_size = (int) all_frames.size();

        // todo solveGyroscopeBias求出来的是bg的值还是bg的变化值？
        delta_bg = solveGyroBias(all_frames);

        for (int i = 0; i < frame_size - 1; ++i) {
            all_frames[i].pre_integral_->rePredict(Eigen::Vector3d::Zero(), delta_bg);
        }

        if (!linearAlignment(all_frames, TIC, gravity_norm, gravity)) {
            return false;
        }
        refineGravity(all_frames, gravity_norm, TIC, gravity, scale, velocities);
        if (scale < 1e-4) {
            return false;
        }

        Eigen::Matrix3d R0 = std::find_if(all_frames.begin(),all_frames.end(),[](auto it){
            return it.is_key_frame_;
        })->R;
        rot_diff = rotGravityToZAxis(gravity, R0);
        gravity = rot_diff * gravity;

        return true;
    }
}