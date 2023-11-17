//
// Created by gjt on 5/14/23.
//

#include "imu_integrator.h"

#include "vins_utils.h"

using namespace Eigen;
namespace vins {
    ImuIntegral::ImuIntegral(IMUParam imu_param, PrevIMUState prev_imu_state, Eigen::Vector3d gravity) :
            ba_{std::move(prev_imu_state.ba)},
            bg_{std::move(prev_imu_state.bg)},
            gravity_(std::move(gravity)) {
        acc_buf_.emplace_back(std::move(prev_imu_state.acc));
        gyr_buf_.emplace_back(std::move(prev_imu_state.gyr));
        time_stamp_buf_.emplace_back(prev_imu_state.time);

        noise_.block<3, 3>(kAccNoise, kAccNoise) = (imu_param.ACC_N * imu_param.ACC_N) * Eigen::Matrix3d::Identity();
        noise_.block<3, 3>(kGyrNoise, kGyrNoise) = (imu_param.GYR_N * imu_param.GYR_N) * Eigen::Matrix3d::Identity();
        noise_.block<3, 3>(kAccWhite, kAccWhite) = (imu_param.ACC_N * imu_param.ACC_N) * Eigen::Matrix3d::Identity();
        noise_.block<3, 3>(kGyrWhite, kGyrWhite) = (imu_param.GYR_N * imu_param.GYR_N) * Eigen::Matrix3d::Identity();
    }

    void ImuIntegral::jointLaterIntegrator(const ImuIntegral &later_int) {
        int size = (int) later_int.time_stamp_buf_.size();
        for (int i = 1; i < size; ++i) {
            predict(later_int.time_stamp_buf_[i], later_int.acc_buf_[i], later_int.gyr_buf_[i]);
        }
    }

    ImuIntegral::State ImuIntegral::evaluate(
            const Eigen::Vector3d& Pi, const Eigen::Quaterniond& Qi, const Eigen::Vector3d& Vi, const Eigen::Vector3d& Bai, const Eigen::Vector3d& Bgi,
            const Eigen::Vector3d& Pj, const Eigen::Quaterniond& Qj, const Eigen::Vector3d& Vj, const Eigen::Vector3d& Baj, const Eigen::Vector3d& Bgj) const {
        Eigen::Matrix3d dp_dba = jacobian_.block<3, 3>(kOrderPos, kOrderBA);
        Eigen::Matrix3d dp_dbg = jacobian_.block<3, 3>(kOrderPos, kOrderBG);

        Eigen::Matrix3d dq_dbg = jacobian_.block<3, 3>(kOrderRot, kOrderBG);

        Eigen::Matrix3d dv_dba = jacobian_.block<3, 3>(kOrderVel, kOrderBA);
        Eigen::Matrix3d dv_dbg = jacobian_.block<3, 3>(kOrderVel, kOrderBG);

        Eigen::Vector3d dba = Bai - ba_;
        Eigen::Vector3d dbg = Bgi - bg_;

        Eigen::Quaterniond corrected_quat = quat_ * utils::deltaQ(dq_dbg * dbg);
        Eigen::Vector3d corrected_vel = vel_ + dv_dba * dba + dv_dbg * dbg;
        Eigen::Vector3d corrected_pos = pos_ + dp_dba * dba + dp_dbg * dbg;

        State residuals;
        double sum_dt = time_stamp_buf_.back() - time_stamp_buf_.front();
        residuals.block<3, 1>(kOrderPos, 0) =
                Qi.inverse() * (0.5 * gravity_ * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt) - corrected_pos;
        residuals.block<3, 1>(kOrderRot, 0) = 2 * (corrected_quat.inverse() * (Qi.inverse() * Qj)).vec();
        residuals.block<3, 1>(kOrderVel, 0) = Qi.inverse() * (gravity_ * sum_dt + Vj - Vi) - corrected_vel;
        residuals.block<3, 1>(kOrderBA, 0) = Baj - Bai;
        residuals.block<3, 1>(kOrderBG, 0) = Bgj - Bgi;
        return residuals;
    }

    void ImuIntegral::predict(double time_stamp, const Eigen::Vector3d& acc, const Eigen::Vector3d& gyr) {
        midPointIntegral(time_stamp_buf_.back(), acc_buf_.back(), gyr_buf_.back(),
                         time_stamp, acc, gyr,
                         ba_, bg_,
                         pos_, quat_, vel_,
                         jacobian_, covariance_, noise_);

        time_stamp_buf_.push_back(time_stamp);
        acc_buf_.push_back(acc);
        gyr_buf_.push_back(gyr);
    }

    void ImuIntegral::rePredict(const Eigen::Vector3d& new_ba, const Eigen::Vector3d& new_bg) {
        pos_.setZero();
        quat_.setIdentity();
        vel_.setZero();
        ba_ = new_ba;
        bg_ = new_bg;
        jacobian_.setIdentity();
        covariance_.setZero();

        for (int i = 1; i < static_cast<int>(time_stamp_buf_.size()); i++) {
            midPointIntegral(time_stamp_buf_[i - 1], acc_buf_[i - 1], gyr_buf_[i - 1],
                             time_stamp_buf_[i], acc_buf_[i], gyr_buf_[i],
                             ba_, bg_,
                             pos_, quat_, vel_,
                             jacobian_, covariance_, noise_);
        }
    }

    void ImuIntegral::midPointIntegral(double pre_time_stamp, const Eigen::Vector3d& pre_acc, const Eigen::Vector3d& pre_gyr,
                                       double cur_time_stamp, const Eigen::Vector3d& cur_acc, const Eigen::Vector3d& cur_gyr,
                                       const Eigen::Vector3d& ba, const Eigen::Vector3d& bg,
                                       Eigen::Vector3d& cur_pos, Eigen::Quaterniond& cur_quat, Eigen::Vector3d& cur_vel,
                                       Jacobian &jacobian, Covariance &covariance, Noise &noise) {
        const double dt = cur_time_stamp - pre_time_stamp;
        const double dt2 = dt * dt;
        const double dt3 = dt2 * dt;

        const Eigen::Vector3d& pre_pos = cur_pos;
        const Eigen::Quaterniond& pre_quat = cur_quat;
        const Eigen::Vector3d& pre_vel = cur_vel;

        const Eigen::Vector3d avg_gyr = 0.5 * (pre_gyr + cur_gyr) - bg;
        cur_quat = (pre_quat * utils::deltaQ(avg_gyr * dt)).normalized();
        const Eigen::Vector3d pre_acc_no_bias = pre_acc - ba;
        const Eigen::Vector3d cur_acc_no_bias = cur_acc - ba;
        const Eigen::Vector3d pre_acc_frame_start_coordinate = pre_quat * pre_acc_no_bias;
        const Eigen::Vector3d cur_acc_frame_start_coordinate = cur_quat * cur_acc_no_bias;
        const Eigen::Vector3d avg_acc = 0.5 * (pre_acc_frame_start_coordinate + cur_acc_frame_start_coordinate);
        cur_pos = pre_pos + pre_vel * dt + 0.5 * avg_acc * dt2;
        cur_vel = pre_vel + avg_acc * dt;

        const Eigen::Matrix3d avg_gyr_hat = utils::skewSymmetric(avg_gyr);
        const Eigen::Matrix3d pre_acc_hat = utils::skewSymmetric(pre_acc_no_bias);
        const Eigen::Matrix3d cur_acc_hat = utils::skewSymmetric(cur_acc_no_bias);

        const Eigen::Matrix3d pre_rot = pre_quat.toRotationMatrix();
        const Eigen::Matrix3d cur_rot = cur_quat.toRotationMatrix();
        const Eigen::Matrix3d mat = Matrix3d::Identity() - avg_gyr_hat * dt;
        const Eigen::Matrix3d mid_rot = pre_rot * pre_acc_hat + cur_rot * cur_acc_hat * mat;
        const Eigen::Matrix3d identity = Matrix3d::Identity();

        Eigen::Matrix<double, StateDim, StateDim>  F = Eigen::Matrix<double, StateDim, StateDim>::Zero();
        F.block<3, 3>(kOrderPos, kOrderPos) = identity;
        F.block<3, 3>(kOrderPos, kOrderRot) = -0.25 * dt2 * mid_rot;
        F.block<3, 3>(kOrderPos, kOrderVel) = identity * dt;
        F.block<3, 3>(kOrderPos, kOrderBA) = -0.25 * (pre_rot + cur_rot) * dt2;
        F.block<3, 3>(kOrderPos, kOrderBG) = 0.25 * cur_rot * cur_acc_hat * dt3;
        F.block<3, 3>(kOrderRot, kOrderRot) = mat;
        F.block<3, 3>(kOrderRot, kOrderBG) = -identity * dt;
        F.block<3, 3>(kOrderVel, kOrderRot) = -0.5 * dt * mid_rot;
        F.block<3, 3>(kOrderVel, kOrderVel) = identity;
        F.block<3, 3>(kOrderVel, kOrderBA) = -0.5 * (pre_rot + cur_rot) * dt;
        F.block<3, 3>(kOrderVel, kOrderBG) = -0.5 * cur_rot * cur_acc_hat * dt * -dt;
        F.block<3, 3>(kOrderBA, kOrderBA) = identity;
        F.block<3, 3>(kOrderBG, kOrderBG) = identity;

        Eigen::Matrix<double, StateDim, NoiseDim> V = Eigen::Matrix<double, StateDim, NoiseDim>::Zero();
        V.block<3, 3>(kOrderPos, kAccNoise) = 0.25 * (pre_rot + cur_rot) * dt2;
        V.block<3, 3>(kOrderPos, kGyrNoise) = -cur_rot * cur_acc_hat * dt3 / 4.0;
        V.block<3, 3>(kOrderRot, kGyrNoise) = identity * dt;
        V.block<3, 3>(kOrderVel, kAccNoise) = 0.5 * (pre_rot + cur_rot) * dt;
        V.block<3, 3>(kOrderVel, kGyrNoise) = -0.5 * cur_rot * cur_acc_hat * dt2;
        V.block<3, 3>(kOrderBA, kAccWhite) = identity * dt;
        V.block<3, 3>(kOrderBG, kGyrWhite) = identity * dt;

        jacobian = F * jacobian;
        covariance = F * covariance * F.transpose() + V * noise * V.transpose();
    }
}//namespace vins
