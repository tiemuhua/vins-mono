//
// Created by gjt on 5/14/23.
//

#include "imu_integrator.h"

#include "vins_utils.h"

using namespace Eigen;
namespace vins {
    ImuIntegrator::ImuIntegrator(double ACC_N,double ACC_W, double GYR_N, double GYR_W,
                                 double time_stamp, Eigen::Vector3d acc, Eigen::Vector3d gyr,
                                 Eigen::Vector3d ba, Eigen::Vector3d bg, Eigen::Vector3d gravity):
            ba_{std::move(ba)},
            bg_{std::move(bg)},
            gravity_(std::move(gravity)) {
        acc_buf_.emplace_back(std::move(acc));
        gyr_buf_.emplace_back(std::move(gyr));
        time_stamp_buf_.emplace_back(time_stamp);

        noise_.block<3, 3>(0, 0) = (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
        noise_.block<3, 3>(3, 3) = (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
        noise_.block<3, 3>(6, 6) = (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
        noise_.block<3, 3>(9, 9) = (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
        noise_.block<3, 3>(12, 12) = (ACC_W * ACC_W) * Eigen::Matrix3d::Identity();
        noise_.block<3, 3>(15, 15) = (GYR_W * GYR_W) * Eigen::Matrix3d::Identity();
    }

    ImuIntegrator::State ImuIntegrator::evaluate(
            ConstVec3dRef Pi, ConstQuatRef Qi, ConstVec3dRef Vi, ConstVec3dRef Bai, ConstVec3dRef Bgi,
            ConstVec3dRef Pj, ConstQuatRef Qj, ConstVec3dRef Vj, ConstVec3dRef Baj, ConstVec3dRef Bgj) const {
        Eigen::Matrix3d dp_dba = jacobian_.block<3, 3>(kOrderPos, kOrderBA);
        Eigen::Matrix3d dp_dbg = jacobian_.block<3, 3>(kOrderPos, kOrderBG);

        Eigen::Matrix3d dq_dbg = jacobian_.block<3, 3>(kOrderRot, kOrderBG);

        Eigen::Matrix3d dv_dba = jacobian_.block<3, 3>(kOrderVel, kOrderBA);
        Eigen::Matrix3d dv_dbg = jacobian_.block<3, 3>(kOrderVel, kOrderBG);

        Eigen::Vector3d dba = Bai - ba_;
        Eigen::Vector3d dbg = Bgi - bg_;

        Eigen::Quaterniond corrected_quat = pre_quat_ * utils::deltaQ(dq_dbg * dbg);
        Eigen::Vector3d corrected_vel = pre_vel_ + dv_dba * dba + dv_dbg * dbg;
        Eigen::Vector3d corrected_pos = pre_pos_ + dp_dba * dba + dp_dbg * dbg;

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

    void ImuIntegrator::predict(double time_stamp, ConstVec3dRef acc, ConstVec3dRef gyr) {
        Eigen::Vector3d cur_pos = Eigen::Vector3d::Zero();
        Eigen::Quaterniond cur_quat = Eigen::Quaterniond::Identity();
        Eigen::Vector3d cur_vel = Eigen::Vector3d::Zero();

        midPointIntegral(time_stamp_buf_.back(), acc_buf_.back(), gyr_buf_.back(),
                         time_stamp, acc, gyr,
                         pre_pos_, pre_quat_, pre_vel_, ba_, bg_,
                         cur_pos, cur_quat, cur_vel, jacobian_, covariance_, noise_);

        time_stamp_buf_.push_back(time_stamp);
        acc_buf_.push_back(acc);
        gyr_buf_.push_back(gyr);

        pre_pos_ = cur_pos;
        pre_quat_ = cur_quat;
        pre_vel_ = cur_vel;
    }

    void ImuIntegrator::rePredict(ConstVec3dRef new_ba, ConstVec3dRef new_bg) {
        pre_pos_.setZero();
        pre_quat_.setIdentity();
        pre_vel_.setZero();
        ba_ = new_ba;
        bg_ = new_bg;
        jacobian_.setIdentity();
        covariance_.setZero();

        Eigen::Vector3d cur_pos = Eigen::Vector3d::Zero();
        Eigen::Quaterniond cur_quat = Eigen::Quaterniond::Identity();
        Eigen::Vector3d cur_vel = Eigen::Vector3d::Zero();

        for (int i = 1; i < static_cast<int>(time_stamp_buf_.size()); i++) {
            midPointIntegral(time_stamp_buf_[i - 1], acc_buf_[i - 1], gyr_buf_[i - 1],
                             time_stamp_buf_[i], acc_buf_[i], gyr_buf_[i],
                             pre_pos_, pre_quat_, pre_vel_, ba_, bg_,
                             cur_pos, cur_quat, cur_vel, jacobian_, covariance_, noise_);
            pre_pos_ = cur_pos;
            pre_quat_ = cur_quat;
            pre_vel_ = cur_vel;
        }
    }

    void ImuIntegrator::midPointIntegral(double pre_time_stamp, ConstVec3dRef pre_acc, ConstVec3dRef pre_gyr,
                                         double cur_time_stamp, ConstVec3dRef cur_acc, ConstVec3dRef cur_gyr,
                                         ConstVec3dRef pre_pos, ConstQuatRef pre_quat, ConstVec3dRef pre_vel,
                                         ConstVec3dRef ba, ConstVec3dRef bg,
                                         Vec3dRef cur_pos, QuatRef cur_quat, Vec3dRef cur_vel,
                                         Jacobian &jacobian, Covariance &covariance, Noise &noise) {
        const double dt = cur_time_stamp - pre_time_stamp;
        const double dt2 = dt * dt;
        const double dt3 = dt2 * dt;

        Vector3d avg_gyr = 0.5 * (pre_gyr + cur_gyr) - bg;
        cur_quat = pre_quat * utils::deltaQ(avg_gyr * dt);
        cur_quat.normalize();
        Vector3d pre_acc_no_bias = pre_acc - ba;
        Vector3d cur_acc_no_bias = cur_acc - ba;
        Vector3d pre_acc_frame_start_coordinate = pre_quat * pre_acc_no_bias;
        Vector3d cur_acc_frame_start_coordinate = cur_quat * cur_acc_no_bias;
        Vector3d avg_acc = 0.5 * (pre_acc_frame_start_coordinate + cur_acc_frame_start_coordinate);
        cur_pos = pre_pos + pre_vel * dt + 0.5 * avg_acc * dt2;
        cur_vel = pre_vel + avg_acc * dt;

        Matrix3d avg_gyr_hat = utils::skewSymmetric(avg_gyr);
        Matrix3d pre_acc_hat = utils::skewSymmetric(pre_acc_no_bias);
        Matrix3d cur_acc_hat = utils::skewSymmetric(cur_acc_no_bias);

        const Eigen::Matrix3d pre_rot = pre_quat.toRotationMatrix();
        const Eigen::Matrix3d cur_rot = cur_quat.toRotationMatrix();
        const Eigen::Matrix3d mat = Matrix3d::Identity() - avg_gyr_hat * dt;
        const Eigen::Matrix3d mid_rot = pre_rot * pre_acc_hat + cur_rot * cur_acc_hat * mat;
        Matrix3d identity = MatrixXd::Identity(3, 3);

        MatrixXd F = MatrixXd::Zero(15, 15);
        F.block<3, 3>(0, 0) = identity;
        F.block<3, 3>(0, 3) = -0.25 * dt2 * mid_rot;
        F.block<3, 3>(0, 6) = identity * dt;
        F.block<3, 3>(0, 9) = -0.25 * (pre_rot + cur_rot) * dt2;
        F.block<3, 3>(0, 12) = 0.25 * cur_rot * cur_acc_hat * dt3;
        F.block<3, 3>(3, 3) = mat;
        F.block<3, 3>(3, 12) = -identity * dt;
        F.block<3, 3>(6, 3) = -0.5 * dt * mid_rot;
        F.block<3, 3>(6, 6) = identity;
        F.block<3, 3>(6, 9) = -0.5 * (pre_rot + cur_rot) * dt;
        F.block<3, 3>(6, 12) = -0.5 * cur_rot * cur_acc_hat * dt * -dt;
        F.block<3, 3>(9, 9) = identity;
        F.block<3, 3>(12, 12) = identity;

        MatrixXd V = MatrixXd::Zero(15, 18);
        V.block<3, 3>(0, 0) = 0.25 * pre_rot * dt2;
        V.block<3, 3>(0, 3) = -cur_rot * cur_acc_hat * dt3 / 8.0;
        V.block<3, 3>(0, 6) = 0.25 * cur_rot * dt2;
        V.block<3, 3>(0, 9) = V.block<3, 3>(0, 3);
        V.block<3, 3>(3, 3) = 0.5 * identity * dt;
        V.block<3, 3>(3, 9) = 0.5 * identity * dt;
        V.block<3, 3>(6, 0) = 0.5 * pre_rot * dt;
        V.block<3, 3>(6, 3) = -0.25 * cur_rot * cur_acc_hat * dt2;
        V.block<3, 3>(6, 6) = 0.5 * cur_rot * dt;
        V.block<3, 3>(6, 9) = V.block<3, 3>(6, 3);
        V.block<3, 3>(9, 12) = identity * dt;
        V.block<3, 3>(12, 15) = identity * dt;

        jacobian = F * jacobian;
        covariance = F * covariance * F.transpose() + V * noise * V.transpose();
    }
}//namespace vins
