#pragma once

#include "../utility/utility.h"
#include "../parameters.h"

using namespace Eigen;

class PreIntegration {
public:
    static constexpr int NoiseDim = 18;
    static constexpr int StateDim = 15;
    typedef Eigen::Matrix<double, StateDim, StateDim> Jacobian, Covariance;
    typedef Eigen::Matrix<double, NoiseDim, NoiseDim> Noise;

    PreIntegration() = delete;

    PreIntegration(const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr,
                   Eigen::Vector3d _ba, Eigen::Vector3d _bg):
            ba_{std::move(_ba)},
            bg_{std::move(_bg)} {
        acc_buf.emplace_back(acc);
        gyr_buf.emplace_back(gyr);
        dt_buf.emplace_back(0);

        noise.block<3, 3>(0, 0) = (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(3, 3) = (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(6, 6) = (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(9, 9) = (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(12, 12) = (ACC_W * ACC_W) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(15, 15) = (GYR_W * GYR_W) * Eigen::Matrix3d::Identity();
    }

    void prediction(double dt, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr) {
        Eigen::Vector3d cur_pos = Eigen::Vector3d::Zero();
        Eigen::Quaterniond cur_quat = Eigen::Quaterniond::Identity();
        Eigen::Vector3d cur_vel = Eigen::Vector3d::Zero();

        midPointIntegration(dt, acc_buf.back(), gyr_buf.back(), acc, gyr,
                            pre_pos, pre_quat, pre_vel, ba_, bg_,
                            cur_pos, cur_quat, cur_vel, jacobian, covariance, noise);

        dt_buf.push_back(dt);
        acc_buf.push_back(acc);
        gyr_buf.push_back(gyr);
        sum_dt += dt;

        pre_pos = cur_pos;
        pre_quat = cur_quat;
        pre_vel = cur_vel;
    }

    void rePrediction(const Eigen::Vector3d &new_ba, const Eigen::Vector3d &new_bg) {
        pre_pos.setZero();
        pre_quat.setIdentity();
        pre_vel.setZero();
        ba_ = new_ba;
        bg_ = new_bg;
        jacobian.setIdentity();
        covariance.setZero();

        Eigen::Vector3d cur_pos = Eigen::Vector3d::Zero();
        Eigen::Quaterniond cur_quat = Eigen::Quaterniond::Identity();
        Eigen::Vector3d cur_vel = Eigen::Vector3d::Zero();

        for (int i = 1; i < static_cast<int>(dt_buf.size()); i++) {
            midPointIntegration(dt_buf[i], acc_buf[i-1], gyr_buf[i-1], acc_buf[i], gyr_buf[i],
                                pre_pos, pre_quat, pre_vel, ba_, bg_,
                                cur_pos, cur_quat, cur_vel, jacobian, covariance, noise);
            pre_pos = cur_pos;
            pre_quat = cur_quat;
            pre_vel = cur_vel;
        }
    }

    Eigen::Matrix<double, 15, 1>
    evaluate(const Eigen::Vector3d &Pi, const Eigen::Quaterniond &Qi, const Eigen::Vector3d &Vi,
             const Eigen::Vector3d &Bai, const Eigen::Vector3d &Bgi,
             const Eigen::Vector3d &Pj, const Eigen::Quaterniond &Qj, const Eigen::Vector3d &Vj,
             const Eigen::Vector3d &Baj, const Eigen::Vector3d &Bgj) {
        Eigen::Matrix3d dp_dba = jacobian.block<3, 3>(O_P, O_BA);
        Eigen::Matrix3d dp_dbg = jacobian.block<3, 3>(O_P, O_BG);

        Eigen::Matrix3d dq_dbg = jacobian.block<3, 3>(O_R, O_BG);

        Eigen::Matrix3d dv_dba = jacobian.block<3, 3>(O_V, O_BA);
        Eigen::Matrix3d dv_dbg = jacobian.block<3, 3>(O_V, O_BG);

        Eigen::Vector3d dba = Bai - ba_;
        Eigen::Vector3d dbg = Bgi - bg_;

        Eigen::Quaterniond corrected_quat = pre_quat * Utility::deltaQ(dq_dbg * dbg);
        Eigen::Vector3d corrected_vel = pre_vel + dv_dba * dba + dv_dbg * dbg;
        Eigen::Vector3d corrected_pos = pre_pos + dp_dba * dba + dp_dbg * dbg;

        Eigen::Matrix<double, 15, 1> residuals;
        residuals.block<3, 1>(O_P, 0) =
                Qi.inverse() * (0.5 * G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt) - corrected_pos;
        residuals.block<3, 1>(O_R, 0) = 2 * (corrected_quat.inverse() * (Qi.inverse() * Qj)).vec();
        residuals.block<3, 1>(O_V, 0) = Qi.inverse() * (G * sum_dt + Vj - Vi) - corrected_vel;
        residuals.block<3, 1>(O_BA, 0) = Baj - Bai;
        residuals.block<3, 1>(O_BG, 0) = Bgj - Bgi;
        return residuals;
    }
    Eigen::Vector3d DeltaPos() {
        return pre_pos;
    }
    Eigen::Vector3d DeltaVel() {
        return pre_vel;
    }
    Eigen::Quaterniond DeltaQuat() {
        return pre_quat;
    }
    Eigen::Vector3d ba_, bg_;
    double sum_dt = 0.0;
    Jacobian jacobian = Jacobian::Identity();
    Covariance covariance = Covariance::Zero();
    Noise noise = Noise::Zero();

private:
    static inline Eigen::Matrix3d AntiSymmetric(const Eigen::Vector3d& vec){
        Eigen::Matrix3d mat;
        mat << 0, -vec(2), vec(1),
                vec(2), 0, -vec(0),
                -vec(1), vec(0), 0;
        return mat;
    }

    static void midPointIntegration(double dt, const Eigen::Vector3d &pre_acc, const Eigen::Vector3d &pre_gyr,
                                    const Eigen::Vector3d &cur_acc, const Eigen::Vector3d &cur_gyr,
                                    const Eigen::Vector3d &pre_pos, const Eigen::Quaterniond &pre_quat, const Eigen::Vector3d &pre_vel,
                                    const Eigen::Vector3d &ba, const Eigen::Vector3d &bg,
                                    Eigen::Vector3d &cur_pos, Eigen::Quaterniond &cur_quat, Eigen::Vector3d &cur_vel,
                                    Jacobian &jacobian, Covariance &covariance, Noise &noise) {
        const double dt2 = dt * dt;
        const double dt3 = dt2 * dt;

        Vector3d avg_gyr = 0.5 * (pre_gyr + cur_gyr) - bg;
        cur_quat = pre_quat * Utility::deltaQ(avg_gyr * dt);
        cur_quat.normalize();
        Vector3d pre_acc_no_bias = pre_acc - ba;
        Vector3d cur_acc_no_bias = cur_acc - ba;
        Vector3d pre_acc_frame_start_coordinate = pre_quat * pre_acc_no_bias;
        Vector3d cur_acc_frame_start_coordinate = cur_quat * cur_acc_no_bias;
        Vector3d avg_acc = 0.5 * (pre_acc_frame_start_coordinate + cur_acc_frame_start_coordinate);
        cur_pos = pre_pos + pre_vel * dt + 0.5 * avg_acc * dt2;
        cur_vel = pre_vel + avg_acc * dt;

        Matrix3d avg_gyr_hat = AntiSymmetric(avg_gyr);
        Matrix3d pre_acc_hat = AntiSymmetric(pre_acc_no_bias);
        Matrix3d cur_acc_hat = AntiSymmetric(cur_acc_no_bias);

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

private:

    Eigen::Vector3d pre_pos = Eigen::Vector3d::Zero();
    Eigen::Vector3d pre_vel = Eigen::Vector3d::Zero();
    Eigen::Quaterniond pre_quat = Eigen::Quaterniond::Identity();

    std::vector<double> dt_buf;
    std::vector<Eigen::Vector3d> acc_buf;
    std::vector<Eigen::Vector3d> gyr_buf;
};