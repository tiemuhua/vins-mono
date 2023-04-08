#pragma once

#include "../utility/utility.h"
#include "../parameters.h"

#include <ceres/ceres.h>

using namespace Eigen;

class IntegrationBase {
public:
    static constexpr int NoiseDim = 18;
    static constexpr int StateDim = 15;
    typedef Eigen::Matrix<double, StateDim, StateDim> Jacobian, Covariance;
    typedef Eigen::Matrix<double, NoiseDim, NoiseDim> Noise;

    IntegrationBase() = delete;

    IntegrationBase(const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0,
                    const Eigen::Vector3d &_linearized_ba, const Eigen::Vector3d &_linearized_bg)
            : acc_0{_acc_0}, gyr_0{_gyr_0}, linearized_acc{_acc_0}, linearized_gyr{_gyr_0},
              ba{_linearized_ba}, bg{_linearized_bg},
              jacobian{Eigen::Matrix<double, 15, 15>::Identity()}, covariance{Eigen::Matrix<double, 15, 15>::Zero()},
              sum_dt{0.0} {
        noise = Eigen::Matrix<double, 18, 18>::Zero();
        noise.block<3, 3>(0, 0) = (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(3, 3) = (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(6, 6) = (ACC_N * ACC_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(9, 9) = (GYR_N * GYR_N) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(12, 12) = (ACC_W * ACC_W) * Eigen::Matrix3d::Identity();
        noise.block<3, 3>(15, 15) = (GYR_W * GYR_W) * Eigen::Matrix3d::Identity();
    }

    void push_back(double dt, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr) {
        dt_buf.push_back(dt);
        acc_buf.push_back(acc);
        gyr_buf.push_back(gyr);
        propagate(dt, acc, gyr);
    }

    void repropagate(const Eigen::Vector3d &new_ba, const Eigen::Vector3d &new_bg) {
        sum_dt = 0.0;
        acc_0 = linearized_acc;
        gyr_0 = linearized_gyr;
        pre_pos.setZero();
        pre_quat.setIdentity();
        pre_vel.setZero();
        ba = new_ba;
        bg = new_bg;
        jacobian.setIdentity();
        covariance.setZero();
        for (int i = 0; i < static_cast<int>(dt_buf.size()); i++)
            propagate(dt_buf[i], acc_buf[i], gyr_buf[i]);
    }

    static inline Eigen::Matrix3d AntiSymmetric(const Eigen::Vector3d& vec){
        Eigen::Matrix3d mat;
        mat << 0, -vec(2), vec(1),
                vec(2), 0, -vec(0),
                -vec(1), vec(0), 0;
    }

    static void midPointIntegration(double _dt,
                             const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0,
                             const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1,
                             const Eigen::Vector3d &pre_pos, const Eigen::Quaterniond &pre_quat, const Eigen::Vector3d &pre_vel,
                             const Eigen::Vector3d &linearized_ba, const Eigen::Vector3d &linearized_bg,
                             Eigen::Vector3d &cur_pos, Eigen::Quaterniond &cur_quat, Eigen::Vector3d &cur_vel,
                             Jacobian &jacobian, Covariance &covariance, Noise &noise) {
        //LOG_I("midpoint integration");
        Vector3d un_acc_0 = pre_quat * (_acc_0 - linearized_ba);
        Vector3d un_gyr = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;
        cur_quat = pre_quat * Quaterniond(1, un_gyr(0) * _dt / 2, un_gyr(1) * _dt / 2, un_gyr(2) * _dt / 2);
        Vector3d un_acc_1 = cur_quat * (_acc_1 - linearized_ba);
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        cur_pos = pre_pos + pre_vel * _dt + 0.5 * un_acc * _dt * _dt;
        cur_vel = pre_vel + un_acc * _dt;

        Vector3d w_x = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;
        Vector3d a_0_x = _acc_0 - linearized_ba;
        Vector3d a_1_x = _acc_1 - linearized_ba;
        Matrix3d R_w_x = AntiSymmetric(w_x);
        Matrix3d R_a_0_x = AntiSymmetric(a_0_x);
        Matrix3d R_a_1_x = AntiSymmetric(a_1_x);

        MatrixXd F = MatrixXd::Zero(15, 15);
        const double dt2 = _dt * _dt;
        const Eigen::Matrix3d pre_rot = pre_quat.toRotationMatrix();
        const Eigen::Matrix3d cur_rot = cur_quat.toRotationMatrix();
        const Eigen::Matrix3d mat = Matrix3d::Identity() - R_w_x * _dt;
        const Eigen::Matrix3d mid_rot = pre_rot * R_a_0_x + cur_rot * R_a_1_x * mat;
        F.block<3, 3>(0, 0) = Matrix3d::Identity();
        F.block<3, 3>(0, 3) = -0.25 * dt2 * mid_rot;
        F.block<3, 3>(0, 6) = MatrixXd::Identity(3, 3) * _dt;
        F.block<3, 3>(0, 9) = -0.25 * (pre_rot + cur_rot) * dt2;
        F.block<3, 3>(0, 12) = 0.25 * cur_rot * R_a_1_x * dt2 * _dt;
        F.block<3, 3>(3, 3) = mat;
        F.block<3, 3>(3, 12) = -1.0 * MatrixXd::Identity(3, 3) * _dt;
        F.block<3, 3>(6, 3) = -0.5 * _dt * mid_rot;
        F.block<3, 3>(6, 6) = Matrix3d::Identity();
        F.block<3, 3>(6, 9) = -0.5 * (pre_rot + cur_rot) * _dt;
        F.block<3, 3>(6, 12) = -0.5 * cur_rot * R_a_1_x * _dt * -_dt;
        F.block<3, 3>(9, 9) = Matrix3d::Identity();
        F.block<3, 3>(12, 12) = Matrix3d::Identity();

        MatrixXd V = MatrixXd::Zero(15, 18);
        V.block<3, 3>(0, 0) = 0.25 * pre_rot * dt2;
        V.block<3, 3>(0, 3) = 0.25 * -cur_rot * R_a_1_x * dt2 * 0.5 * _dt;
        V.block<3, 3>(0, 6) = 0.25 * cur_rot * dt2;
        V.block<3, 3>(0, 9) = V.block<3, 3>(0, 3);
        V.block<3, 3>(3, 3) = 0.5 * MatrixXd::Identity(3, 3) * _dt;
        V.block<3, 3>(3, 9) = 0.5 * MatrixXd::Identity(3, 3) * _dt;
        V.block<3, 3>(6, 0) = 0.5 * pre_rot * _dt;
        V.block<3, 3>(6, 3) = 0.5 * -cur_rot * R_a_1_x * _dt * 0.5 * _dt;
        V.block<3, 3>(6, 6) = 0.5 * cur_rot * _dt;
        V.block<3, 3>(6, 9) = V.block<3, 3>(6, 3);
        V.block<3, 3>(9, 12) = MatrixXd::Identity(3, 3) * _dt;
        V.block<3, 3>(12, 15) = MatrixXd::Identity(3, 3) * _dt;

        jacobian = F * jacobian;
        covariance = F * covariance * F.transpose() + V * noise * V.transpose();
    }

    void propagate(double dt, const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1) {
        acc_1 = _acc_1;
        gyr_1 = _gyr_1;
        Vector3d cur_pos;
        Quaterniond cur_quat;
        Vector3d cur_vel;

        midPointIntegration(dt, acc_0, gyr_0, _acc_1, _gyr_1, pre_pos, pre_quat, pre_vel,
                            ba, bg,
                            cur_pos, cur_quat, cur_vel, jacobian, covariance, noise);

        pre_pos = cur_pos;
        pre_quat = cur_quat;
        pre_vel = cur_vel;
        pre_quat.normalize();
        sum_dt += dt;
        acc_0 = acc_1;
        gyr_0 = gyr_1;

    }

    Eigen::Matrix<double, 15, 1>
    evaluate(const Eigen::Vector3d &Pi, const Eigen::Quaterniond &Qi, const Eigen::Vector3d &Vi,
             const Eigen::Vector3d &Bai, const Eigen::Vector3d &Bgi,
             const Eigen::Vector3d &Pj, const Eigen::Quaterniond &Qj, const Eigen::Vector3d &Vj,
             const Eigen::Vector3d &Baj, const Eigen::Vector3d &Bgj) {
        Eigen::Matrix<double, 15, 1> residuals;

        Eigen::Matrix3d dp_dba = jacobian.block<3, 3>(O_P, O_BA);
        Eigen::Matrix3d dp_dbg = jacobian.block<3, 3>(O_P, O_BG);

        Eigen::Matrix3d dq_dbg = jacobian.block<3, 3>(O_R, O_BG);

        Eigen::Matrix3d dv_dba = jacobian.block<3, 3>(O_V, O_BA);
        Eigen::Matrix3d dv_dbg = jacobian.block<3, 3>(O_V, O_BG);

        Eigen::Vector3d dba = Bai - ba;
        Eigen::Vector3d dbg = Bgi - bg;

        Eigen::Quaterniond corrected_quat = pre_quat * Utility::deltaQ(dq_dbg * dbg);
        Eigen::Vector3d corrected_vel = pre_vel + dv_dba * dba + dv_dbg * dbg;
        Eigen::Vector3d corrected_pos = pre_pos + dp_dba * dba + dp_dbg * dbg;

        residuals.block<3, 1>(O_P, 0) =
                Qi.inverse() * (0.5 * G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt) - corrected_pos;
        residuals.block<3, 1>(O_R, 0) = 2 * (corrected_quat.inverse() * (Qi.inverse() * Qj)).vec();
        residuals.block<3, 1>(O_V, 0) = Qi.inverse() * (G * sum_dt + Vj - Vi) - corrected_vel;
        residuals.block<3, 1>(O_BA, 0) = Baj - Bai;
        residuals.block<3, 1>(O_BG, 0) = Bgj - Bgi;
        return residuals;
    }

    Eigen::Vector3d acc_0, gyr_0;
    Eigen::Vector3d acc_1, gyr_1;

    const Eigen::Vector3d linearized_acc, linearized_gyr;
    Eigen::Vector3d ba, bg;

    Jacobian jacobian;
    Covariance covariance;
    Noise noise;

    double sum_dt = 0.0;
    Eigen::Vector3d pre_pos = Eigen::Vector3d::Zero();
    Eigen::Vector3d pre_vel = Eigen::Vector3d::Zero();
    Eigen::Quaterniond pre_quat = Eigen::Quaterniond::Identity();

    std::vector<double> dt_buf;
    std::vector<Eigen::Vector3d> acc_buf;
    std::vector<Eigen::Vector3d> gyr_buf;
};