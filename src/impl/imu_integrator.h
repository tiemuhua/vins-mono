//
// Created by gjt on 5/14/23.
//

#ifndef VINS_IMU_INTEGRATOR_H
#define VINS_IMU_INTEGRATOR_H

#include <Eigen/Eigen>
#include "param.h"

namespace vins {
    enum EStateOrder {
        kOrderPos = 0,
        kOrderRot = 3,
        kOrderVel = 6,
        kOrderBA = 9,
        kOrderBG = 12
    };

    enum NoiseOrder {
        kAccNoise = 0,
        kGyrNoise = 3,
        kAccWhite = 6,
        kGyrWhite = 9
    };

    struct PrevIMUState {
        Eigen::Vector3d acc = Eigen::Vector3d::Zero();
        Eigen::Vector3d gyr = Eigen::Vector3d::Zero();
        double time_stamp = 0;
        Eigen::Vector3d ba = Eigen::Vector3d::Zero();
        Eigen::Vector3d bg = Eigen::Vector3d::Zero();
    };

    class ImuIntegrator {
    public:
        static constexpr int NoiseDim = 12;
        static constexpr int StateDim = 15;
        typedef Eigen::Matrix<double, StateDim, StateDim> Jacobian, Covariance, SqrtInfo;
        typedef Eigen::Matrix<double, NoiseDim, NoiseDim> Noise;
        typedef Eigen::Matrix<double, StateDim, 1> State;

        ImuIntegrator() = delete;

        ImuIntegrator(IMUParam imu_param, PrevIMUState prev_imu_state, Eigen::Vector3d gravity);

        void predict(double time_stamp, const Eigen::Vector3d& acc, const Eigen::Vector3d& gyr);

        void rePredict(const Eigen::Vector3d& new_ba, const Eigen::Vector3d& new_bg);

        [[nodiscard]] State
        evaluate(const Eigen::Vector3d& Pi, const Eigen::Quaterniond& Qi, const Eigen::Vector3d& Vi, const Eigen::Vector3d& Bai, const Eigen::Vector3d& Bgi,
                 const Eigen::Vector3d& Pj, const Eigen::Quaterniond& Qj, const Eigen::Vector3d& Vj, const Eigen::Vector3d& Baj, const Eigen::Vector3d& Bgj) const;

        void jointLaterIntegrator(const ImuIntegrator &later_int);

        [[nodiscard]] const Eigen::Vector3d &deltaPos() const {
            return pos_;
        }

        [[nodiscard]] const Eigen::Vector3d &deltaVel() const {
            return vel_;
        }

        [[nodiscard]] const Eigen::Quaterniond &deltaQuat() const {
            return quat_;
        }

        [[nodiscard]] double deltaTime() const {
            return time_stamp_buf_.back() - time_stamp_buf_.front();
        }

        [[nodiscard]] const Jacobian &getJacobian() const {
            return jacobian_;
        }

        [[nodiscard]] const Covariance &getCovariance() const {
            return covariance_;
        }

        [[nodiscard]] const Eigen::Vector3d &getBg() const {
            return bg_;
        }

    public:
        std::vector<double> time_stamp_buf_;
        std::vector<Eigen::Vector3d> acc_buf_;
        std::vector<Eigen::Vector3d> gyr_buf_;

    private:
        static void midPointIntegral(double pre_time_stamp, const Eigen::Vector3d& pre_acc, const Eigen::Vector3d& pre_gyr,
                                     double cur_time_stamp, const Eigen::Vector3d& cur_acc, const Eigen::Vector3d& cur_gyr,
                                     const Eigen::Vector3d& ba, const Eigen::Vector3d& bg,
                                     Eigen::Vector3d& cur_pos, Eigen::Quaterniond& cur_quat, Eigen::Vector3d& cur_vel,
                                     Jacobian &jacobian, Covariance &covariance, Noise &noise);

    private:
        Eigen::Vector3d pos_ = Eigen::Vector3d::Zero();
        Eigen::Vector3d vel_ = Eigen::Vector3d::Zero();
        Eigen::Quaterniond quat_ = Eigen::Quaterniond::Identity();

        Eigen::Vector3d ba_ = Eigen::Vector3d::Zero();
        Eigen::Vector3d bg_ = Eigen::Vector3d::Zero();
        Eigen::Vector3d gravity_ = Eigen::Vector3d(0, 0, -9.81);

        Jacobian jacobian_ = Jacobian::Identity();
        Covariance covariance_ = Covariance::Zero();
        Noise noise_ = Noise::Zero();

    };

    typedef std::unique_ptr<ImuIntegrator> ImuIntegratorPtr;
}

#endif //VINS_IMU_INTEGRATOR_H
