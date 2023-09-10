//
// Created by gjt on 5/14/23.
//

#ifndef VINS_IMU_INTEGRATOR_H
#define VINS_IMU_INTEGRATOR_H

#include <Eigen/Eigen>
#include "vins_define_internal.h"
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

        void predict(double time_stamp, ConstVec3dRef acc, ConstVec3dRef gyr);
        void rePredict(ConstVec3dRef new_ba, ConstVec3dRef new_bg);
        [[nodiscard]] State evaluate(ConstVec3dRef Pi, ConstQuatRef Qi, ConstVec3dRef Vi, ConstVec3dRef Bai, ConstVec3dRef Bgi,
                                     ConstVec3dRef Pj, ConstQuatRef Qj, ConstVec3dRef Vj, ConstVec3dRef Baj, ConstVec3dRef Bgj) const;
        void jointLaterIntegrator(const ImuIntegrator &later_int);

        [[nodiscard]] const Eigen::Vector3d& deltaPos() const {
            return pos_;
        }
        [[nodiscard]] const Eigen::Vector3d& deltaVel() const {
            return vel_;
        }
        [[nodiscard]] const Eigen::Quaterniond& deltaQuat() const {
            return quat_;
        }
        [[nodiscard]] double deltaTime() const {
            return time_stamp_buf_.back() - time_stamp_buf_.front();
        }
        [[nodiscard]] const Jacobian& getJacobian() const {
            return jacobian_;
        }
        [[nodiscard]] const Covariance& getCovariance() const {
            return covariance_;
        }
        [[nodiscard]] const Eigen::Vector3d& getBg() const {
            return bg_;
        }

    public:
        std::vector<double> time_stamp_buf_;
        std::vector<Eigen::Vector3d> acc_buf_;
        std::vector<Eigen::Vector3d> gyr_buf_;

    private:
        static void midPointIntegral(double pre_time_stamp, ConstVec3dRef pre_acc, ConstVec3dRef pre_gyr,
                                     double cur_time_stamp, ConstVec3dRef cur_acc, ConstVec3dRef cur_gyr,
                                     ConstVec3dRef ba, ConstVec3dRef bg,
                                     Vec3dRef cur_pos, QuatRef cur_quat, Vec3dRef cur_vel,
                                     Jacobian &jacobian, Covariance &covariance, Noise &noise);
    private:
        Eigen::Vector3d pos_ = Eigen::Vector3d::Zero();
        Eigen::Vector3d vel_ = Eigen::Vector3d::Zero();
        Eigen::Quaterniond quat_ = Eigen::Quaterniond::Identity();

        Eigen::Vector3d ba_ = Eigen::Vector3d::Zero();
        Eigen::Vector3d bg_ = Eigen::Vector3d::Zero();
        Eigen::Vector3d gravity_ = Eigen::Vector3d(0,0,-9.81);

        Jacobian jacobian_ = Jacobian::Identity();
        Covariance covariance_ = Covariance::Zero();
        Noise noise_ = Noise::Zero();

    };
    typedef std::shared_ptr<ImuIntegrator> ImuIntegratorPtr;
}

#endif //VINS_IMU_INTEGRATOR_H
