#pragma once

#include <iostream>

#include <Eigen/Dense>
#include <ceres/ceres.h>

#include "impl/imu_integrator.h"

namespace vins {
    // 残差为15维，位置、四元数、速度、角速度漂移、加速度漂移各5维
    // 参数分别为pos_i、quat_i、vel_i、ba_i、bg_i、pos_j、quat_j、vel_j、ba_j、bg_j，维度见模版参数
    class IMUCost : public ceres::SizedCostFunction<15, 3, 4, 3, 3, 3, 3, 4, 3, 3, 3> {
    public:
        IMUCost() = delete;

        explicit IMUCost(const ImuIntegrator &_pre_integral) : pre_integral_(_pre_integral) {}

        bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;

        const ImuIntegrator &pre_integral_;
    };
}