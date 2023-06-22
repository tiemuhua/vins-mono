#pragma once

#include <iostream>

#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>

#include "log.h"
#include "vins/imu_integrator.h"

namespace vins {
    class IMUFactor : public ceres::SizedCostFunction<15, 7, 9, 7, 9> {
    public:
        IMUFactor() = delete;
        explicit IMUFactor(const ImuIntegrator &_pre_integration) : pre_integration(_pre_integration) {}
        bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;
        const ImuIntegrator &pre_integration;
    };
}