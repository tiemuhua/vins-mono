#pragma once

#include <iostream>

#include <Eigen/Dense>
#include <ceres/ceres.h>

#include "log.h"
#include "vins/imu_integrator.h"

namespace vins {
    class IMUCost : public ceres::SizedCostFunction<15, 7, 9, 7, 9> {
    public:
        IMUCost() = delete;
        explicit IMUCost(const ImuIntegrator &_pre_integral) : pre_integral_(_pre_integral) {}
        bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;
        const ImuIntegrator &pre_integral_;
    };
}