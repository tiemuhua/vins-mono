#pragma once

#include <ceres/ceres.h>
#include <Eigen/Dense>
#include "vins/vins_define_internal.h"
#include "vins/vins_run_info.h"

namespace vins{
    class ProjectionTdFactor : public ceres::SizedCostFunction<2, 7, 7, 7, 1, 1>
    {
    public:
        ProjectionTdFactor(const FeaturePoint2D& p1, const FeaturePoint2D& p2);
        bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;
        void check(double **parameters);

        Eigen::Vector3d pts_i, pts_j;
        Eigen::Vector3d velocity_i, velocity_j;
        double td_i, td_j;
        Eigen::Matrix<double, 2, 3> tangent_base;
        double row_i, row_j;
        static Eigen::Matrix2d sqrt_info;
        static double sum_t;
    };
}
