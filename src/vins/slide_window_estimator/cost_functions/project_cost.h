#pragma once

#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <opencv2/core/types.hpp>

namespace vins {
    class ProjectCost : public ceres::SizedCostFunction<2, 7, 7, 7, 1>
    {
    public:
        ProjectCost(const cv::Point2f &_pts_i, const cv::Point2f &_pts_j);
        bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;
        void check(double **parameters);

        Eigen::Vector3d pts_i, pts_j;
        Eigen::Matrix<double, 2, 3> tangent_base;
        static Eigen::Matrix2d sqrt_info;
        static double sum_t;
    };
}
