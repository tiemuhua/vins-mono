#pragma once

#include <ceres/ceres.h>
#include <Eigen/Dense>
#include "../utility/utility.h"
#include "../utility/tic_toc.h"
#include "../parameters.h"

class ProjectionTdFactor : public ceres::SizedCostFunction<2, 7, 7, 7, 1, 1>
{
  public:
    ProjectionTdFactor(const cv::Point2f &_pts_i, const cv::Point2f &_pts_j,
    				   const cv::Point2f &_velocity_i, const cv::Point2f &_velocity_j,
    				   double _td_i, double _td_j, double _row_i, double _row_j);
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
