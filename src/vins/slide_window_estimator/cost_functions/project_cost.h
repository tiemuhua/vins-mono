#pragma once

#include <ceres/ceres.h>
#include <Eigen/Dense>
#include <opencv2/core/types.hpp>

namespace vins {
    // 残差为二维平面上的投影误差
    // 参数为旧帧位置、旧帧四元数、新帧位置、新帧四元数、相机位置、相机四元数、逆深度
    class ProjectCost : public ceres::SizedCostFunction<2, 3, 4, 3, 4, 3, 4, 1> {
    public:
        ProjectCost(const cv::Point2f &_pts_i, const cv::Point2f &_pts_j);
        bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;

        Eigen::Vector3d pts_i, pts_j;
        Eigen::Matrix<double, 2, 3> tangent_base;
        static Eigen::Matrix2d sqrt_info;
    };
}
