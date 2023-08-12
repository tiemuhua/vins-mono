#pragma once

#include <ceres/ceres.h>
#include <Eigen/Dense>
#include "vins/vins_define_internal.h"
#include "vins/vins_run_info.h"

namespace vins{
    // 残差为二维平面上的投影误差
    // 参数为旧帧位置、旧帧四元数、新帧位置、新帧四元数、相机位置、相机四元数、逆深度、相机-imu时间差
    class ProjectTdCost : public ceres::SizedCostFunction<2, 3, 4, 3, 4, 3, 4, 1, 1> {
    public:
        ProjectTdCost(const cv::Point2f &p1, const cv::Point2f& p2,
                      const cv::Point2f &vel1, const cv::Point2f &vel2,
                      double time_stamp1_ms, double time_stamp2_ms);
        bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;

        Eigen::Vector3d pts_i, pts_j;
        Eigen::Vector3d velocity_i, velocity_j;
        double td_i, td_j;
        Eigen::Matrix<double, 2, 3> tangent_base;
        double row_i, row_j;
    };
}
