#pragma once

#include <vector>

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include "vins/impl/vins_define_internal.h"

namespace vins {
    bool estimateRIC(const std::vector<Eigen::Matrix3d> &img_rots,
                     const std::vector<Eigen::Matrix3d> &imu_rots,
                     Eigen::Matrix3d &calib_ric_result);
}
