#pragma once

#include <vector>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

#include "vins_define_internal.h"

namespace vins{
    class MotionEstimator {
    public:
        static bool solveRelativeRT(const Correspondences &correspondences,
                                    Eigen::Matrix3d &Rotation, Eigen::Vector3d &Translation);
    };
}
