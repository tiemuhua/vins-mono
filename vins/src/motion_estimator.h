#pragma once

#include <vector>

using namespace std;

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

using namespace Eigen;

class MotionEstimator {
public:
    static bool solveRelativeRT(const vector<pair<cv::Point2f , cv::Point2f>> &correspondences,
                                Matrix3d &Rotation, Vector3d &Translation);
};
