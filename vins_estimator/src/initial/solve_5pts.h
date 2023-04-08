#pragma once

#include <vector>

using namespace std;

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>

using namespace Eigen;

class MotionEstimator {
public:

    static bool solveRelativeRT(const vector<pair<Vector3d, Vector3d>> &corres, Matrix3d &R, Vector3d &T);
};


