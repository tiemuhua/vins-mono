#pragma once

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <eigen3/Eigen/Dense>
#include <iostream>
#include <cstdlib>
#include <deque>
#include <map>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

using namespace Eigen;
using namespace std;

struct SFMFeature {
    bool state;
    int id;
    vector<pair<int, cv::Point2f>> observation;
    double position[3];
    double depth;
};

struct ReProjectionError3D {
    ReProjectionError3D(double observed_u, double observed_v)
            : observed_u(observed_u), observed_v(observed_v) {}

    template<typename T>
    bool operator()(const T *const camera_R, const T *const camera_T, const T *point, T *residuals) const {
        T p[3];
        ceres::QuaternionRotatePoint(camera_R, point, p);
        p[0] += camera_T[0];
        p[1] += camera_T[1];
        p[2] += camera_T[2];
        T xp = p[0] / p[2];
        T yp = p[1] / p[2];
        residuals[0] = xp - T(observed_u);
        residuals[1] = yp - T(observed_v);
        return true;
    }

    static ceres::CostFunction *Create(const double observed_x,
                                       const double observed_y) {
        return (new ceres::AutoDiffCostFunction<
                ReProjectionError3D, 2, 4, 3, 3>(
                new ReProjectionError3D(observed_x, observed_y)));
    }

    double observed_u;
    double observed_v;
};

class GlobalSFM {
public:
    static bool construct(int frame_num, Quaterniond *q, Vector3d *T, int l,
                   const Matrix3d &relative_R, const Vector3d &relative_T,
                   vector<SFMFeature> &sfm_f, map<int, Vector3d> &sfm_tracked_points);
};