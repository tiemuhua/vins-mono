#pragma once

#include <eigen3/Eigen/Dense>
#include <iostream>
#include "../factor/imu_factor.h"
#include "../utility/utility.h"
#include <map>
#include <utility>
#include "../feature_manager.h"

using namespace Eigen;
using namespace std;

class ImageFrame {
public:
    ImageFrame() = delete;

    ImageFrame(std::vector<FeaturePoint> _points, double _t):
            points{std::move(_points)},
            t{_t}{};
    ~ImageFrame() {
        delete pre_integration;
        pre_integration = nullptr;
    }
    std::vector<FeaturePoint> points;
    double t{};
    Matrix3d R;
    Vector3d T;
    PreIntegration *pre_integration{};
    bool is_key_frame = false;
};

bool VisualIMUAlignment(const vector<ImageFrame> &all_image_frame, BgWindow Bgs, Vector3d &g, VectorXd &x);