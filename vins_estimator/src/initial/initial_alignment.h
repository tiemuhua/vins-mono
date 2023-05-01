#pragma once

#include <eigen3/Eigen/Dense>
#include <iostream>
#include "../factor/imu_factor.h"
#include "../utility/utility.h"
#include <map>
#include "../feature_manager.h"

using namespace Eigen;
using namespace std;

class ImageFrame {
public:
    ImageFrame() = delete;

    ImageFrame(const map<int, vector<FeaturePoint>> &_points, double _t):
        feature_id_2_points{_points},
        t{_t}{};
    ~ImageFrame() {
        delete pre_integration;
        pre_integration = nullptr;
    }
    map<int, vector<FeaturePoint> > feature_id_2_points;
    double t{};
    Matrix3d R;
    Vector3d T;
    PreIntegration *pre_integration{};
    bool is_key_frame = false;
};

bool VisualIMUAlignment(vector<ImageFrame> &all_image_frame, Vector3d *Bgs, Vector3d &g, VectorXd &x);