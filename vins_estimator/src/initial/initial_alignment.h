#pragma once

#include <eigen3/Eigen/Dense>
#include <iostream>
#include "../factor/imu_factor.h"
#include "../utility/utility.h"
#include <map>
#include <utility>
#include "../feature_manager.h"

class ImageFrame {
public:
    ImageFrame() = delete;

    ImageFrame(std::vector<FeaturePoint> _points, double _t, PreIntegration pre_integrate_, bool _is_key_frame):
            points{std::move(_points)},
            t{_t},
            pre_integrate_(std::move(pre_integrate_)),
            is_key_frame_(_is_key_frame){};
    std::vector<FeaturePoint> points;
    double t{};
    Matrix3d R;
    Vector3d T;
    PreIntegration pre_integrate_;
    bool is_key_frame_ = false;
};

bool visualInitialAlign(vector<ImageFrame> &all_image_frame_, Eigen::Vector3d& gravity_,
                        int frame_count_, BgWindow &bg_window, PosWindow& pos_window, RotWindow &rot_window,
                        VelWindow &vel_window, PreIntegrateWindow &pre_integrate_window,
                        FeatureManager &feature_manager_);