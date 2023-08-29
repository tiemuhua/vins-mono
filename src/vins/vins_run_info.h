//
// Created by gjt on 6/10/23.
//

#ifndef GJT_VINS_VINS_DATA_H
#define GJT_VINS_VINS_DATA_H

#include <string>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include "DVision/BRIEF.h"
#include <camodocal/camera_models/Camera.h>
#include "imu_integrator.h"

namespace vins {
    struct State {
        Eigen::Vector3d pos;
        Eigen::Matrix3d rot;
        Eigen::Vector3d vel;
        Eigen::Vector3d ba;
        Eigen::Vector3d bg;
    };

    struct RunInfo {
        Eigen::Vector3d tic;
        Eigen::Matrix3d ric;
        Eigen::Vector3d gravity;

        // 下面三个window的大小都是Param.window_size
        std::vector<State> state_window;
        std::vector<ImuIntegrator> pre_int_window;

        // 滑动窗口中所有关键帧所观测到的特征点的集合
        std::vector<Feature> feature_window;

        std::vector<Frame> all_frames;
    };
}

#endif //GJT_VINS_VINS_DATA_H
