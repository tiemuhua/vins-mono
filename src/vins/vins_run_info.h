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
    struct KeyFrameState {
        Eigen::Vector3d pos = Eigen::Vector3d::Zero();
        Eigen::Matrix3d rot = Eigen::Matrix3d::Identity();
        Eigen::Vector3d vel = Eigen::Vector3d::Zero();
        Eigen::Vector3d ba = Eigen::Vector3d::Zero();
        Eigen::Vector3d bg = Eigen::Vector3d::Zero();
    };

    struct RunInfo {
        Eigen::Vector3d tic;
        Eigen::Matrix3d ric;
        Eigen::Vector3d gravity;

        // 所有帧
        std::vector<Frame> frame_window;
        // 关键帧，大小为Param.window_size
        std::vector<KeyFrameState> kf_state_window;
        // 预积分，大小为Param.window_size-1
        std::vector<ImuIntegrator> pre_int_window;
        // 滑窗中关键帧所观测到的特征点的集合
        std::vector<Feature> feature_window;
    };
}

#endif //GJT_VINS_VINS_DATA_H
