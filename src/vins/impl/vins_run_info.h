//
// Created by gjt on 6/10/23.
//

#ifndef GJT_VINS_VINS_DATA_H
#define GJT_VINS_VINS_DATA_H

#include <vector>

#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include <DVision/BRIEF.h>
#include <camodocal/camera_models/Camera.h>

#include "imu_integrator.h"
#include "vins_define_internal.h"

namespace vins {
    struct KeyFrameState {
        Eigen::Vector3d pos = Eigen::Vector3d::Zero();
        Eigen::Matrix3d rot = Eigen::Matrix3d::Identity();
        Eigen::Vector3d vel = Eigen::Vector3d::Zero();
        Eigen::Vector3d ba = Eigen::Vector3d::Zero();
        Eigen::Vector3d bg = Eigen::Vector3d::Zero();
        double time_stamp;
    };

    struct RunInfo {
        Eigen::Vector3d tic;
        Eigen::Matrix3d ric;
        Eigen::Vector3d gravity;

        // 所有帧
        std::vector<Frame> frame_window;
        // 关键帧，大小为Param.window_size，pos、rot、vel均为惯导状态
        std::vector<KeyFrameState> kf_state_window;
        // 预积分，大小为Param.window_size-1
        std::vector<ImuIntegratorPtr> pre_int_window;
        // 滑窗中关键帧所观测到的特征点的集合
        std::vector<Feature> feature_window;
        // 滑动窗口中的回环
        std::vector<vins::LoopMatchInfo> loop_match_infos;

        std::vector<Correspondences> neighbor_kf_correspondence_pts;

        PrevIMUState prev_imu_state;
    };
}

#endif //GJT_VINS_VINS_DATA_H
