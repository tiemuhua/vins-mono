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

    struct FeatureTrackerModel {
        std::shared_ptr<cv::Mat> prev_img;
        std::vector<cv::Point2f> prev_raw_pts;
        std::vector<cv::Point2f> prev_norm_pts;
        double prev_time{};
        std::unordered_map<int, cv::Point2f> prev_feature_id_2_norm_pts;
        std::vector<int> feature_ids;
        int feature_id_cnt = 0;
    };

    struct VinsModel {
        Eigen::Vector3d tic;
        Eigen::Matrix3d ric;
        Eigen::Vector3d gravity;

        // 所有帧
        std::vector<Frame> frame_window;
        // 关键帧，大小为Param.window_size，pos、rot、vel均为惯导状态
        std::vector<KeyFrameState> kf_state_window;
        // 预积分，大小为Param.window_size-1
        std::vector<ImuIntegralUniPtr> pre_int_window;
        // 滑窗中关键帧所观测到的特征点的集合
        std::vector<Feature> feature_window;
        // 滑动窗口中的回环
        std::vector<vins::LoopMatchInfo> loop_match_infos;

        PrevIMUState prev_imu_state;

        FeatureTrackerModel feature_tracker_model;

        ImuIntegralUniPtr kf_pre_integral_ptr_ = nullptr;

        double last_init_time_stamp_ = 0.0;
    };
}

#endif //GJT_VINS_VINS_DATA_H
