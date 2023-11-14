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

namespace vins {

    // point和velocity已经投影到归一化平面
    struct FeaturePoint2D {
        cv::Point2f point;
        cv::Point2f velocity;
        int feature_id;
    };

    struct Feature {
        int feature_id = -1;
        int start_kf_window_idx = -1;
        double inv_depth = -1;
        std::vector<cv::Point2f> points;
        std::vector<cv::Point2f> velocities;
        std::vector<double> time_stamps_ms;

        Feature() = default;
        Feature(int _feature_id, int _start_frame)
                : feature_id(_feature_id), start_kf_window_idx(_start_frame) {}

        [[nodiscard]] int endFrame() const {
            return start_kf_window_idx + (int) points.size() - 1;
        }
    };

    class Frame {
    public:
        Frame() = default;
        Frame(const std::vector<FeaturePoint2D> &_features,
              ImuIntegralUniPtr _imu_integral,
              bool _is_key_frame,
              double ts) {
            for (const FeaturePoint2D &feature: _features) {
                points.emplace_back(feature.point);
                feature_ids.emplace_back(feature.feature_id);
            }
            imu_integral_ = std::move(_imu_integral);
            is_key_frame_ = _is_key_frame;
            time_stamp = ts;
        };
        std::vector<cv::Point2f> points;
        std::vector<int> feature_ids;
        Eigen::Matrix3d imu_rot;
        Eigen::Vector3d imu_pos;
        ImuIntegralUniPtr imu_integral_;
        bool is_key_frame_ = false;
        double time_stamp = -1;
    };

    struct LoopMatchInfo {
        //.特征点在当前帧中的ID.
        std::vector<int> feature_ids;
        //.特征点在匹配帧中的二维坐标.
        std::vector<cv::Point2f> peer_pts;

        //.匹配帧的位置与姿态.
        Eigen::Vector3d peer_pos;
        Eigen::Matrix3d peer_rot;

        size_t window_idx = -1;     //.当window_idx递减至-1时，从loop_match_infos中移出.
    };

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
