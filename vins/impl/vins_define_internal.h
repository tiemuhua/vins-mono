//
// Created by gjt on 5/14/23.
//

#ifndef VINS_VINS_H
#define VINS_VINS_H

#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <utility>
#include <DBoW2/DBoW2.h>

#include "imu_integrator.h"

namespace vins {
#define Synchronized(mutex_)  for(ScopedLocker locker(mutex_); locker.cnt < 1; locker.cnt++)

    class ScopedLocker {
    public:
        explicit ScopedLocker(std::mutex &mutex) : guard(mutex) {}

        std::lock_guard<std::mutex> guard;
        int cnt = 0;
    };

    // point和velocity已经投影到归一化平面
    struct FeaturePoint2D {
        cv::Point2f point;
        cv::Point2f velocity;
        int feature_id;
    };

    struct Feature {
        int feature_id = -1;
        int start_kf_window_idx = -1;
        bool is_outlier = false;
        double inv_depth = -1;
        std::vector<cv::Point2f> points;
        std::vector<cv::Point2f> velocities;
        std::vector<double> time_stamps_ms;

        enum {
            kDepthUnknown,
            kDepthSolved,
            kDepthSolvedFail,
        } solve_flag_ = kDepthUnknown;

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

    constexpr double pi = 3.1415926;
}


#endif //VINS_VINS_H
