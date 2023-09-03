//
// Created by gjt on 5/14/23.
//

#ifndef VINS_VINS_H
#define VINS_VINS_H

#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include <utility>

namespace vins {
#define Synchronized(mutex_)  for(ScopedLocker locker(mutex_); locker.cnt < 1; locker.cnt++)

    class ScopedLocker {
    public:
        explicit ScopedLocker(std::mutex& mutex) : guard(mutex) {}
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
        int feature_id          = -1;
        int start_kf_window_idx        = -1;
        bool is_outlier         = false;
        double inv_depth        = -1;
        std::vector<cv::Point2f> points;
        std::vector<cv::Point2f> velocities;
        std::vector<double> time_stamps_ms;

        enum {
            kDepthUnknown,
            kDepthSolved,
            kDepthSolvedFail,
        }solve_flag_ = kDepthUnknown;

        Feature() = default;
        Feature(int _feature_id, int _start_frame)
                : feature_id(_feature_id), start_kf_window_idx(_start_frame) {}

        [[nodiscard]] int endFrame() const {
            return start_kf_window_idx + (int )points.size() - 1;
        }
    };

    class ImuIntegrator;
    class Frame {
    public:
        Frame() = default;
        Frame(const std::vector<FeaturePoint2D>& _features,
              std::shared_ptr<ImuIntegrator> _pre_integral,
              bool _is_key_frame) {
            for (const FeaturePoint2D &feature:_features) {
                points.emplace_back(feature.point);
                feature_ids.emplace_back(feature.feature_id);
            }
            pre_integral_ = std::move(_pre_integral);
            is_key_frame_ = _is_key_frame;
        };
        std::vector<cv::Point2f> points;
        std::vector<int> feature_ids;
        Eigen::Matrix3d R;
        Eigen::Vector3d T;
        std::shared_ptr<ImuIntegrator> pre_integral_;
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

        int window_idx = -1;     //.当window_idx递减至-1时，从loop_match_infos中移出.
        int peer_kf_id = -1;    //.匹配帧的ID，通过kfIdToKfListIdx获取匹配帧在key_frame_list_中的下标.
    };

    typedef const Eigen::Matrix3d & ConstMat3dRef;
    typedef const Eigen::Vector3d & ConstVec3dRef;
    typedef const Eigen::Quaterniond & ConstQuatRef;
    typedef Eigen::Matrix3d & Mat3dRef;
    typedef Eigen::Vector3d & Vec3dRef;
    typedef Eigen::Quaterniond & QuatRef;

    typedef std::vector<std::pair<cv::Point2f , cv::Point2f>> Correspondences;

    constexpr double pi = 3.1415926;
}


#endif //VINS_VINS_H
