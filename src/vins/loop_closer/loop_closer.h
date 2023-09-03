#pragma once

#include <thread>
#include <mutex>
#include <queue>
#include <cassert>
#include <cstdio>
#include <string>

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "vins/vins_define_internal.h"
namespace vins{

    class LoopDetector;
    class KeyFrame;
    typedef std::shared_ptr<KeyFrame> KeyFramePtr;
    typedef const std::shared_ptr<const KeyFrame> ConstKeyFramePtr;

    class LoopCloser {
    public:
        LoopCloser();

        ~LoopCloser();

        void addKeyFrame(const Frame &base_frame,
                         const cv::Mat &_img,
                         const std::vector<cv::Point3f> &key_pts_3d,
                         const std::vector<cv::KeyPoint> &external_key_points_un_normalized,
                         const std::vector<cv::Point2f> &external_key_pts2d);

    private:
        [[noreturn]] void optimize4DoF();
        void optimize4DoFImpl();
        bool findLoop(const KeyFramePtr& cur_kf, int& peer_loop_id);

        std::vector<KeyFramePtr> key_frame_list_;
        std::vector<KeyFramePtr> key_frame_buffer_;
        std::mutex key_frame_buffer_mutex_;

        Eigen::Vector3d t_drift = Eigen::Vector3d::Zero();
        Eigen::Matrix3d r_drift = Eigen::Matrix3d::Identity();

        std::thread thread_optimize_;
        int loop_interval_lower_bound_ = -1;
        int loop_interval_upper_bound_ = -1;

        LoopDetector* loop_detector_{};
    };

}
