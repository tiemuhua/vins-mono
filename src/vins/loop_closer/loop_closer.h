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
    class BriefExtractor;
    class KeyFrame;
    typedef std::shared_ptr<KeyFrame> KeyFramePtr;
    typedef const std::shared_ptr<const KeyFrame> ConstKeyFramePtr;

    class LoopCloser {
    public:
        LoopCloser();

        ~LoopCloser();

        void addKeyFrame(const vins::Frame &base_frame, const cv::Mat &_image,
                         const std::vector<cv::Point3f> &_point_3d,
                         const std::vector<cv::Point2f> &_point_2d_uv);

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
        BriefExtractor* brief_extractor_{};
    };

}
