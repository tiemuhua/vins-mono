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

#include "impl/vins_model.h"
#include "keyframe.h"

namespace vins {

    class LoopDetector;

    class LoopCloser {
    public:
        LoopCloser();

        void addKeyFrame(KeyFrameUniPtr kf_ptr);

        bool findLoop(KeyFrame &kf, LoopMatchInfo &info);

    private:
        [[noreturn]] void optimize4DoF();

        void optimize4DoFImpl();

        std::vector<KeyFrameUniPtr> key_frame_list_;
        std::vector<KeyFrameUniPtr> key_frame_buffer_;
        std::mutex key_frame_buffer_mutex_;

        Eigen::Vector3d t_drift = Eigen::Vector3d::Zero();
        Eigen::Matrix3d r_drift = Eigen::Matrix3d::Identity();

        int loop_interval_lower_bound_ = -1;
        int loop_interval_upper_bound_ = -1;

        LoopDetector *loop_detector_{};
    };

}
