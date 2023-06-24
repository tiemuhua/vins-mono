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

#include "DBoW2/DBoW2.h"
#include "DVision/DVision.h"
#include "DBoW2/TemplatedDatabase.h"
#include "DBoW2/TemplatedVocabulary.h"

#include "keyframe.h"

namespace vins{

    class LoopCloser {
    public:
        LoopCloser();

        ~LoopCloser();

        void addKeyFrame(const KeyFramePtr& cur_kf, bool flag_detect_loop);

        void loadVocabulary(const std::string &voc_path);

    private:
        [[nodiscard]] int _detectLoop(ConstKeyFramePtr& keyframe, int frame_index) const;

        void optimize4DoF();

        std::vector<KeyFramePtr> key_frame_list_;
        std::mutex key_frame_list_mutex_;

        Eigen::Vector3d t_drift = Eigen::Vector3d::Zero();
        Eigen::Matrix3d r_drift = Eigen::Matrix3d::Identity();
        std::mutex drift_mutex_;

        std::thread thread_optimize_;
        int loop_interval_lower_bound_ = -1;
        int loop_interval_upper_bound_ = -1;
        std::mutex loop_interval_mutex_;

        BriefDatabase db;
        BriefVocabulary *voc{};
    };

}
