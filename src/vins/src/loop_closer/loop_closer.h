#pragma once

#include <thread>
#include <mutex>
#include <queue>
#include <cassert>
#include <cstdio>
#include <string>

#include <opencv2/opencv.hpp>
#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "ThirdParty/DBoW/DBoW2.h"
#include "ThirdParty/DVision/DVision.h"
#include "ThirdParty/DBoW/TemplatedDatabase.h"
#include "ThirdParty/DBoW/TemplatedVocabulary.h"

#include "keyframe.h"

#define SHOW_S_EDGE false
#define SHOW_L_EDGE true
#define SAVE_LOOP_PATH true

namespace vins{

    class LoopCloser {
    public:
        LoopCloser();

        ~LoopCloser();

        void addKeyFrame(KeyFramePtr cur_kf, bool flag_detect_loop);

        void loadVocabulary(const std::string &voc_path);

    private:
        int _detectLoop(ConstKeyFramePtr keyframe, int frame_index) const;

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
