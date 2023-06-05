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

        void addKeyFrame(KeyFrame *cur_kf, bool flag_detect_loop);

        void loadVocabulary(const std::string &voc_path);

        void updateKeyFrameLoop(int index, Eigen::Matrix<double, 8, 1> &_loop_info);

        KeyFrame *getKeyFrame(int index);

        Eigen::Vector3d t_drift = Eigen::Vector3d::Zero();
        double yaw_drift = 0;
        Eigen::Matrix3d r_drift = Eigen::Matrix3d::Identity();

    private:
        int detectLoop(KeyFrame *keyframe, int frame_index);

        void _addKeyFrameIntoVoc(KeyFrame *keyframe);

        void optimize4DoF();

        std::vector<KeyFrame *> keyframelist_;
        std::mutex m_keyframelist;
        std::mutex m_optimize_buf;
        std::mutex m_path;
        std::mutex m_drift;
        std::thread t_optimization;
        std::queue<int> optimize_buf;

        map<int, cv::Mat> image_pool;
        int earliest_loop_index = -1;

        BriefDatabase db;
        BriefVocabulary *voc{};

    };

}
