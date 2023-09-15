#pragma once

#include <execinfo.h>
#include <csignal>

#include "opencv2/opencv.hpp"
#include <Eigen/Dense>

#include "param.h"
#include "vins_define_internal.h"
#include "camera_wrapper.h"

namespace vins {
    class FeatureTracker {
    public:
        explicit FeatureTracker(Param *param, CameraWrapper *camera_wrapper);

        void extractFeatures(const cv::Mat &_img, double _cur_time,
                             std::vector<FeaturePoint2D> &pts,
                             std::vector<cv::KeyPoint> &pts_raw);

    private:
        cv::Mat prev_img_;
        std::vector<cv::Point2f> prev_raw_pts_;
        std::vector<cv::Point2f> prev_norm_pts_;
        double prev_time_{};
        std::unordered_map<int, cv::Point2f> prev_feature_id_2_norm_pts_;
        std::vector<int> feature_ids_;
        static Param *param_;
        // 与VinsCore中的camera_wrapper_是同一实例，由VinsCore负责控制生命周期
        CameraWrapper *camera_wrapper_;

        static int s_feature_id_cnt_;
    };
}
