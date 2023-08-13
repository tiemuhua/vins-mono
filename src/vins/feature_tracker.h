#pragma once

#include <cstdio>
#include <iostream>
#include <queue>
#include <execinfo.h>
#include <csignal>

#include "opencv2/opencv.hpp"
#include <Eigen/Dense>

#include "camodocal/camera_models/CameraFactory.h"
#include "camodocal/camera_models/CataCamera.h"
#include "camodocal/camera_models/PinholeCamera.h"

#include "parameters.h"
#include "vins_define_internal.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;

namespace vins{
    class FeatureTracker
    {
    public:
        explicit FeatureTracker(Param *param);

        std::vector<FeaturePoint2D> extractFeatures(const cv::Mat &_img, double _cur_time);

        cv::Point2f rawPoint2UniformedPoint(const cv::Point2f& p);

    private:
        cv::Mat prev_img_;
        vector<cv::Point2f> prev_raw_pts_;
        vector<cv::Point2f> prev_normalized_pts_;
        double prev_time_{};
        unordered_map<int, cv::Point2f> prev_feature_id_2_normalized_pts_;
        vector<int> feature_ids_;
        camodocal::CameraPtr camera_;
        Param* param_;

        static int s_feature_id_cnt_;
    };
}
