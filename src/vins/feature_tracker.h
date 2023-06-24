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

using namespace std;
using namespace camodocal;
using namespace Eigen;

bool inBorder(const cv::Point2f &pt);

namespace vins{
    class FeatureTracker
    {
    public:
        explicit FeatureTracker(const string &calib_file);

        struct FeaturesPerImage {
            std::vector<cv::Point2f> points;
            std::vector<cv::Point2f> unified_points;
            std::vector<cv::Point2f> points_velocity;
            std::vector<int> feature_ids;
        };

        FeaturesPerImage extractFeatures(const cv::Mat &_img, double _cur_time);

    private:
        cv::Point2f rawPoint2UniformedPoint(const cv::Point2f& p);

        cv::Mat prev_img_;
        vector<cv::Point2f> prev_pts_;
        vector<cv::Point2f> prev_uniformed_pts_;
        double prev_time_{};
        unordered_map<int, cv::Point2f> prev_feature_id_2_uniformed_points_map_;
        vector<int> feature_ids_;
        camodocal::CameraPtr m_camera_;

        static int s_feature_id_cnt_;
    };
}
