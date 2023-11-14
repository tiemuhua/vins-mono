#pragma once

#include <execinfo.h>
#include <csignal>

#include "opencv2/opencv.hpp"
#include <Eigen/Dense>

#include "param.h"
#include "camera_wrapper.h"
#include "impl/vins_model.h"

namespace vins { namespace FeatureTracker {
    void extractFeatures(const std::shared_ptr<cv::Mat> &_img,
                         double _cur_time,
                         const CameraWrapper& camera_wrapper,
                         const FrameTrackerParam& feature_tracker_param,
                         std::vector<FeaturePoint2D> &pts,
                         std::vector<cv::KeyPoint> &pts_raw,
                         FeatureTrackerModel &feature_tracker_model);
} }
