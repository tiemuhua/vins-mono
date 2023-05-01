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
#include "tic_toc.h"

using namespace std;
using namespace camodocal;
using namespace Eigen;

bool inBorder(const cv::Point2f &pt);

void reduceVector(vector<cv::Point2f> &v, vector<uchar> status);
void reduceVector(vector<int> &v, vector<uchar> status);

class FeatureTracker
{
  public:
    FeatureTracker(const string &calib_file);

    struct FeaturesPerImage {
        std::vector<cv::Point2f> points;
        std::vector<cv::Point2f> unified_points;
        std::vector<cv::Point2f> points_velocity;
        std::vector<int> feature_ids;
    };

    FeaturesPerImage readImage(const cv::Mat &_img,double _cur_time);

private:
    void readIntrinsicParameter(const string &calib_file);

    void showUndistortion(const string &name);

    cv::Point2f rawPoint2UniformedPoint(cv::Point2f p);

    cv::Mat prev_img_;
    vector<cv::Point2f> prev_pts_;
    vector<cv::Point2f> prev_uniformed_pts_;
    double prev_time_{};
    map<int, cv::Point2f> prev_feature_id_2_uniformed_points_map_;
    vector<int> feature_ids_;
    camodocal::CameraPtr m_camera_;

    static int s_feature_id_cnt_;
};
