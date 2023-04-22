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
    FeatureTracker();

    struct FeatureTrackerReturn{
        vector<cv::Point2f> uniformed_points;
        vector<cv::Point2f> points;
        vector<int> ids;
        vector<cv::Point2f> vel;
    };

    FeatureTrackerReturn readImage(const cv::Mat &_img,double _cur_time);

private:
    bool updateID(unsigned int i);

    void readIntrinsicParameter(const string &calib_file);

    void showUndistortion(const string &name);

    cv::Point2f rawPoint2UniformedPoint(cv::Point2f p);

    cv::Mat prev_img_;
    vector<cv::Point2f> prev_pts_;
    vector<cv::Point2f> prev_uniformed_pts_;
    double prev_time_{};
    map<int, cv::Point2f> prev_feature_id_2_uniformed_points_map_;
    vector<int> feature_ids_;
    vector<int> track_cnt_;
    camodocal::CameraPtr m_camera_;

    static int s_feature_id_cnt_;
};
