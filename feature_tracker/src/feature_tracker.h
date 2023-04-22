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
        vector<cv::Point2f> cur_un_pts;
        vector<cv::Point2f> cur_pts;
        vector<int> ids;
        vector<cv::Point2f> pts_velocity;
    };

    FeatureTrackerReturn readImage(const cv::Mat &_img,double _cur_time);

private:
    bool updateID(unsigned int i);

    void readIntrinsicParameter(const string &calib_file);

    void showUndistortion(const string &name);

    void undistortedPoints();

    cv::Mat cur_img;
    vector<cv::Point2f> cur_pts;
    vector<int> ids;
    vector<int> track_cnt;
    map<int, cv::Point2f> prev_un_pts_map;
    camodocal::CameraPtr m_camera;
    double prev_time{};

    static int s_feature_id_cnt_;
};
