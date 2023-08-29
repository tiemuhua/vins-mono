#include "feature_tracker.h"
#include "log.h"
#include "vins_utils.h"
#include "parameters.h"

using namespace vins;

int FeatureTracker::s_feature_id_cnt_ = 0;

static inline bool inBorder(const cv::Point2f &pt, int col, int row) {
    constexpr int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return  BORDER_SIZE <= img_x && img_x < col - BORDER_SIZE &&
            BORDER_SIZE <= img_y && img_y < row - BORDER_SIZE;
}

FeatureTracker::FeatureTracker(Param* param, CameraWrapper *camera_wrapper) {
    LOG_I("reading parameter of camera %s", param->camera.calib_file.c_str());
    param_ = param;
    camera_wrapper_ = camera_wrapper;
}

std::vector<FeaturePoint2D> FeatureTracker::extractFeatures(const cv::Mat &_img, double _cur_time){
    cv::Mat next_img = _img;
    if (prev_img_.empty()) {
        prev_img_ = _img;
    }

    std::vector<cv::Point2f> next_raw_pts;

    if (!prev_raw_pts_.empty()) {
        std::vector<uchar> status;
        std::vector<float> err;
        cv::Size winSize(21, 21);
        cv::calcOpticalFlowPyrLK(prev_img_, next_img, prev_raw_pts_, next_raw_pts, status, err, winSize, 3);

        for (int i = 0; i < int(next_raw_pts.size()); i++) {
            status[i] = status[i] && inBorder(next_raw_pts[i], param_->camera.col, param_->camera.row);
        }
        utils::reduceVector(prev_raw_pts_, status);
        utils::reduceVector(next_raw_pts, status);
        utils::reduceVector(prev_normalized_pts_, status);
        utils::reduceVector(feature_ids_, status);
    }

    std::vector<cv::Point2f> next_normalized_pts(next_raw_pts.size());
    for (const cv::Point2f &p: next_raw_pts) {
        next_normalized_pts.emplace_back(camera_wrapper_->rawPoint2UniformedPoint(p));
    }

    if (next_raw_pts.size() >= 8) {
        std::vector<uchar> mask;
        cv::findFundamentalMat(prev_normalized_pts_, next_normalized_pts, cv::FM_RANSAC,
                               param_->frame_tracker.fundamental_threshold, 0.99, mask);
        size_t size_a = prev_raw_pts_.size();
        utils::reduceVector(prev_raw_pts_, mask);
        utils::reduceVector(next_raw_pts, mask);
        utils::reduceVector(prev_normalized_pts_, mask);
        utils::reduceVector(next_normalized_pts, mask);
        utils::reduceVector(feature_ids_, mask);
        LOG_D("FM ransac: %zu -> %lu: %f", size_a, next_raw_pts.size(), 1.0 * next_raw_pts.size() / size_a);
    }

    // 去除过于密集的特征点，优先保留跟踪时间长的特征点，即next_pts中靠前的特征点
    cv::Mat mask = cv::Mat(param_->camera.row, param_->camera.col, CV_8UC1, cv::Scalar(255));
    for (const cv::Point2f & p: next_raw_pts) {
        if (mask.at<uchar>(p.x, p.y) == 255) {
            cv::circle(mask, p, param_->frame_tracker.min_dist, 0, -1);
        }
    }

    int max_new_pnt_num = param_->frame_tracker.max_cnt - static_cast<int>(next_raw_pts.size());
    if (max_new_pnt_num > 0) {
        std::vector<cv::Point2f> new_pts;
        constexpr double qualityLevel = 0.01;
        cv::goodFeaturesToTrack(next_img, new_pts, max_new_pnt_num, qualityLevel, param_->frame_tracker.min_dist, mask);
        for (auto &p: new_pts) {
            next_raw_pts.push_back(p);
            next_normalized_pts.emplace_back(camera_wrapper_->rawPoint2UniformedPoint(p));
            feature_ids_.push_back(s_feature_id_cnt_++);
        }
    }

    // calculate points velocity
    double dt = _cur_time - prev_time_;
    std::vector<cv::Point2f> pts_velocity;
    for (unsigned int i = 0; i < next_normalized_pts.size(); i++) {
        auto it = prev_feature_id_2_normalized_pts_.find(feature_ids_[i]);
        if (it != prev_feature_id_2_normalized_pts_.end()) {
            double v_x = (next_normalized_pts[i].x - it->second.x) / dt;
            double v_y = (next_normalized_pts[i].y - it->second.y) / dt;
            pts_velocity.emplace_back(cv::Point2f(v_x, v_y));
            continue;
        }
        pts_velocity.emplace_back(cv::Point2f(0, 0));
    }
    std::unordered_map<int, cv::Point2f> next_feature_id_2_uniformed_points_map;
    for (unsigned int i = 0; i < next_normalized_pts.size(); i++) {
        next_feature_id_2_uniformed_points_map[feature_ids_[i]] = next_normalized_pts[i];
    }

    prev_feature_id_2_normalized_pts_ = std::move(next_feature_id_2_uniformed_points_map);
    prev_img_ = std::move(next_img);
    prev_raw_pts_ = std::move(next_raw_pts);
    prev_time_ = _cur_time;
    prev_normalized_pts_ = std::move(next_normalized_pts);

    int feature_points_num = prev_raw_pts_.size();
    std::vector<FeaturePoint2D> feature_window(feature_points_num);
    for (int i = 0; i < feature_points_num; ++i) {
        feature_window[i].feature_id = feature_ids_[i];
        feature_window[i].point = prev_normalized_pts_[i];
        feature_window[i].velocity = pts_velocity[i];
    }
}
