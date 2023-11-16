#include "feature_tracker.h"
#include <glog/logging.h>
#include "vins_utils.h"
#include "param.h"
#include "impl/vins_model.h"

using namespace vins;

static inline bool inBorder(const cv::Point2f &pt, int cols, int rows) {
    constexpr int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < cols - BORDER_SIZE &&
           BORDER_SIZE <= img_y && img_y < rows - BORDER_SIZE;
}

void FeatureTracker::extractFeatures(const std::shared_ptr<cv::Mat> &_img,
                                     const double _cur_time,
                                     const CameraWrapper& camera_wrapper,
                                     const FrameTrackerParam& feature_tracker_param,
                                     std::vector<FeaturePoint2D> &pts,
                                     std::vector<cv::KeyPoint> &pts_raw,
                                     PrevImgFeatureInfo &prev_img_feature_info) {
    const int cols = camera_wrapper.camera_->imageWidth();
    const int rows = camera_wrapper.camera_->imageHeight();
    assert(_img != nullptr);
    std::shared_ptr<cv::Mat> next_img = _img;
    if (prev_img_feature_info.img == nullptr) {
        prev_img_feature_info.img = _img;
    }

    std::vector<cv::Point2f> next_raw_pts;

    if (!prev_img_feature_info.raw_pts.empty()) {
        std::vector<uchar> status;
        std::vector<float> err;
        cv::Size winSize(21, 21);
        cv::calcOpticalFlowPyrLK(*(prev_img_feature_info.img),
                                 *next_img,
                                 prev_img_feature_info.raw_pts,
                                 next_raw_pts,
                                 status,
                                 err,
                                 winSize,
                                 3);

        for (int i = 0; i < int(next_raw_pts.size()); i++) {
            status[i] = status[i] && inBorder(next_raw_pts[i], cols, rows);
        }
        utils::reduceVector(prev_img_feature_info.raw_pts, status);
        utils::reduceVector(next_raw_pts, status);
        utils::reduceVector(prev_img_feature_info.norm_pts, status);
        utils::reduceVector(prev_img_feature_info.feature_ids, status);
    }

    std::vector<cv::Point2f> next_norm_pts(next_raw_pts.size());
    for (int i = 0; i < next_raw_pts.size(); ++i) {
        next_norm_pts[i] = camera_wrapper.rawPoint2NormPoint(next_raw_pts[i]);
    }

    if (next_raw_pts.size() >= 8) {
        std::vector<uchar> mask;
        cv::findFundamentalMat(prev_img_feature_info.norm_pts, next_norm_pts, cv::FM_RANSAC,
                               feature_tracker_param.fundamental_threshold, 0.99, mask);
        utils::reduceVector(prev_img_feature_info.raw_pts, mask);
        utils::reduceVector(next_raw_pts, mask);
        utils::reduceVector(prev_img_feature_info.norm_pts, mask);
        utils::reduceVector(next_norm_pts, mask);
        utils::reduceVector(prev_img_feature_info.feature_ids, mask);
        LOG(INFO) << "FM ransac: prev_raw_pts_ size:" << prev_img_feature_info.raw_pts.size() << ", next_raw_pts.size:" << next_raw_pts.size();
    }

    // 去除过于密集的特征点，优先保留跟踪时间长的特征点，即next_pts中靠前的特征点
    cv::Mat mask = cv::Mat(rows, cols, CV_8UC1, cv::Scalar(255));
    for (const cv::Point2f &p: next_raw_pts) {
        if (mask.at<uchar>(p.x, p.y) == 255) {
            cv::circle(mask, p, feature_tracker_param.min_dist, 0, -1);
        }
    }

    int max_new_pnt_num = feature_tracker_param.max_cnt - static_cast<int>(next_raw_pts.size());
    if (max_new_pnt_num > 0) {
        std::vector<cv::Point2f> new_pts;
        constexpr double qualityLevel = 0.01;
        cv::goodFeaturesToTrack(*next_img,
                                new_pts,
                                max_new_pnt_num,
                                qualityLevel,
                                feature_tracker_param.min_dist,
                                mask);
        for (auto &p: new_pts) {
            next_raw_pts.push_back(p);
            next_norm_pts.emplace_back(camera_wrapper.rawPoint2NormPoint(p));
            prev_img_feature_info.feature_ids.push_back(prev_img_feature_info.feature_id_cnt++);
        }
    }

    // calculate points velocity
    double dt = _cur_time - prev_img_feature_info.time;
    std::vector<cv::Point2f> pts_velocity;
    for (unsigned int i = 0; i < next_norm_pts.size(); i++) {
        auto it = prev_img_feature_info.feature_id_2_norm_pts.find(prev_img_feature_info.feature_ids[i]);
        if (it != prev_img_feature_info.feature_id_2_norm_pts.end()) {
            double v_x = (next_norm_pts[i].x - it->second.x) / dt;
            double v_y = (next_norm_pts[i].y - it->second.y) / dt;
            pts_velocity.emplace_back(cv::Point2f(v_x, v_y));
            continue;
        }
        pts_velocity.emplace_back(cv::Point2f(0, 0));
    }
    std::unordered_map<int, cv::Point2f> next_feature_id_2_norm_pts;
    for (unsigned int i = 0; i < next_norm_pts.size(); i++) {
        next_feature_id_2_norm_pts[prev_img_feature_info.feature_ids[i]] = next_norm_pts[i];
    }

    prev_img_feature_info.feature_id_2_norm_pts = std::move(next_feature_id_2_norm_pts);
    prev_img_feature_info.img = std::move(next_img);
    prev_img_feature_info.raw_pts = std::move(next_raw_pts);
    prev_img_feature_info.time = _cur_time;
    prev_img_feature_info.norm_pts = std::move(next_norm_pts);

    int feature_points_num = (int )prev_img_feature_info.raw_pts.size();
    pts = std::vector<FeaturePoint2D>(feature_points_num);
    for (int i = 0; i < feature_points_num; ++i) {
        pts[i].feature_id = prev_img_feature_info.feature_ids[i];
        pts[i].point = prev_img_feature_info.norm_pts[i];
        pts[i].velocity = pts_velocity[i];
    }
    pts_raw = std::vector<cv::KeyPoint>(prev_img_feature_info.raw_pts.size());
    for (int i = 0; i < prev_img_feature_info.raw_pts.size(); ++i) {
        pts_raw[i].pt = prev_img_feature_info.raw_pts[i];
    }
}
