#include "feature_tracker.h"
#include "log.h"
#include "vins_utils.h"
#include "parameters.h"

using namespace vins;

int FeatureTracker::s_feature_id_cnt_ = 0;

bool inBorder(const cv::Point2f &pt) {
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return  BORDER_SIZE <= img_x &&
            img_x < Param::Instance().camera.col - BORDER_SIZE &&
            BORDER_SIZE <= img_y &&
            img_y < Param::Instance().camera.row - BORDER_SIZE;
}

FeatureTracker::FeatureTracker(const string &calib_file) {
    LOG_I("reading parameter of camera %s", calib_file.c_str());
    m_camera_ = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}

cv::Point2f FeatureTracker::rawPoint2UniformedPoint(const cv::Point2f& p) {
    Eigen::Vector3d tmp_p;
    m_camera_->liftProjective(Eigen::Vector2d(p.x, p.y), tmp_p);
    float col = Param::Instance().camera.focal * tmp_p.x() / tmp_p.z() + Param::Instance().camera.col / 2.0;
    float row = Param::Instance().camera.focal * tmp_p.y() / tmp_p.z() + Param::Instance().camera.row / 2.0;
    return {col, row};
}

FeatureTracker::FeaturesPerImage FeatureTracker::extractFeatures(const cv::Mat &_img, double _cur_time){
    int COL = Param::Instance().camera.col;
    int ROW = Param::Instance().camera.row;

    cv::Mat next_img = _img;
    if (prev_img_.empty()) {
        prev_img_ = _img;
    }

    vector<cv::Point2f> next_pts;

    if (!prev_pts_.empty()) {
        vector<uchar> status;
        vector<float> err;
        cv::Size winSize(21, 21);
        cv::calcOpticalFlowPyrLK(prev_img_, next_img, prev_pts_, next_pts, status, err, winSize, 3);

        for (int i = 0; i < int(next_pts.size()); i++) {
            status[i] = status[i] && inBorder(next_pts[i]);
        }
        utils::reduceVector(prev_pts_, status);
        utils::reduceVector(next_pts, status);
        utils::reduceVector(prev_uniformed_pts_, status);
        utils::reduceVector(feature_ids_, status);
    }

    vector<cv::Point2f> next_uniformed_pts(next_pts.size());
    for (const cv::Point2f &p: next_pts) {
        next_uniformed_pts.emplace_back(rawPoint2UniformedPoint(p));
    }

    if (next_pts.size() >= 8) {
        vector<uchar> mask;
        cv::findFundamentalMat(prev_uniformed_pts_, next_uniformed_pts, cv::FM_RANSAC,
                               Param::Instance().frame_tracker.fundamental_threshold, 0.99, mask);
        size_t size_a = prev_pts_.size();
        utils::reduceVector(prev_pts_, mask);
        utils::reduceVector(next_pts, mask);
        utils::reduceVector(prev_uniformed_pts_, mask);
        utils::reduceVector(next_uniformed_pts, mask);
        utils::reduceVector(feature_ids_, mask);
        LOG_D("FM ransac: %zu -> %lu: %f", size_a, next_pts.size(), 1.0 * next_pts.size() / size_a);
    }

    // 去除过于密集的特征点，优先保留跟踪时间长的特征点，即next_pts中靠前的特征点
    cv::Mat mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));
    for (const cv::Point2f & p: next_pts) {
        if (mask.at<uchar>(p.x, p.y) == 255) {
            cv::circle(mask, p, Param::Instance().frame_tracker.min_dist, 0, -1);
        }
    }

    int max_new_pnt_num = Param::Instance().frame_tracker.max_cnt - static_cast<int>(next_pts.size());
    if (max_new_pnt_num > 0) {
        vector<cv::Point2f> new_pts;
        constexpr double qualityLevel = 0.01;
        cv::goodFeaturesToTrack(next_img, new_pts, max_new_pnt_num, qualityLevel, Param::Instance().frame_tracker.min_dist, mask);
        for (auto &p: new_pts) {
            next_pts.push_back(p);
            next_uniformed_pts.emplace_back(rawPoint2UniformedPoint(p));
            feature_ids_.push_back(s_feature_id_cnt_++);
        }
    }

    // calculate points velocity
    double dt = _cur_time - prev_time_;
    vector<cv::Point2f> pts_velocity;
    for (unsigned int i = 0; i < next_uniformed_pts.size(); i++) {
        auto it = prev_feature_id_2_uniformed_points_map_.find(feature_ids_[i]);
        if (it != prev_feature_id_2_uniformed_points_map_.end()) {
            double v_x = (next_uniformed_pts[i].x - it->second.x) / dt;
            double v_y = (next_uniformed_pts[i].y - it->second.y) / dt;
            pts_velocity.emplace_back(cv::Point2f(v_x, v_y));
            continue;
        }
        pts_velocity.emplace_back(cv::Point2f(0, 0));
    }
    unordered_map<int, cv::Point2f> next_feature_id_2_uniformed_points_map;
    for (unsigned int i = 0; i < next_uniformed_pts.size(); i++) {
        next_feature_id_2_uniformed_points_map[feature_ids_[i]] = next_uniformed_pts[i];
    }

    prev_feature_id_2_uniformed_points_map_ = std::move(next_feature_id_2_uniformed_points_map);
    prev_img_ = std::move(next_img);
    prev_pts_ = std::move(next_pts);
    prev_time_ = _cur_time;
    prev_uniformed_pts_ = std::move(next_uniformed_pts);

    FeaturesPerImage features;
    features.points = prev_pts_;
    features.points_velocity = std::move(pts_velocity);
    features.unified_points = prev_uniformed_pts_;
    features.feature_ids = feature_ids_;
}
