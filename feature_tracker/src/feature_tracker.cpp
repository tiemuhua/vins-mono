#include "feature_tracker.h"
#include "log.h"

int FeatureTracker::s_feature_id_cnt_ = 0;

bool inBorder(const cv::Point2f &pt) {
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

template<typename T>
void reduceVector(vector<T> &v, vector<uchar> mask) {
    int j = 0;
    for (int i = 0; i < int(v.size()); i++)
        if (mask[i])
            v[j++] = v[i];
    v.resize(j);
}

FeatureTracker::FeatureTracker(const string &calib_file) {
    LOG_I("reading parameter of camera %s", calib_file.c_str());
    m_camera_ = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}

cv::Point2f FeatureTracker::rawPoint2UniformedPoint(cv::Point2f p) {
    Eigen::Vector3d tmp_p;
    m_camera_->liftProjective(Eigen::Vector2d(p.x, p.y), tmp_p);
    float col = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
    float row = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
    return {col, row};
}

FeatureTracker::FeaturesPerImage FeatureTracker::readImage(const cv::Mat &_img, double _cur_time){
    cv::Mat img;
    TicToc t_r;

    if (EQUALIZE) {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        TicToc t_c;
        clahe->apply(_img, img);
        LOG_D("CLAHE costs: %fms", t_c.toc());
    } else
        img = _img;

    cv::Mat next_img = img;
    if (prev_img_.empty()) {
        prev_img_ = img;
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
        reduceVector(prev_pts_, status);
        reduceVector(next_pts, status);
        reduceVector(prev_uniformed_pts_, status);
        reduceVector(feature_ids_, status);
    }

    vector<cv::Point2f> next_uniformed_pts(next_pts.size());
    for (const cv::Point2f &p: next_pts) {
        next_uniformed_pts.emplace_back(rawPoint2UniformedPoint(p));
    }

    if (PUB_THIS_FRAME) {
        if (next_pts.size() >= 8) {
            vector<uchar> mask;
            cv::findFundamentalMat(prev_uniformed_pts_, next_uniformed_pts,
                                   cv::FM_RANSAC, F_THRESHOLD, 0.99, mask);
            size_t size_a = prev_pts_.size();
            reduceVector(prev_pts_, mask);
            reduceVector(next_pts, mask);
            reduceVector(prev_uniformed_pts_, mask);
            reduceVector(next_uniformed_pts, mask);
            reduceVector(feature_ids_, mask);
            LOG_D("FM ransac: %zu -> %lu: %f", size_a, next_pts.size(), 1.0 * next_pts.size() / size_a);
        }

        // 去除过于密集的特征点，优先保留跟踪时间长的特征点，即next_pts中靠前的特征点
        cv::Mat mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));
        for (const cv::Point2f & p: next_pts) {
            if (mask.at<uchar>(p.x, p.y) == 255) {
                cv::circle(mask, p, MIN_DIST, 0, -1);
            }
        }

        int max_new_pnt_num = MAX_CNT - static_cast<int>(next_pts.size());
        if (max_new_pnt_num > 0) {
            vector<cv::Point2f> new_pts;
            constexpr double qualityLevel = 0.01;
            cv::goodFeaturesToTrack(next_img, new_pts, max_new_pnt_num, qualityLevel, MIN_DIST, mask);
            for (auto &p: new_pts) {
                next_pts.push_back(p);
                next_uniformed_pts.emplace_back(rawPoint2UniformedPoint(p));
                feature_ids_.push_back(s_feature_id_cnt_++);
            }
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
    map<int, cv::Point2f> next_feature_id_2_uniformed_points_map;
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

void FeatureTracker::showUndistortion(const string &name) {
    cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < COL; i++)
        for (int j = 0; j < ROW; j++) {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera_->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.emplace_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
        }
    for (int i = 0; i < int(undistortedp.size()); i++) {
        float col = undistortedp[i].x() * FOCAL_LENGTH + COL / 2;
        float row = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
        if (col + 300 >= 0 && col < COL + 300 && row + 300 >= 0 && row < ROW + 300) {
            undistortedImg.at<uchar>(row + 300, col + 300) =
                    prev_img_.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
    }
    cv::imshow(name, undistortedImg);
    cv::waitKey(0);
}
