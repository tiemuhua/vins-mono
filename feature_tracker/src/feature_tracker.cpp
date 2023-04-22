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

FeatureTracker::FeatureTracker() = default;

FeatureTracker::FeatureTrackerReturn FeatureTracker::readImage(const cv::Mat &_img, double _cur_time){
    cv::Mat img;
    TicToc t_r;

    if (EQUALIZE) {
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3.0, cv::Size(8, 8));
        TicToc t_c;
        clahe->apply(_img, img);
        LOG_D("CLAHE costs: %fms", t_c.toc());
    } else
        img = _img;

    cv::Mat forw_img = img;
    if (cur_img.empty()) {
        cur_img = forw_img = img;
    }

    vector<cv::Point2f> forw_pts;

    if (!cur_pts.empty()) {
        vector<uchar> status;
        vector<float> err;
        cv::Size winSize(21, 21);
        cv::calcOpticalFlowPyrLK(cur_img, forw_img, cur_pts, forw_pts, status, err, winSize, 3);

        for (int i = 0; i < int(forw_pts.size()); i++) {
            status[i] = status[i] && inBorder(forw_pts[i]);
        }
        reduceVector(cur_pts, status);
        reduceVector(forw_pts, status);
        reduceVector(ids, status);
        reduceVector(track_cnt, status);
    }

    for (auto &n: track_cnt)
        n++;

    if (PUB_THIS_FRAME) {
        if (forw_pts.size() >= 8) {
            vector<cv::Point2f> un_cur_pts(cur_pts.size()), un_forw_pts(forw_pts.size());
            for (unsigned int i = 0; i < cur_pts.size(); i++) {
                Eigen::Vector3d tmp_p;
                m_camera->liftProjective(Eigen::Vector2d(cur_pts[i].x, cur_pts[i].y), tmp_p);
                double cur_row = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
                double cur_col = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
                un_cur_pts[i] = cv::Point2f(cur_row, cur_col);

                m_camera->liftProjective(Eigen::Vector2d(forw_pts[i].x, forw_pts[i].y), tmp_p);
                double forw_row = FOCAL_LENGTH * tmp_p.x() / tmp_p.z() + COL / 2.0;
                double forw_col = FOCAL_LENGTH * tmp_p.y() / tmp_p.z() + ROW / 2.0;
                un_forw_pts[i] = cv::Point2f(forw_row, forw_col);
            }

            vector<uchar> mask;
            cv::findFundamentalMat(un_cur_pts, un_forw_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, mask);
            size_t size_a = cur_pts.size();
            reduceVector(cur_pts, mask);
            reduceVector(forw_pts, mask);
            reduceVector(ids, mask);
            reduceVector(track_cnt, mask);
            LOG_D("FM ransac: %d -> %lu: %f", size_a, forw_pts.size(), 1.0 * forw_pts.size() / size_a);
        }

        // prefer to keep features that are tracked for long time
        vector<std::tuple<int, cv::Point2f, int>> cnt_pts_id;
        for (unsigned int i = 0; i < forw_pts.size(); i++)
            cnt_pts_id.emplace_back(track_cnt[i], forw_pts[i], ids[i]);
        sort(cnt_pts_id.begin(), cnt_pts_id.end(),[](const auto &a, const auto &b) {
            return std::get<0>(a) > std::get<0>(b);
        });
        forw_pts.clear();
        ids.clear();
        track_cnt.clear();
        for (auto &it: cnt_pts_id) {
            track_cnt.emplace_back(std::get<0>(it));
            forw_pts.emplace_back(std::get<1>(it));
            ids.emplace_back(std::get<2>(it));
        }

        cv::Mat mask = cv::Mat(ROW, COL, CV_8UC1, cv::Scalar(255));
        for (const auto &pt:forw_pts) {
            cv::circle(mask, pt, MIN_DIST, 0, -1);
        }

        int n_max_cnt = MAX_CNT - static_cast<int>(forw_pts.size());
        if (n_max_cnt > 0) {
            vector<cv::Point2f> n_pts;
            if (mask.type() != CV_8UC1) {
                LOG_E("wrong mask type:%d", mask.type());
            }
            if (mask.size() != forw_img.size()) {
                LOG_E("wrong mask size, width:%d, height:%d", mask.size().width, mask.size().height);
            }
            constexpr double qualityLevel = 0.01;
            cv::goodFeaturesToTrack(forw_img, n_pts, MAX_CNT - forw_pts.size(), qualityLevel, MIN_DIST, mask);
            for (auto &p: n_pts) {
                forw_pts.push_back(p);
                ids.push_back(-1);
                track_cnt.push_back(1);
            }
        }
    }
    vector<cv::Point2f> forw_un_pts;
    map<int, cv::Point2f> cur_un_pts_map;
    for (unsigned int i = 0; i < forw_pts.size(); i++) {
        Eigen::Vector2d a(forw_pts[i].x, forw_pts[i].y);
        Eigen::Vector3d b;
        m_camera->liftProjective(a, b);
        cv::Point2f p(b.x() / b.z(), b.y() / b.z());
        forw_un_pts.emplace_back(p);
        cur_un_pts_map[ids[i]] = p;
    }

    // calculate points velocity
    double dt = _cur_time - prev_time;
    vector<cv::Point2f> pts_velocity;
    for (unsigned int i = 0; i < forw_un_pts.size(); i++) {
        if (ids[i] != -1) {
            auto it = prev_un_pts_map.find(ids[i]);
            if (it != prev_un_pts_map.end()) {
                double v_x = (forw_un_pts[i].x - it->second.x) / dt;
                double v_y = (forw_un_pts[i].y - it->second.y) / dt;
                pts_velocity.emplace_back(cv::Point2f(v_x, v_y));
                continue;
            }
        }
        pts_velocity.emplace_back(cv::Point2f(0, 0));
    }

    prev_un_pts_map = cur_un_pts_map;
    cur_img = forw_img;
    cur_pts = forw_pts;
    prev_time = _cur_time;
}

bool FeatureTracker::updateID(unsigned int i) {
    if (i >= ids.size()) {
        return false;
    }
    if (ids[i] == -1) {
        ids[i] = s_feature_id_cnt_++;
    }
    return true;
}

void FeatureTracker::readIntrinsicParameter(const string &calib_file) {
    LOG_I("reading paramerter of camera %s", calib_file.c_str());
    m_camera = CameraFactory::instance()->generateCameraFromYamlFile(calib_file);
}

void FeatureTracker::showUndistortion(const string &name) {
    cv::Mat undistortedImg(ROW + 600, COL + 600, CV_8UC1, cv::Scalar(0));
    vector<Eigen::Vector2d> distortedp, undistortedp;
    for (int i = 0; i < COL; i++)
        for (int j = 0; j < ROW; j++) {
            Eigen::Vector2d a(i, j);
            Eigen::Vector3d b;
            m_camera->liftProjective(a, b);
            distortedp.push_back(a);
            undistortedp.emplace_back(Eigen::Vector2d(b.x() / b.z(), b.y() / b.z()));
        }
    for (int i = 0; i < int(undistortedp.size()); i++) {
        cv::Mat pp(3, 1, CV_32FC1);
        pp.at<float>(0, 0) = undistortedp[i].x() * FOCAL_LENGTH + COL / 2;
        pp.at<float>(1, 0) = undistortedp[i].y() * FOCAL_LENGTH + ROW / 2;
        pp.at<float>(2, 0) = 1.0;
        if (pp.at<float>(1, 0) + 300 >= 0 && pp.at<float>(1, 0) + 300 < ROW + 600 && pp.at<float>(0, 0) + 300 >= 0 &&
            pp.at<float>(0, 0) + 300 < COL + 600) {
            undistortedImg.at<uchar>(pp.at<float>(1, 0) + 300, pp.at<float>(0, 0) + 300) =
                    cur_img.at<uchar>(distortedp[i].y(), distortedp[i].x());
        }
    }
    cv::imshow(name, undistortedImg);
    cv::waitKey(0);
}
