#include "feature_manager.h"
#include "log.h"

int FeaturePerId::endFrame() const {
    return start_frame_ + (int )feature_per_frames_.size() - 1;
}

void FeatureManager::clearState() {
    features_.clear();
}

int FeatureManager::getFeatureCount() {
    int cnt = 0;
    for (FeaturePerId &it: features_) {
        if (it.feature_per_frames_.size() >= 2 && it.start_frame_ < WINDOW_SIZE - 2) {
            cnt++;
        }
    }
    return cnt;
}

bool FeatureManager::addFeatureCheckParallax(int frame_id, const std::vector<FeaturePoint> &feature_points, double td) {
    LOG_D("input feature: %d, num of feature: %d", (int) feature_points.size(), getFeatureCount());
    double parallax_sum = 0;
    int parallax_num = 0;
    last_track_num = 0;

    for (const FeaturePoint &point: feature_points) {
        auto it = find_if(features_.begin(), features_.end(), [point](const FeaturePerId &it)->bool {
            return it.feature_id_ == point.feature_id;
        });

        if (it == features_.end()) {
            features_.emplace_back(FeaturePerId(point.feature_id, frame_id));
            features_.back().feature_per_frames_.push_back(point);
        } else {
            it->feature_per_frames_.emplace_back(point);
            last_track_num++;
        }
    }

    if (frame_id < 2 || last_track_num < 20)
        return true;

    for (const FeaturePerId &feature_per_id: features_) {
        if (feature_per_id.start_frame_ <= frame_id - 2 &&
                feature_per_id.start_frame_ + int(feature_per_id.feature_per_frames_.size()) >= frame_id) {
            parallax_sum += compensatedParallax2(feature_per_id, frame_id);
            parallax_num++;
        }
    }

    if (parallax_num == 0) {
        return true;
    } else {
        LOG_D("parallax_sum: %lf, parallax_num: %d", parallax_sum, parallax_num);
        LOG_D("current parallax: %lf", parallax_sum / parallax_num * FOCAL_LENGTH);
        return parallax_sum / parallax_num >= MIN_PARALLAX;
    }
}

vector<pair<cv::Point2f, cv::Point2f>> FeatureManager::getCorresponding(int frame_count_l, int frame_count_r) {
    vector<pair<cv::Point2f , cv::Point2f>> corres;
    for (FeaturePerId &it: features_) {
        if (it.start_frame_ <= frame_count_l && it.endFrame() >= frame_count_r) {
            int idx_l = frame_count_l - it.start_frame_;
            int idx_r = frame_count_r - it.start_frame_;
            cv::Point2f a = it.feature_per_frames_[idx_l].unified_point;
            cv::Point2f b = it.feature_per_frames_[idx_r].unified_point;
            corres.emplace_back(make_pair(a, b));
        }
    }
    return corres;
}

void FeatureManager::setDepth(const VectorXd &x) {
    int feature_index = -1;
    for (auto &it_per_id: features_) {
        if (!(it_per_id.feature_per_frames_.size() >= 2 && it_per_id.start_frame_ < WINDOW_SIZE - 2))
            continue;

        it_per_id.estimated_depth = 1.0 / x(++feature_index);
        if (it_per_id.estimated_depth < 0) {
            it_per_id.solve_flag_ = FeaturePerId::FeatureSolveFail;
        } else
            it_per_id.solve_flag_ = FeaturePerId::FeatureSolvedSucc;
    }
}

void FeatureManager::removeFailures() {
    for (auto it = features_.begin(), it_next = features_.begin();
         it != features_.end(); it = it_next) {
        it_next++;
        if (it->solve_flag_ == 2)
            features_.erase(it);
    }
}

void FeatureManager::clearDepth(const VectorXd &x) {
    int feature_index = -1;
    for (FeaturePerId &it_per_id: features_) {
        if (!(it_per_id.feature_per_frames_.size() >= 2 && it_per_id.start_frame_ < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth = 1.0 / x(++feature_index);
    }
}

VectorXd FeatureManager::getDepthVector() {
    VectorXd dep_vec(getFeatureCount());
    int feature_index = -1;
    for (FeaturePerId &it_per_id: features_) {
        if (!(it_per_id.feature_per_frames_.size() >= 2 && it_per_id.start_frame_ < WINDOW_SIZE - 2))
            continue;
        dep_vec(++feature_index) = 1. / it_per_id.estimated_depth;
    }
    return dep_vec;
}

void FeatureManager::triangulate(const PosWindow pos_window, const RotWindow rot_window,
                                 const Vector3d& tic, const Matrix3d &ric) {
    for (FeaturePerId &it_per_id: features_) {
        if (!(it_per_id.feature_per_frames_.size() >= 2 && it_per_id.start_frame_ < WINDOW_SIZE - 2))
            continue;

        if (it_per_id.estimated_depth > 0)
            continue;

        Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frames_.size(), 4);

        int imu_i = it_per_id.start_frame_;
        Eigen::Vector3d t0 = pos_window[imu_i] + rot_window[imu_i] * tic[0];
        Eigen::Matrix3d R0 = rot_window[imu_i] * ric[0];

        for (int i = 0; i < it_per_id.feature_per_frames_.size(); ++i) {
            int imu_j = it_per_id.start_frame_ + i;
            Eigen::Vector3d t1 = pos_window[imu_j] + rot_window[imu_j] * tic[0];
            Eigen::Matrix3d R1 = rot_window[imu_j] * ric[0];
            Eigen::Vector3d t = R0.transpose() * (t1 - t0);
            Eigen::Matrix3d R = R0.transpose() * R1;
            Eigen::Matrix<double, 3, 4> P;
            P.leftCols<3>() = R.transpose();
            P.rightCols<1>() = -R.transpose() * t;
            const cv::Point2f &unified_point = it_per_id.feature_per_frames_[i].unified_point;
            Eigen::Vector3d f = Eigen::Vector3d(unified_point.x, unified_point.y, 1.0).normalized();
            svd_A.row(2 * i) = f[0] * P.row(2) - f[2] * P.row(0);
            svd_A.row(2 * i + 1) = f[1] * P.row(2) - f[2] * P.row(1);
        }
        Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        double svd_method = svd_V[2] / svd_V[3];

        it_per_id.estimated_depth = svd_method;

        if (it_per_id.estimated_depth < 0.1) {
            it_per_id.estimated_depth = INIT_DEPTH;
        }
    }
}

void FeatureManager::removeOutlier() {
    for (auto it_next = features_.begin(); it_next != features_.end();) {
        auto it = it_next;
        it_next++;
        if (!it->feature_per_frames_.empty() && it->is_outlier_) {
            features_.erase(it);
        }
    }
}

void FeatureManager::removeBackShiftDepth() {
    for (auto it_next = features_.begin(); it_next!=features_.end();) {
        auto it = it_next;
        it_next++;
        if (it->start_frame_ != 0) {
            it->start_frame_--;
            continue;
        }
        const cv::Point2f &unified_point = it->feature_per_frames_[0].unified_point;
        Eigen::Vector3d uv_i = Eigen::Vector3d(unified_point.x, unified_point.y, 1.0);
        it->feature_per_frames_.erase(it->feature_per_frames_.begin());
        if (it->feature_per_frames_.size() < 2) {
            features_.erase(it);
            continue;
        }
        Eigen::Vector3d pts_i = uv_i * it->estimated_depth;
        double dep_j = pts_i(2);
        if (dep_j > 0)
            it->estimated_depth = dep_j;
        else
            it->estimated_depth = INIT_DEPTH;
    }
}

void FeatureManager::removeBack() {
    for (auto it_next = features_.begin(); it_next != features_.end();) {
        auto it = it_next;
        it_next++;

        if (it->start_frame_ != 0)
            it->start_frame_--;
        else {
            it->feature_per_frames_.erase(it->feature_per_frames_.begin());
            if (it->feature_per_frames_.empty())
                features_.erase(it);
        }
    }
}

void FeatureManager::removeFront(int frame_count) {
    for (auto it_next = features_.begin(); it_next != features_.end();) {
        auto it = it_next;
        it_next++;

        if (it->start_frame_ == frame_count) {
            it->start_frame_--;
            continue;
        }
        if (it->endFrame() < frame_count - 1) {
            continue;
        }
        int j = WINDOW_SIZE - 1 - it->start_frame_;
        it->feature_per_frames_.erase(it->feature_per_frames_.begin() + j);
        if (it->feature_per_frames_.empty())
            features_.erase(it);
    }
}

double FeatureManager::compensatedParallax2(const FeaturePerId &it_per_id, int frame_count) {
    //check the second last frame is keyframe or not
    //parallax between second last frame and third last frame
    cv::Point2f p_i = it_per_id.feature_per_frames_[frame_count - 2 - it_per_id.start_frame_].unified_point;
    cv::Point2f p_j = it_per_id.feature_per_frames_[frame_count - 1 - it_per_id.start_frame_].unified_point;
    return cv::norm(p_i - p_j);
}