#include "feature_manager.h"
#include "log.h"

int FeaturePerId::endFrame() const {
    return start_frame_ + (int )feature_per_frame_.size() - 1;
}

FeatureManager::FeatureManager(Matrix3d _Rs[])
        : Rs(_Rs) {
    for (Matrix3d & i : ric)
        i.setIdentity();
}

void FeatureManager::setRic(Matrix3d _ric[]) {
    for (int i = 0; i < NUM_OF_CAM; i++) {
        ric[i] = _ric[i];
    }
}

void FeatureManager::clearState() {
    features_.clear();
}

int FeatureManager::getFeatureCount() {
    int cnt = 0;
    for (FeaturePerId &it: features_) {
        if (it.feature_per_frame_.size() >= 2 && it.start_frame_ < WINDOW_SIZE - 2) {
            cnt++;
        }
    }
    return cnt;
}


bool FeatureManager::addFeatureCheckParallax(int frame_count,
                                             const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image,
                                             double td) {
    LOG_D("input feature: %d, num of feature: %d", (int) image.size(), getFeatureCount());
    double parallax_sum = 0;
    int parallax_num = 0;
    last_track_num = 0;
    for (auto &id_pts: image) {
        FeaturePerFrame f_per_fra(id_pts.second[0].second, td);

        int feature_id = id_pts.first;
        auto it = find_if(features_.begin(), features_.end(), [feature_id](const FeaturePerId &it)->bool {
            return it.feature_id_ == feature_id;
        });

        if (it == features_.end()) {
            features_.emplace_back(FeaturePerId(feature_id, frame_count));
            features_.back().feature_per_frame_.push_back(f_per_fra);
        } else {
            it->feature_per_frame_.emplace_back(f_per_fra);
            last_track_num++;
        }
    }

    if (frame_count < 2 || last_track_num < 20)
        return true;

    for (const FeaturePerId &it_per_id: features_) {
        if (it_per_id.start_frame_ <= frame_count - 2 &&
            it_per_id.start_frame_ + int(it_per_id.feature_per_frame_.size()) - 1 >= frame_count - 1) {
            parallax_sum += compensatedParallax2(it_per_id, frame_count);
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

vector<pair<Vector3d, Vector3d>> FeatureManager::getCorresponding(int frame_count_l, int frame_count_r) {
    vector<pair<Vector3d, Vector3d>> corres;
    for (FeaturePerId &it: features_) {
        if (it.start_frame_ <= frame_count_l && it.endFrame() >= frame_count_r) {
            Vector3d a = Vector3d::Zero(), b = Vector3d::Zero();
            int idx_l = frame_count_l - it.start_frame_;
            int idx_r = frame_count_r - it.start_frame_;

            a = it.feature_per_frame_[idx_l].point_;

            b = it.feature_per_frame_[idx_r].point_;

            corres.emplace_back(make_pair(a, b));
        }
    }
    return corres;
}

void FeatureManager::setDepth(const VectorXd &x) {
    int feature_index = -1;
    for (auto &it_per_id: features_) {
        if (!(it_per_id.feature_per_frame_.size() >= 2 && it_per_id.start_frame_ < WINDOW_SIZE - 2))
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
        if (!(it_per_id.feature_per_frame_.size() >= 2 && it_per_id.start_frame_ < WINDOW_SIZE - 2))
            continue;
        it_per_id.estimated_depth = 1.0 / x(++feature_index);
    }
}

VectorXd FeatureManager::getDepthVector() {
    VectorXd dep_vec(getFeatureCount());
    int feature_index = -1;
    for (FeaturePerId &it_per_id: features_) {
        if (!(it_per_id.feature_per_frame_.size() >= 2 && it_per_id.start_frame_ < WINDOW_SIZE - 2))
            continue;
        dep_vec(++feature_index) = 1. / it_per_id.estimated_depth;
    }
    return dep_vec;
}

void FeatureManager::triangulate(Vector3d Ps[], Vector3d tic[], Matrix3d ric[]) {
    for (FeaturePerId &it_per_id: features_) {
        if (!(it_per_id.feature_per_frame_.size() >= 2 && it_per_id.start_frame_ < WINDOW_SIZE - 2))
            continue;

        if (it_per_id.estimated_depth > 0)
            continue;
        int imu_i = it_per_id.start_frame_, imu_j = imu_i - 1;

        assert(NUM_OF_CAM == 1);
        Eigen::MatrixXd svd_A(2 * it_per_id.feature_per_frame_.size(), 4);
        int svd_idx = 0;

        Eigen::Vector3d t0 = Ps[imu_i] + Rs[imu_i] * tic[0];
        Eigen::Matrix3d R0 = Rs[imu_i] * ric[0];

        for (FeaturePerFrame &it_per_frame: it_per_id.feature_per_frame_) {
            imu_j++;

            Eigen::Vector3d t1 = Ps[imu_j] + Rs[imu_j] * tic[0];
            Eigen::Matrix3d R1 = Rs[imu_j] * ric[0];
            Eigen::Vector3d t = R0.transpose() * (t1 - t0);
            Eigen::Matrix3d R = R0.transpose() * R1;
            Eigen::Matrix<double, 3, 4> P;
            P.leftCols<3>() = R.transpose();
            P.rightCols<1>() = -R.transpose() * t;
            Eigen::Vector3d f = it_per_frame.point_.normalized();
            svd_A.row(svd_idx++) = f[0] * P.row(2) - f[2] * P.row(0);
            svd_A.row(svd_idx++) = f[1] * P.row(2) - f[2] * P.row(1);

            if (imu_i == imu_j)
                continue;
        }
        assert(svd_idx == svd_A.rows());
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
        if (!it->feature_per_frame_.empty() && it->is_outlier_) {
            features_.erase(it);
        }
    }
}

void FeatureManager::removeBackShiftDepth() {
    for (auto it_next = features_.begin(); it_next!=features_.end();) {
        auto it = it_next;
        it_next++;
        if (it->start_frame_ != 0)
            it->start_frame_--;
        else {
            Eigen::Vector3d uv_i = it->feature_per_frame_[0].point_;
            it->feature_per_frame_.erase(it->feature_per_frame_.begin());
            if (it->feature_per_frame_.size() < 2) {
                features_.erase(it);
                continue;
            } else {
                Eigen::Vector3d pts_i = uv_i * it->estimated_depth;
                double dep_j = pts_i(2);
                if (dep_j > 0)
                    it->estimated_depth = dep_j;
                else
                    it->estimated_depth = INIT_DEPTH;
            }
        }
    }
}

void FeatureManager::removeBack() {
    for (auto it_next = features_.begin(); it_next != features_.end();) {
        auto it = it_next;
        it_next++;

        if (it->start_frame_ != 0)
            it->start_frame_--;
        else {
            it->feature_per_frame_.erase(it->feature_per_frame_.begin());
            if (it->feature_per_frame_.empty())
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
        } else {
            int j = WINDOW_SIZE - 1 - it->start_frame_;
            if (it->endFrame() < frame_count - 1)
                continue;
            it->feature_per_frame_.erase(it->feature_per_frame_.begin() + j);
            if (it->feature_per_frame_.empty())
                features_.erase(it);
        }
    }
}

double FeatureManager::compensatedParallax2(const FeaturePerId &it_per_id, int frame_count) {
    //check the second last frame is keyframe or not
    //parallax between second last frame and third last frame
    Vector3d p_i = it_per_id.feature_per_frame_[frame_count - 2 - it_per_id.start_frame_].point_;
    Vector3d p_j = it_per_id.feature_per_frame_[frame_count - 1 - it_per_id.start_frame_].point_;

    double u_j = p_j(0);
    double v_j = p_j(1);

    double u_i = p_i(0) / p_i(2);
    double v_i = p_i(1) / p_i(2);

    // todo tiemuhuaguo u_i - u_j有物理意义吗？？？
    double du = u_i - u_j, dv = v_i - v_j;

    return sqrt(du * du + dv * dv);
}