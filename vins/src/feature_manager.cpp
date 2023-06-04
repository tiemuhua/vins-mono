#include "feature_manager.h"
#include "log.h"
#define WINDOW_SIZE 100
#define FOCAL_LENGTH 10
#define MIN_PARALLAX 1
namespace vins {
    using namespace std;
    using namespace Eigen;
    void FeatureManager::clearState() {
        features_.clear();
    }

    bool FeatureManager::addFeatureCheckParallax(int frame_id, const std::vector<FeaturePoint> &feature_points) {
        LOG_D("input feature: %d, num of feature: %d", (int) feature_points.size(), (int )features_.size());
        double parallax_sum = 0;
        int parallax_num = 0;
        last_track_num = 0;

        for (const FeaturePoint &point: feature_points) {
            auto it = find_if(features_.begin(), features_.end(), [point](const FeaturesOfId &it)->bool {
                return it.feature_id_ == point.feature_id;
            });

            if (it == features_.end()) {
                features_.emplace_back(FeaturesOfId(point.feature_id, frame_id));
                features_.back().feature_points_.push_back(point);
            } else {
                it->feature_points_.emplace_back(point);
                last_track_num++;
            }
        }

        if (frame_id < 2 || last_track_num < 20)
            return true;

        for (const FeaturesOfId &feature_per_id: features_) {
            if (feature_per_id.start_frame_ <= frame_id - 2 &&
                feature_per_id.start_frame_ + int(feature_per_id.feature_points_.size()) >= frame_id) {
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

    vector<pair<cv::Point2f, cv::Point2f>> FeatureManager::getCorresponding(int frame_count_l, int frame_count_r) const {
        vector<pair<cv::Point2f , cv::Point2f>> corres;
        for (const FeaturesOfId &it: features_) {
            if (it.start_frame_ <= frame_count_l && it.endFrame() >= frame_count_r) {
                int idx_l = frame_count_l - it.start_frame_;
                int idx_r = frame_count_r - it.start_frame_;
                cv::Point2f a = it.feature_points_[idx_l].unified_point;
                cv::Point2f b = it.feature_points_[idx_r].unified_point;
                corres.emplace_back(make_pair(a, b));
            }
        }
        return corres;
    }

    void FeatureManager::setInvDepth(const VectorXd &x) {
        for (int i = 0; i < features_.size(); ++i) {
            features_[i].inv_depth = x(i);
            if (features_[i].inv_depth < 0) {
                features_[i].solve_flag_ = FeaturesOfId::FeatureSolveFail;
            } else
                features_[i].solve_flag_ = FeaturesOfId::FeatureSolvedSucc;
        }
    }

    void FeatureManager::removeFailures() {
        for (auto it = features_.begin(), it_next = features_.begin();
             next(it) != features_.end(); it = it_next) {
            it_next++;
            if (it->solve_flag_ == FeaturesOfId::FeatureSolveFail)
                features_.erase(it);
        }
    }

    void FeatureManager::clearDepth() {
        for (FeaturesOfId &it_per_id: features_) {
            it_per_id.inv_depth = -1.0;
        }
    }

    std::vector<double> FeatureManager::getInvDepth() const {
        std::vector<double> dep_vec(features_.size());
        for (int i = 0; i < features_.size(); ++i) {
            dep_vec[i] = features_[i].inv_depth;
        }
        return dep_vec;
    }

    void FeatureManager::triangulate(const PosWindow& pos_window, const RotWindow& rot_window,
                                     const Vector3d& tic, const Matrix3d &ric) {
        for (FeaturesOfId &it_per_id: features_) {
            if (!(it_per_id.feature_points_.size() >= 2 && it_per_id.start_frame_ < WINDOW_SIZE - 2))
                continue;

            if (it_per_id.inv_depth > 0)
                continue;

            Eigen::MatrixXd svd_A(2 * it_per_id.feature_points_.size(), 4);

            int imu_i = it_per_id.start_frame_;
            Eigen::Vector3d t0 = pos_window[imu_i] + rot_window[imu_i] * tic;
            Eigen::Matrix3d R0 = rot_window[imu_i] * ric;

            for (int i = 0; i < it_per_id.feature_points_.size(); ++i) {
                int imu_j = it_per_id.start_frame_ + i;
                Eigen::Vector3d t1 = pos_window[imu_j] + rot_window[imu_j] * tic;
                Eigen::Matrix3d R1 = rot_window[imu_j] * ric;
                Eigen::Vector3d t = R0.transpose() * (t1 - t0);
                Eigen::Matrix3d R = R0.transpose() * R1;
                Eigen::Matrix<double, 3, 4> P;
                P.leftCols<3>() = R.transpose();
                P.rightCols<1>() = -R.transpose() * t;
                const cv::Point2f &unified_point = it_per_id.feature_points_[i].unified_point;
                Eigen::Vector3d f = Eigen::Vector3d(unified_point.x, unified_point.y, 1.0).normalized();
                svd_A.row(2 * i) = f[0] * P.row(2) - f[2] * P.row(0);
                svd_A.row(2 * i + 1) = f[1] * P.row(2) - f[2] * P.row(1);
            }
            Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
            double svd_method = svd_V[2] / svd_V[3];

            if (svd_method < 0.1) {
                it_per_id.inv_depth = INIT_DEPTH;
            } else {
                it_per_id.inv_depth = svd_method;
            }
        }
    }

    void FeatureManager::removeOutlier() {
        for (auto it_next = features_.begin(); it_next != features_.end();) {
            auto it = it_next;
            it_next++;
            if (!it->feature_points_.empty() && it->is_outlier_) {
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
            const cv::Point2f &unified_point = it->feature_points_[0].unified_point;
            it->feature_points_.erase(it->feature_points_.begin());
            if (it->feature_points_.size() < 2) {
                features_.erase(it);
                continue;
            }
            if (it->inv_depth < 0)
                it->inv_depth = INIT_DEPTH;
        }
    }

    void FeatureManager::removeFront() {
        for (auto it_next = features_.begin(); it_next != features_.end();) {
            auto it = it_next;
            it_next++;

            if (it->start_frame_ == WINDOW_SIZE) {
                it->start_frame_--;
                continue;
            }
            if (it->endFrame() < WINDOW_SIZE - 1) {
                continue;
            }
            int j = WINDOW_SIZE - 1 - it->start_frame_;
            it->feature_points_.erase(it->feature_points_.begin() + j);
            if (it->feature_points_.empty())
                features_.erase(it);
        }
    }

    double FeatureManager::compensatedParallax2(const FeaturesOfId &it_per_id, int frame_count) {
        //check the second last frame is keyframe or not
        //parallax between second last frame and third last frame
        cv::Point2f p_i = it_per_id.feature_points_[frame_count - 2 - it_per_id.start_frame_].unified_point;
        cv::Point2f p_j = it_per_id.feature_points_[frame_count - 1 - it_per_id.start_frame_].unified_point;
        return cv::norm(p_i - p_j);
    }
}