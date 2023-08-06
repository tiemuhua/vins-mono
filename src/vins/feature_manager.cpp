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

    bool FeatureManager::isKeyFrame(int frame_id, const std::vector<FeaturePoint2D> &feature_points) const {
        LOG_D("input feature: %d, num of feature: %d", (int) feature_points.size(), (int )features_.size());

        int last_track_num = 0;
        for (const FeaturePoint2D &point: feature_points) {
            auto it = find_if(features_.begin(), features_.end(), [point](const SameFeatureInDifferentFrames &it)->bool {
                return it.feature_id == point.feature_id;
            });

            if (it != features_.end()) {
                last_track_num++;
            }
        }

        if (frame_id < 2 || last_track_num < 20)
            return true;

        int parallax_num = 0;
        double parallax_sum = 0;
        for (const SameFeatureInDifferentFrames &feature_per_id: features_) {
            if (feature_per_id.start_frame <= frame_id - 2 &&
                feature_per_id.start_frame + int(feature_per_id.points.size()) >= frame_id) {
                parallax_sum += compensatedParallax2(feature_per_id, frame_id);
                parallax_num++;
            }
        }

        if (parallax_num == 0) {
            return true;
        }
        LOG_I("parallax_sum: %lf, parallax_num: %d", parallax_sum, parallax_num);
        LOG_I("current parallax: %lf", parallax_sum / parallax_num * FOCAL_LENGTH);
        return parallax_sum / parallax_num >= MIN_PARALLAX;
    }

    void FeatureManager::addFeatures(int frame_id, double time_stamp,
                                     const std::vector<FeaturePoint2D> &feature_points) {
        for (const FeaturePoint2D &point: feature_points) {
            auto it = find_if(features_.begin(), features_.end(), [point](const SameFeatureInDifferentFrames &it)->bool {
                return it.feature_id == point.feature_id;
            });

            if (it == features_.end()) {
                features_.emplace_back(SameFeatureInDifferentFrames(point.feature_id, frame_id));
                it = features_.end()--;
            }
            it->points.push_back(point.point);
            it->velocities.push_back(point.velocity);
            it->time_stamps_ms.emplace_back(time_stamp);
        }
    }

    Correspondences FeatureManager::getCorrespondences(int frame_count_l, int frame_count_r) const {
        Correspondences correspondences;
        for (const SameFeatureInDifferentFrames &it: features_) {
            if (it.start_frame <= frame_count_l && it.endFrame() >= frame_count_r) {
                int idx_l = frame_count_l - it.start_frame;
                int idx_r = frame_count_r - it.start_frame;
                cv::Point2f a = it.points[idx_l];
                cv::Point2f b = it.points[idx_r];
                correspondences.emplace_back(make_pair(a, b));
            }
        }
        return correspondences;
    }

    std::unordered_map<int, int> FeatureManager::getFeatureId2Index() {
        std::unordered_map<int, int> id2index;
        for (int i = 0; i < features_.size(); ++i) {
            id2index[features_[i].feature_id] = i;
        }
        return id2index;
    }

    void FeatureManager::discardFeaturesOfFrameId(int frame_id) {
        std::vector<SameFeatureInDifferentFrames> features;
        for (auto &feature:features_) {
            if (feature.start_frame != frame_id) {
                features.emplace_back(std::move(feature));
            }
        }
        features_ = std::move(features);
    }

    void FeatureManager::removeFailures() {
        for (auto it = features_.begin(), it_next = features_.begin();
             next(it) != features_.end(); it = it_next) {
            it_next++;
            if (it->solve_flag_ == SameFeatureInDifferentFrames::kDepthSolvedFail)
                features_.erase(it);
        }
    }

    void FeatureManager::triangulate(const Window<Eigen::Vector3d>& pos_window,
                                     const Window<Eigen::Matrix3d>& rot_window,
                                     const Vector3d& tic, const Matrix3d &ric) {
        for (SameFeatureInDifferentFrames &it_per_id: features_) {
            it_per_id.inv_depth = -1.0;
        }

        for (SameFeatureInDifferentFrames &it_per_id: features_) {
            if (!(it_per_id.points.size() >= 2 && it_per_id.start_frame < WINDOW_SIZE - 2))
                continue;

            if (it_per_id.inv_depth > 0)
                continue;

            Eigen::MatrixXd svd_A(2 * it_per_id.points.size(), 4);

            int imu_i = it_per_id.start_frame;
            Eigen::Vector3d t0 = pos_window.at(imu_i) + rot_window.at(imu_i) * tic;
            Eigen::Matrix3d R0 = rot_window.at(imu_i) * ric;

            for (int i = 0; i < it_per_id.points.size(); ++i) {
                int imu_j = it_per_id.start_frame + i;
                Eigen::Vector3d t1 = pos_window.at(imu_j) + rot_window.at(imu_j) * tic;
                Eigen::Matrix3d R1 = rot_window.at(imu_j) * ric;
                Eigen::Vector3d t = R0.transpose() * (t1 - t0);
                Eigen::Matrix3d R = R0.transpose() * R1;
                Eigen::Matrix<double, 3, 4> P;
                P.leftCols<3>() = R.transpose();
                P.rightCols<1>() = -R.transpose() * t;
                const cv::Point2f &unified_point = it_per_id.points[i];
                Eigen::Vector3d f = Eigen::Vector3d(unified_point.x, unified_point.y, 1.0).normalized();
                svd_A.row(2 * i) = f[0] * P.row(2) - f[2] * P.row(0);
                svd_A.row(2 * i + 1) = f[1] * P.row(2) - f[2] * P.row(1);
            }
            Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
            double svd_method = svd_V[2] / svd_V[3];

            if (svd_method < 0.1) {
                it_per_id.inv_depth = -1;
            } else {
                it_per_id.inv_depth = svd_method;
            }
        }
    }

    void FeatureManager::removeOutlier() {
        for (auto it_next = features_.begin(); it_next != features_.end();) {
            auto it = it_next;
            it_next++;
            if (!it->points.empty() && it->is_outlier) {
                features_.erase(it);
            }
        }
    }

    void FeatureManager::removeBackShiftDepth() {
        for (auto it_next = features_.begin(); it_next!=features_.end();) {
            auto it = it_next;
            it_next++;
            if (it->start_frame != 0) {
                it->start_frame--;
                continue;
            }
            const cv::Point2f &unified_point = it->points[0];
            it->points.erase(it->points.begin());
            it->velocities.erase(it->velocities.begin());
            it->time_stamps_ms.erase(it->time_stamps_ms.begin());
            if (it->points.size() < 2) {
                features_.erase(it);
                continue;
            }
            if (it->inv_depth < 0)
                it->inv_depth = -1;
        }
    }

    void FeatureManager::removeFront() {
        for (auto it_next = features_.begin(); it_next != features_.end();) {
            auto it = it_next;
            it_next++;

            if (it->start_frame == WINDOW_SIZE) {
                it->start_frame--;
                continue;
            }
            if (it->endFrame() < WINDOW_SIZE - 1) {
                continue;
            }
            int j = WINDOW_SIZE - 1 - it->start_frame;
            it->points.erase(it->points.begin());
            it->velocities.erase(it->velocities.begin());
            it->time_stamps_ms.erase(it->time_stamps_ms.begin());
            if (it->points.empty())
                features_.erase(it);
        }
    }

    double FeatureManager::compensatedParallax2(const SameFeatureInDifferentFrames &feature, int frame_id) {
        // check the second last frame is keyframe or not
        // parallax between second last frame and third last frame
        cv::Point2f p_i = feature.points[frame_id - 2 - feature.start_frame];
        cv::Point2f p_j = feature.points[frame_id - 1 - feature.start_frame];
        return cv::norm(p_i - p_j);
    }
}