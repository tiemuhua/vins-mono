#include "feature_helper.h"
#include "log.h"
#include "vins/parameters.h"

namespace vins {
    using namespace std;
    using namespace Eigen;

    bool FeatureHelper::isKeyFrame(int frame_idx,
                                   const std::vector<FeaturePoint2D> &feature_points,
                                   const std::vector<Feature>& features) {
        LOG_D("input feature: %d, num of feature: %d", (int) feature_points.size(), (int )features.size());

        int last_track_num = 0;
        for (const FeaturePoint2D &point: feature_points) {
            auto it = find_if(features.begin(), features.end(), [point](const Feature &it)->bool {
                return it.feature_id == point.feature_id;
            });

            if (it != features.end()) {
                last_track_num++;
            }
        }

        if (frame_idx < 2 || last_track_num < 20)
            return true;

        int parallax_num = 0;
        double parallax_sum = 0;
        for (const Feature &feature: features) {
            if (feature.start_kf_idx <= frame_idx - 2 && feature.start_kf_idx + int(feature.points.size()) >= frame_idx) {
                cv::Point2f p_i = feature.points[frame_idx - 2 - feature.start_kf_idx];
                cv::Point2f p_j = feature.points[frame_idx - 1 - feature.start_kf_idx];
                parallax_sum += cv::norm(p_i - p_j);
                parallax_num++;
            }
        }

        if (parallax_num == 0) {
            return true;
        }
        LOG_I("parallax_sum: %lf, parallax_num: %d", parallax_sum, parallax_num);
        LOG_I("current parallax: %lf", parallax_sum / parallax_num * Param::Instance().camera.focal);
        return parallax_sum / parallax_num >= Param::Instance().key_frame_parallax_threshold;
    }

    void FeatureHelper::addFeatures(int frame_idx, double time_stamp,
                                    const std::vector<FeaturePoint2D> &feature_points,
                                    std::vector<Feature>& features) {
        for (const FeaturePoint2D &point: feature_points) {
            auto it = find_if(features.begin(), features.end(), [point](const Feature &it)->bool {
                return it.feature_id == point.feature_id;
            });

            if (it == features.end()) {
                features.emplace_back(Feature(point.feature_id, frame_idx));
                it = features.end()--;
            }
            it->points.push_back(point.point);
            it->velocities.push_back(point.velocity);
            it->time_stamps_ms.emplace_back(time_stamp);
        }
    }

    Correspondences FeatureHelper::getCorrespondences(int frame_idx_left, int frame_idx_right,
                                                      const std::vector<Feature>& features) {
        Correspondences correspondences;
        for (const Feature &feature: features) {
            if (feature.start_kf_idx <= frame_idx_left && feature.endFrame() >= frame_idx_right) {
                int idx_l = frame_idx_left - feature.start_kf_idx;
                int idx_r = frame_idx_right - feature.start_kf_idx;
                cv::Point2f a = feature.points[idx_l];
                cv::Point2f b = feature.points[idx_r];
                correspondences.emplace_back(make_pair(a, b));
            }
        }
        return correspondences;
    }

    std::unordered_map<int, int> FeatureHelper::getFeatureId2Index(const std::vector<Feature>& features) {
        std::unordered_map<int, int> id2index;
        for (int i = 0; i < features.size(); ++i) {
            id2index[features[i].feature_id] = i;
        }
        return id2index;
    }

    // 初始帧的坐标？
    std::vector<Eigen::Vector3d> getPtsVecForFrame(const int frame_idx, const std::vector<Feature>& features) {
        for (const Feature& feature: features) {
            if (feature.)
        }
    }
}