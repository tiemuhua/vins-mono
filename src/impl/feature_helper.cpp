#include "feature_helper.h"
#include <glog/logging.h>
#include "param.h"

namespace vins {
    using namespace std;
    using namespace Eigen;

    bool FeatureHelper::isKeyFrame(const int key_frame_idx,
                                   const double focal,
                                   const double kf_parallax_threshold,
                                   const std::vector<FeaturePoint2D> &feature_points,
                                   const std::vector<Feature> &feature_window) {
        LOG(INFO) << "input feature: " << feature_points.size()
                  << "\tnum of feature: " << feature_window.size();

        int last_track_num = 0;
        for (const FeaturePoint2D &point: feature_points) {
            auto it = find_if(feature_window.begin(), feature_window.end(), [point](const Feature &it) -> bool {
                return it.feature_id == point.feature_id;
            });

            if (it != feature_window.end()) {
                last_track_num++;
            }
        }

        if (key_frame_idx < 2 || last_track_num < 20)
            return true;

        int parallax_num = 0;
        double parallax_sum = 0;
        for (const Feature &feature: feature_window) {
            if (feature.start_kf_window_idx <= key_frame_idx - 2 &&
                feature.start_kf_window_idx + int(feature.points.size()) >= key_frame_idx) {
                cv::Point2f p_i = feature.points[key_frame_idx - 2 - feature.start_kf_window_idx];
                cv::Point2f p_j = feature.points[key_frame_idx - 1 - feature.start_kf_window_idx];
                parallax_sum += cv::norm(p_i - p_j);
                parallax_num++;
            }
        }

        if (parallax_num == 0) {
            return true;
        }
        LOG(INFO) << "parallax_sum:" << parallax_sum
                  << "\tparallax_num:" << parallax_num
                  << "\tcurrent parallax:" << parallax_sum / parallax_num * focal;
        return parallax_sum / parallax_num >= kf_parallax_threshold;
    }

    void FeatureHelper::addFeatures(int frame_idx, double time_stamp,
                                    const std::vector<FeaturePoint2D> &feature_points,
                                    std::vector<Feature> &feature_window) {
        for (const FeaturePoint2D &point: feature_points) {
            auto it = find_if(feature_window.begin(), feature_window.end(), [point](const Feature &it) -> bool {
                return it.feature_id == point.feature_id;
            });

            if (it == feature_window.end()) {
                feature_window.emplace_back(Feature(point.feature_id, frame_idx));
                it = feature_window.end()--;
            }
            it->points.push_back(point.point);
            it->velocities.push_back(point.velocity);
            it->time_stamps_ms.emplace_back(time_stamp);
        }
    }

    Correspondences FeatureHelper::getCorrespondences(int frame_idx_left, int frame_idx_right,
                                                      const std::vector<Feature> &feature_window) {
        Correspondences correspondences;
        for (const Feature &feature: feature_window) {
            if (feature.start_kf_window_idx <= frame_idx_left && feature.endFrame() >= frame_idx_right) {
                int idx_l = frame_idx_left - feature.start_kf_window_idx;
                int idx_r = frame_idx_right - feature.start_kf_window_idx;
                cv::Point2f a = feature.points[idx_l];
                cv::Point2f b = feature.points[idx_r];
                correspondences.emplace_back(make_pair(a, b));
            }
        }
        return correspondences;
    }

    std::unordered_map<int, int> FeatureHelper::getFeatureId2Index(const std::vector<Feature> &feature_window) {
        std::unordered_map<int, int> id2index;
        for (int i = 0; i < feature_window.size(); ++i) {
            id2index[feature_window[i].feature_id] = i;
        }
        return id2index;
    }

    double featureIdToDepth(const int feature_id, const std::vector<Feature> &feature_window) {
        auto it = std::find_if(feature_window.begin(), feature_window.end(), [feature_id](const Feature &feature) {
            return feature.feature_id == feature_id;
        });
        if (it == feature_window.end()) {
            assert(false);
            return -1;
        }
        assert(it->inv_depth > 0);
        return 1.0 / it->inv_depth;
    }
}