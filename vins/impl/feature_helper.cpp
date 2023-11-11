#include "feature_helper.h"
#include <glog/logging.h>

namespace vins {
    using namespace std;
    using namespace Eigen;

    bool FeatureHelper::isKeyFrame(const int key_frame_idx,
                                   const double kf_parallax_threshold,
                                   const std::vector<FeaturePoint2D> &feature_points,
                                   const std::vector<Feature> &feature_window) {
        if (key_frame_idx == 0) {
            return true;
        }
        auto new_track_num = count_if(feature_points.begin(),
                                     feature_points.end(),
                                     [&](const FeaturePoint2D &it) -> bool {
            return it.feature_id > feature_window.back().feature_id;
        });

        if (key_frame_idx < 2 || new_track_num < 20)
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
        LOG(INFO) << "parallax_sum:" << parallax_sum << "\t"
                << "parallax_num:" << parallax_num << "\t";
        return parallax_sum / parallax_num >= kf_parallax_threshold;
    }

    void FeatureHelper::addFeatures(int frame_idx, double time_stamp,
                                    const std::vector<FeaturePoint2D> &feature_pts,
                                    std::vector<Feature> &feature_window) {
        std::unordered_map<int,int> feature_id_2_feature_idx = FeatureHelper::getFeatureId2Index(feature_window);
        for (const FeaturePoint2D &point: feature_pts) {
            if (!feature_id_2_feature_idx.count(point.feature_id)) {
                feature_window.emplace_back(Feature(point.feature_id, frame_idx));
                feature_id_2_feature_idx[point.feature_id] = feature_window.size() - 1;
            }
            int idx = feature_id_2_feature_idx[point.feature_id];
            feature_window[idx].points.push_back(point.point);
            feature_window[idx].velocities.push_back(point.velocity);
            feature_window[idx].time_stamps_ms.emplace_back(time_stamp);
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

    double FeatureHelper::featureIdToDepth(const int feature_id, const std::vector<Feature> &feature_window) {
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
