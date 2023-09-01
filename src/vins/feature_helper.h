#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include <list>
#include <algorithm>
#include <vector>
#include <numeric>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

#include "vins_define_internal.h"
#include "vins_run_info.h"

namespace vins {
    namespace FeatureHelper {
        [[nodiscard]] bool isKeyFrame(int frame_idx,
                                             const std::vector<FeaturePoint2D> &feature_points,
                                             const std::vector<Feature>& feature_window);

        [[nodiscard]] void addFeatures(int frame_idx, double time_stamp,
                                              const std::vector<FeaturePoint2D> &feature_points,
                                              std::vector<Feature>& feature_window);

        [[nodiscard]] Correspondences getCorrespondences(int frame_idx_left, int frame_idx_right,
                                                                const std::vector<Feature>& feature_window);

        [[nodiscard]] std::unordered_map<int, int> getFeatureId2Index(const std::vector<Feature>& feature_window);

        [[nodiscard]] std::vector<Eigen::Vector3d> getPtsVecForFrame(const int frame_idx,
                                                                            const std::vector<Feature>& feature_window);

        double featureIdToDepth(const int feature_id, const std::vector<Feature>& feature_window) {
            auto it = std::find_if(feature_window.begin(), feature_window.end(), [feature_id](const Feature&feature) {
                return feature.feature_id == feature_id;
            });
            if (it == feature_window.end()) {
                assert(false);
                return -1;
            }
            assert(it->inv_depth > 0);
            return 1.0 / it->inv_depth;
        }
    };
}

#endif