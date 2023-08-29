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
    class FeatureHelper {
    public:
        [[nodiscard]] static bool isKeyFrame(int frame_idx,
                                             const std::vector<FeaturePoint2D> &feature_points,
                                             const std::vector<Feature>& feature_window);

        [[nodiscard]] static void addFeatures(int frame_idx, double time_stamp,
                                              const std::vector<FeaturePoint2D> &feature_points,
                                              std::vector<Feature>& feature_window);

        [[nodiscard]] static Correspondences getCorrespondences(int frame_idx_left, int frame_idx_right,
                                                                const std::vector<Feature>& feature_window);

        [[nodiscard]] static std::unordered_map<int, int> getFeatureId2Index(const std::vector<Feature>& feature_window);

        [[nodiscard]] static std::vector<Eigen::Vector3d> getPtsVecForFrame(const int frame_idx,
                                                                            const std::vector<Feature>& feature_window);
    };
}

#endif