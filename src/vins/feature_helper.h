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
        [[nodiscard]] static bool isKeyFrame(int frame_id,
                                             const std::vector<FeaturePoint2D> &feature_points,
                                             const std::vector<Feature>& features);

        [[nodiscard]] static void addFeatures(int frame_id, double time_stamp,
                                              const std::vector<FeaturePoint2D> &feature_points,
                                              std::vector<Feature>& features);

        [[nodiscard]] static Correspondences getCorrespondences(int frame_count_l, int frame_count_r,
                                                                const std::vector<Feature>& features);

        [[nodiscard]] static std::unordered_map<int, int> getFeatureId2Index(const std::vector<Feature>& features);
    };
}

#endif