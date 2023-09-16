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
    typedef std::vector<std::pair<cv::Point2f, cv::Point2f>> Correspondences;
}
namespace vins::FeatureHelper {
    [[nodiscard]] bool isKeyFrame(int key_frame_idx,
                                  double focal,
                                  double kf_parallax_threshold,
                                  const std::vector<FeaturePoint2D> &feature_points,
                                  const std::vector<Feature> &feature_window);

    void addFeatures(int frame_idx, double time_stamp,
                     const std::vector<FeaturePoint2D> &feature_points,
                     std::vector<Feature> &feature_window);

    [[nodiscard]] Correspondences getCorrespondences(int frame_idx_left, int frame_idx_right,
                                                     const std::vector<Feature> &feature_window);

    [[nodiscard]] std::unordered_map<int, int> getFeatureId2Index(const std::vector<Feature> &feature_window);

    [[nodiscard]] std::vector<Eigen::Vector3d> getPtsVecForFrame(int frame_idx,
                                                                 const std::vector<Feature> &feature_window);

    [[nodiscard]]double featureIdToDepth(int feature_id, const std::vector<Feature> &feature_window);
}

#endif