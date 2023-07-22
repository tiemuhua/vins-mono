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
    class FeatureManager {
    public:
        void clearState();

        [[nodiscard]] bool isKeyFrame(int frame_id, const std::vector<FeaturePoint2D> &feature_points) const;
        void addFeatures(int frame_id, const std::vector<FeaturePoint2D> &feature_points);

        [[nodiscard]] Correspondences getCorrespondences(int frame_count_l, int frame_count_r) const;

        void triangulate(const Window<Eigen::Vector3d>& pos_window,
                         const Window<Eigen::Matrix3d>& rot_window,
                         ConstVec3dRef tic,
                         ConstMat3dRef ric);

        std::unordered_map<int, int> getFeatureId2Index();

        void removeFailures();
        void removeBackShiftDepth();
        void removeFront();
        void removeOutlier();

        std::vector<FeaturesOfId> features_;

    private:
        static double compensatedParallax2(const FeaturesOfId &it_per_id, int frame_count);
    };
}

#endif