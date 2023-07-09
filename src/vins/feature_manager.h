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

        bool addFeaturesAndCheckIsKeyFrame(int frame_id, const std::vector<FeaturePoint2D> &feature_points);

        [[nodiscard]] std::vector<std::pair<cv::Point2f, cv::Point2f>> getCorresponding(int frame_count_l, int frame_count_r) const;

        void triangulate(const Window<Eigen::Vector3d>& pos_window,
                         const Window<Eigen::Matrix3d>& rot_window,
                         ConstVec3dRef tic,
                         ConstMat3dRef ric);

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