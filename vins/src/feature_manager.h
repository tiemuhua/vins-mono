#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include <list>
#include <algorithm>
#include <vector>
#include <numeric>

#include <eigen3/Eigen/Dense>

#include "vins_define_internal.h"

namespace vins {
    class FeatureManager {
    public:
        void clearState();

        bool addFeatureCheckParallax(int frame_id, const std::vector<FeaturePoint> &feature_points);

        [[nodiscard]] vector<pair<cv::Point2f, cv::Point2f>> getCorresponding(int frame_count_l, int frame_count_r) const;

        void removeFailures();

        void clearDepth();
        std::vector<double> getInvDepth() const;
        void setInvDepth(const VectorXd &x);

        void triangulate(const PosWindow& pos_window, const RotWindow& rot_window, const Vector3d& tic, const Matrix3d &ric);

        void removeBackShiftDepth();

        void removeFront();

        void removeOutlier();

        vector<FeaturesOfId> features_;
        int last_track_num{};

    private:
        static double compensatedParallax2(const FeaturesOfId &it_per_id, int frame_count);
    };
}

#endif