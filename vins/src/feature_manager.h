#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include <list>
#include <algorithm>
#include <vector>
#include <numeric>

using namespace std;

#include <eigen3/Eigen/Dense>

using namespace Eigen;

#include "feature_tracker/src/feature_tracker.h"

#include "vins_define_internal.h"

namespace vins {
    class FeatureManager {
    public:
        void clearState();

        int getFeatureCount();

        bool addFeatureCheckParallax(int frame_id, const std::vector<FeaturePoint> &feature_points, double td);

        vector<pair<cv::Point2f, cv::Point2f>> getCorresponding(int frame_count_l, int frame_count_r);

        // todo tiemuhua 换个靠谱点的函数名
        bool relativePose(const int last_key_frame_id, Matrix3d &relative_R, Vector3d &relative_T, int &l);

        void setInvDepth(const VectorXd &x);

        void removeFailures();

        void clearDepth();

        VectorXd getInvDepthVector();

        void triangulate(const PosWindow& pos_window, const RotWindow& rot_window, const Vector3d& tic, const Matrix3d &ric);

        void removeBackShiftDepth();

        void removeFront();

        void removeOutlier();

        list<FeaturesOfId> features_;
        int last_track_num{};

    private:
        static double compensatedParallax2(const FeaturesOfId &it_per_id, int frame_count);
    };
}

#endif