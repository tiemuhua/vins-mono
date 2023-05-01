#ifndef FEATURE_MANAGER_H
#define FEATURE_MANAGER_H

#include <list>
#include <algorithm>
#include <vector>
#include <numeric>

using namespace std;

#include <eigen3/Eigen/Dense>

using namespace Eigen;

#include "parameters.h"
#include "feature_tracker/src/feature_tracker.h"

struct FeaturePoint {
    cv::Point2f point;
    cv::Point2f unified_point;
    cv::Point2f point_velocity;
    int feature_id;
    double cur_td;
};

class FeaturePerId {
public:
    const int feature_id_;
    int start_frame_;
    vector<FeaturePoint> feature_per_frames_;

    bool is_outlier_{};
    double estimated_depth;
    enum {
        FeatureHaveNotSolved,
        FeatureSolvedSucc,
        FeatureSolveFail,
    }solve_flag_;

    FeaturePerId(int _feature_id, int _start_frame)
            : feature_id_(_feature_id), start_frame_(_start_frame),
              estimated_depth(-1.0), solve_flag_(FeatureHaveNotSolved) {
    }

    [[nodiscard]] int endFrame() const;
};

class FeatureManager {
public:
    void clearState();

    int getFeatureCount();

    bool addFeatureCheckParallax(int frame_id, const std::vector<FeaturePoint> &feature_points, double td);

    vector<pair<cv::Point2f, cv::Point2f>> getCorresponding(int frame_count_l, int frame_count_r);

    void setDepth(const VectorXd &x);

    void removeFailures();

    void clearDepth(const VectorXd &x);

    VectorXd getDepthVector();

    void triangulate(const PosWindow pos_window, const RotWindow rot_window, const Vector3d& tic, const Matrix3d &ric);

    void removeBackShiftDepth();

    void removeBack();

    void removeFront(int frame_count);

    void removeOutlier();

    list<FeaturePerId> features_;
    int last_track_num{};

private:
    static double compensatedParallax2(const FeaturePerId &it_per_id, int frame_count);
};

#endif