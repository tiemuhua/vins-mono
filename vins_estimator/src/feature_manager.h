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

class FeaturePerFrame {
public:
    FeaturePerFrame(const Eigen::Matrix<double, 7, 1> &_point, double td) {
        point_ = _point.block<3,1>(0, 0);
        uv = _point.block<2,1>(3,0);
        velocity = _point.block<2,1>(5,0);
        cur_td = td;
    }

    double cur_td;
    Vector3d point_;
    Vector2d uv;
    Vector2d velocity;
};

class FeaturePerId {
public:
    const int feature_id_;
    int start_frame_;
    vector<FeaturePerFrame> feature_per_frame_;

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
    explicit FeatureManager(Matrix3d _Rs[]);

    void setRic(Matrix3d _ric[]);

    void clearState();

    int getFeatureCount();

    bool addFeatureCheckParallax(int frame_count, const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image,
                                 double td);

    vector<pair<Vector3d, Vector3d>> getCorresponding(int frame_count_l, int frame_count_r);

    void setDepth(const VectorXd &x);

    void removeFailures();

    void clearDepth(const VectorXd &x);

    VectorXd getDepthVector();

    void triangulate(PosWindow pos_window, Vector3d tic[], Matrix3d ric[]);

    void removeBackShiftDepth();

    void removeBack();

    void removeFront(int frame_count);

    void removeOutlier();

    list<FeaturePerId> features_;
    int last_track_num{};

private:
    static double compensatedParallax2(const FeaturePerId &it_per_id, int frame_count);

    const Matrix3d *Rs;
    Matrix3d ric[NUM_OF_CAM];
};

#endif