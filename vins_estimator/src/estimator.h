#pragma once

#include "parameters.h"
#include "feature_manager.h"
#include "utility/utility.h"
#include "utility/tic_toc.h"
#include "initial/solve_5pts.h"
#include "initial/initial_sfm.h"
#include "initial/initial_alignment.h"
#include "initial/initial_ex_rotation.h"

#include <ceres/ceres.h>
#include "factor/imu_factor.h"
#include "factor/projection_factor.h"
#include "factor/projection_td_factor.h"
#include "factor/marginalization_factor.h"

#include <unordered_map>
#include <queue>
#include <opencv2/core/eigen.hpp>


class Estimator {
public:
    struct MatchPoint{
        cv::Point2f point;
        int feature_id;
    };

public:
    Estimator();

    void setParameter();

    void processIMU(double t, const Vector3d &acc, const Vector3d &gyr);

    void processImage(const FeatureTracker::FeaturesPerImage &image, const double &time_stamp);

    void setReLocalFrame(double _frame_stamp, int _frame_index, vector<MatchPoint> &_match_points,
                         const Vector3d &_re_local_t, const Matrix3d &_re_local_r);

private:
    void clearState();

    bool initialStructure();

    bool visualInitialAlign();

    bool relativePose(Matrix3d &relative_R, Vector3d &relative_T, int &l);

    void slideWindow();

    void solveOdometry();

    void slideWindowNew();

    void slideWindowOld();

    void optimization();

    void vector2double();

    void double2vector();

    bool failureDetection();


    enum SolverFlag {
        INITIAL,
        NON_LINEAR
    };

    enum MarginalizationFlag {
        MARGIN_OLD = 0,
        MARGIN_SECOND_NEW = 1
    };

    SolverFlag solver_flag;
    MarginalizationFlag marginalization_flag;
    Vector3d g;

    PosWindow pos_window;
    VecWindow vec_window;
    RotWindow rot_window;
    BaWindow ba_window;
    BgWindow bg_window;
    TimeStampWindow time_stamp_window{};
    PreIntegrateWindow pre_integrate_window{};
    // todo tiemuhuaguo pre_integrate里面已经有IntegrationBase了，为什么还要保留DtBufWindow、AccBufWindow、GyrBufWindow
    DtBufWindow dt_buf_window;
    AccBufWindow acc_buf_window;
    GyrBufWindow gyr_buf_window;

    double td{};

    Matrix3d last_R, last_R0;
    Vector3d last_P, last_P0;

    Vector3d acc_0, gyr_0;

    int frame_count{};
    int sum_of_back{}, sum_of_front{};

    FeatureManager feature_manager;
    InitialEXRotation initial_ex_rotation;

    bool first_imu{};
    bool failure_occur{};

    vector<Vector3d> key_poses;
    double initial_timestamp{};

    Pose para_Pose[WINDOW_SIZE + 1]{};
    SpeedBias para_SpeedBias[WINDOW_SIZE + 1]{};
    Feature para_Feature[NUM_OF_F]{};
    Pose para_Ex_Pose[1]{};
    double para_Td[1][1]{};

    MarginalizationInfo *last_marginalization_info_{};
    vector<double *> last_marginal_param_blocks_;

    vector<ImageFrame> all_image_frame;
    PreIntegration *tmp_pre_integration{};

    //re-localization variable
    bool is_re_localization_{};
    double re_local_frame_stamp{};
    int re_local_frame_local_index{};
    vector<MatchPoint> match_points;
    double re_local_Pose[SIZE_POSE]{};
};
