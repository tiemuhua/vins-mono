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
#include "factor/pose_local_parameterization.h"
#include "factor/projection_factor.h"
#include "factor/projection_td_factor.h"
#include "factor/marginalization_factor.h"

#include <unordered_map>
#include <queue>
#include <opencv2/core/eigen.hpp>


class Estimator {
public:
    Estimator();

    void setParameter();

    void processIMU(double t, const Vector3d &linear_acceleration, const Vector3d &angular_velocity);

    void
    processImage(const map<int, vector<pair<int, Eigen::Matrix<double, 7, 1>>>> &image, const double &time_stamp);

    void setReloFrame(double _frame_stamp, int _frame_index, vector<Vector3d> &_match_points,
                      const Vector3d &_relo_t, const Matrix3d &_relo_r);

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

    Matrix3d ric[NUM_OF_CAM];
    Vector3d tic[NUM_OF_CAM];

    Vector3d Ps[(WINDOW_SIZE + 1)];
    Vector3d Vs[(WINDOW_SIZE + 1)];
    Matrix3d Rs[(WINDOW_SIZE + 1)];
    Vector3d Bas[(WINDOW_SIZE + 1)];
    Vector3d Bgs[(WINDOW_SIZE + 1)];
    double td{};

    Matrix3d back_R0, last_R, last_R0;
    Vector3d back_P0, last_P, last_P0;
    double time_stamps[WINDOW_SIZE + 1]{};

    IntegrationBase *pre_integrations[(WINDOW_SIZE + 1)]{};
    Vector3d acc_0, gyr_0;

    vector<double> dt_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> linear_acceleration_buf[(WINDOW_SIZE + 1)];
    vector<Vector3d> angular_velocity_buf[(WINDOW_SIZE + 1)];

    int frame_count{};
    int sum_of_back{}, sum_of_front{};

    FeatureManager f_manager;
    MotionEstimator m_estimator;
    InitialEXRotation initial_ex_rotation;

    bool first_imu{};
    bool failure_occur{};

    vector<Vector3d> key_poses;
    double initial_timestamp{};

    Pose para_Pose[WINDOW_SIZE + 1]{};
    SpeedBias para_SpeedBias[WINDOW_SIZE + 1]{};
    Feature para_Feature[NUM_OF_F]{};
    Pose para_Ex_Pose[NUM_OF_CAM]{};
    double para_Td[1][1]{};

    MarginalizationInfo *last_marginalization_info{};
    vector<double *> last_marginal_param_blocks;

    map<double, ImageFrame> all_image_frame;
    IntegrationBase *tmp_pre_integration{};

    //relocalization variable
    bool relocalization_info{};
    double relo_frame_stamp{};
    int relo_frame_local_index{};
    vector<Vector3d> match_points;
    double relo_Pose[SIZE_POSE]{};
    Matrix3d drift_correct_r;
    Vector3d prev_relo_t;
    Matrix3d prev_relo_r;
};
