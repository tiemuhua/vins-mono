#pragma once

#include <vector>
#include <eigen3/Eigen/Dense>
#include "utility/utility.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <fstream>

const double FOCAL_LENGTH = 460.0;
const int WINDOW_SIZE = 10;
const int NUM_OF_CAM = 1;
const int NUM_OF_F = 1000;
//#define UNIT_SPHERE_ERROR

typedef Eigen::Vector3d PosWindow[WINDOW_SIZE + 1];
typedef Eigen::Vector3d VecWindow[WINDOW_SIZE + 1];
typedef Eigen::Matrix3d RotWindow[WINDOW_SIZE + 1];
typedef Eigen::Vector3d BaWindow[WINDOW_SIZE + 1];
typedef Eigen::Vector3d BgWindow[WINDOW_SIZE + 1];
class PreIntegration;
typedef PreIntegration* PreIntegrateWindow[WINDOW_SIZE + 1];
typedef double TimeStampWindow[WINDOW_SIZE + 1];
typedef std::vector<double> DtBufWindow[WINDOW_SIZE + 1];
typedef std::vector<Eigen::Vector3d> AccBufWindow[WINDOW_SIZE + 1];
typedef std::vector<Eigen::Vector3d> GyrBufWindow[WINDOW_SIZE + 1];

extern double INIT_DEPTH;
extern double MIN_PARALLAX;
enum EstimateExtrinsicState{
    EstimateExtrinsicFix,
    EstimateExtrinsicInitiating,
    EstimateExtrinsicInitiated,
};
extern EstimateExtrinsicState estimate_extrinsic_state;

extern double ACC_N, ACC_W;
extern double GYR_N, GYR_W;

extern Eigen::Matrix3d RIC;
extern Eigen::Vector3d TIC;
extern Eigen::Vector3d G;

extern double BIAS_ACC_THRESHOLD;
extern double BIAS_GYR_THRESHOLD;
extern double SOLVER_TIME;
extern int NUM_ITERATIONS;
extern std::string EX_CALIB_RESULT_PATH;
extern std::string VINS_RESULT_PATH;
extern std::string IMU_TOPIC;
extern double TD;
extern double TR;
extern int ESTIMATE_TD;
extern int ROLLING_SHUTTER;
extern double ROW, COL;


void readParameters();

enum SIZE_PARAMETERIZATION {
    SIZE_POSE = 7,
    SIZE_SPEED_BIAS = 9,
    SIZE_FEATURE = 1
};

typedef double Pose[SIZE_POSE];
typedef double SpeedBias[SIZE_SPEED_BIAS];
typedef double Feature[SIZE_FEATURE];
typedef double Td[1];

enum StateOrder {
    O_P = 0,
    O_R = 3,
    O_V = 6,
    O_BA = 9,
    O_BG = 12
};

enum NoiseOrder {
    O_AN = 0,
    O_GN = 3,
    O_AW = 6,
    O_GW = 9
};
