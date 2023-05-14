#pragma once

#include <vector>
#include <eigen3/Eigen/Dense>
#include "utility/utility.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <fstream>

const double FOCAL_LENGTH = 460.0;
constexpr int WINDOW_SIZE = 10;
const int FEATURE_SIZE = 1000;
#define UNIT_SPHERE_ERROR

class PreIntegration;
typedef std::array<PreIntegration*, WINDOW_SIZE + 1> PreIntegrateWindow;
typedef std::array<double, WINDOW_SIZE + 1> TimeStampWindow;
typedef std::array<Eigen::Vector3d, WINDOW_SIZE + 1> BaWindow;
typedef std::array<Eigen::Vector3d, WINDOW_SIZE + 1> BgWindow;
typedef std::array<Eigen::Vector3d, WINDOW_SIZE + 1> PosWindow;
typedef std::array<Eigen::Vector3d, WINDOW_SIZE + 1> VelWindow;
typedef std::array<Eigen::Matrix3d, WINDOW_SIZE + 1> RotWindow;
typedef std::array<Eigen::Matrix3d, WINDOW_SIZE + 1> RotWindow;

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

typedef double Pose[7];
typedef double AccBias[3];
typedef double GyrBias[3];
typedef double Velocity[3];
typedef double FeatureDepth[1];
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
