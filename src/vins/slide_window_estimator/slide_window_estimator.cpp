//
// Created by gjt on 5/14/23.
//

#include "slide_window_estimator.h"

#include <ceres/ceres.h>

#include "log.h"
#include "vins/vins_utils.h"

#include "cost_functions/imu_cost.h"
#include "cost_functions/marginal_cost.h"
#include "cost_functions/project_cost.h"
#include "cost_functions/project_td_cost.h"

constexpr int WINDOW_SIZE = 10;
constexpr int FeaturePointSize = 100;

typedef double arr1d[1];
typedef double arr3d[3];
typedef double arr4d[4];

static arr3d c_pos[WINDOW_SIZE];
static arr3d c_vel[WINDOW_SIZE];
static arr3d c_ba[WINDOW_SIZE];
static arr3d c_bg[WINDOW_SIZE];
static arr4d c_quat[WINDOW_SIZE];

static arr3d c_tic;
static arr4d c_ric;

static arr1d c_inv_depth[FeaturePointSize];
static arr1d c_time_delay;

static arr3d c_loop_peer_pos;
static arr4d c_loop_peer_quat;

static std::vector<double*> s_marginal_param_blocks;
static vins::MarginalInfo *sp_marginal_info;

using namespace vins;

static void eigen2c(const BundleAdjustWindow& window,
                    const std::vector<FeaturesOfId> &features_,
                    const Eigen::Vector3d& tic,
                    const Eigen::Matrix3d &ric){
    for (int i = 0; i < window.pos_window.size(); ++i) {
        utils::vec3d2array(window.pos_window.at(i), c_pos[i]);
    }
    for (int i = 0; i < window.rot_window.size(); ++i) {
        utils::quat2array(Eigen::Quaterniond(window.rot_window.at(i)), c_quat[i]);
    }
    for (int i = 0; i < window.vel_window.size(); ++i) {
        utils::vec3d2array(window.vel_window.at(i), c_vel[i]);
    }
    for (int i = 0; i < window.ba_window.size(); ++i) {
        utils::vec3d2array(window.ba_window.at(i), c_ba[i]);
    }
    for (int i = 0; i < window.bg_window.size(); ++i) {
        utils::vec3d2array(window.bg_window.at(i), c_bg[i]);
    }

    for (int i = 0; i < features_.size(); ++i) {
        if (features_[i].feature_points_.size() < 2 || features_[i].start_frame_ >= WINDOW_SIZE - 2) {
            continue;
        }
        c_inv_depth[i][0] = features_[i].inv_depth;
    }

    utils::vec3d2array(tic, c_tic);
    utils::quat2array(Eigen::Quaterniond(ric), c_ric);
}

static void c2eigen(BundleAdjustWindow &window,
                    std::vector<FeaturesOfId> &features_,
                    Eigen::Vector3d& tic,
                    Eigen::Matrix3d &ric) {

}

void SlideWindowEstimator::optimize(const SlideWindowEstimatorParam &param,
                                    std::vector<FeaturesOfId> &features_,
                                    BundleAdjustWindow &window,
                                    Eigen::Vector3d &tic,
                                    Eigen::Matrix3d &ric) {
    eigen2c(window, features_, tic, ric);

    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::CauchyLoss(1.0);
    for (int i = 0; i < WINDOW_SIZE + 1; i++) {
        problem.AddParameterBlock(c_pos[i], 3);
        problem.AddParameterBlock(c_quat[i], 4);
        problem.AddParameterBlock(c_vel[i], 3);
        problem.AddParameterBlock(c_ba[i], 3);
        problem.AddParameterBlock(c_bg[i], 3);
    }
    problem.AddParameterBlock(c_ric, 4);
    problem.AddParameterBlock(c_tic, 3);
    if (param.fix_extrinsic) {
        LOG_D("fix extrinsic param");
        problem.SetParameterBlockConstant(c_ric);
        problem.SetParameterBlockConstant(c_tic);
    } else {
        LOG_D("estimate extrinsic param");
    }
    if (param.estimate_time_delay) {
        problem.AddParameterBlock(c_time_delay, 1);
    }

    /*************** 1:边缘化 **************************/
    if (sp_marginal_info) {
        // todo tiemuhuaguo last_marginal_param_blocks和sp_marginal_info是怎么维护的
        auto *cost_function = new MarginalFactor(sp_marginal_info);
        problem.AddResidualBlock(cost_function, nullptr, s_marginal_param_blocks);
    }

    /*************** 2:IMU **************************/
    for (int i = 0; i < window.pre_int_window.size(); i++) {
        int j = i + 1;
        if (window.pre_int_window.at(i).deltaTime() > 10.0)// todo why???
            continue;
        auto *cost_function = new IMUFactor(window.pre_int_window.at(i));
        problem.AddResidualBlock(cost_function, nullptr,
                                 c_pos[i], c_quat[i], c_vel[i], c_ba[i], c_bg[i],
                                 c_pos[j], c_quat[j], c_vel[j], c_ba[j], c_bg[j]);
    }

    /*************** 3:特征点 **************************/
    for (int feature_id = 0; feature_id < features_.size(); ++feature_id) {
        const FeaturesOfId &features_of_id = features_[feature_id];
        if (features_of_id.feature_points_.size() < 2 || features_of_id.start_frame_ >= WINDOW_SIZE - 2)
            continue;
        int start_frame_id = features_of_id.start_frame_;
        const FeaturePoint2D & point0 = features_of_id.feature_points_[0];
        for (int frame_bias = 0; frame_bias < features_of_id.feature_points_.size(); ++frame_bias) {
            const FeaturePoint2D &point = features_of_id.feature_points_[frame_bias];
            int cur_frame_id = start_frame_id + frame_bias;
            if (param.estimate_time_delay) {
                auto *cost_function = new ProjectionTdFactor(point0, point);
                problem.AddResidualBlock(cost_function, loss_function,
                                         c_pos[start_frame_id], c_quat[start_frame_id],
                                         c_pos[cur_frame_id], c_quat[cur_frame_id],
                                         c_tic, c_ric, c_inv_depth[feature_id], c_time_delay);
            } else {
                auto *cost_function = new ProjectionFactor(point0.point, point.point);
                problem.AddResidualBlock(cost_function, loss_function,
                                         c_pos[start_frame_id], c_quat[start_frame_id],
                                         c_pos[cur_frame_id], c_quat[cur_frame_id],
                                         c_tic, c_ric, c_inv_depth[feature_id]);
            }
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = param.max_iter_num;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    eigen2c(window, features_, RunInfo::Instance().tic, RunInfo::Instance().ric);
}

void SlideWindowEstimator::marginalize(std::vector<FeaturesOfId> &features_, BundleAdjustWindow &window) {

}