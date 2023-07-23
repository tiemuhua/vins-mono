//
// Created by gjt on 5/14/23.
//

#include "slide_window_estimator.h"

#include <ceres/ceres.h>

#include "log.h"
#include "vins/vins_utils.h"
#include "vins/vins_define_internal.h"

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
static arr4d c_quat[WINDOW_SIZE];
static arr3d c_vel[WINDOW_SIZE];
static arr3d c_ba[WINDOW_SIZE];
static arr3d c_bg[WINDOW_SIZE];

static arr3d c_tic;
static arr4d c_ric;

static arr1d c_inv_depth[FeaturePointSize];
static arr1d c_time_delay;

static arr3d c_loop_peer_pos;
static arr4d c_loop_peer_quat;

static std::vector<double*> s_marginal_blocks;
static vins::MarginalInfo *sp_marginal_info;

static vins::LoopMatchInfo* sp_loop_match_info;

using namespace vins;
using namespace std;

static void eigen2c(const BundleAdjustWindow& window,
                    const std::vector<FeaturesOfId> &features,
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

    for (int i = 0; i < features.size(); ++i) {
        if (features[i].feature_points_.size() < 2 || features[i].start_frame_ >= WINDOW_SIZE - 2) {
            continue;
        }
        c_inv_depth[i][0] = features[i].inv_depth;
    }

    utils::vec3d2array(tic, c_tic);
    utils::quat2array(Eigen::Quaterniond(ric), c_ric);
}

static void c2eigen(BundleAdjustWindow &window,
                    std::vector<FeaturesOfId> &features,
                    Eigen::Vector3d& tic,
                    Eigen::Matrix3d &ric) {
    for (int i = 0; i < window.pos_window.size(); ++i) {
        window.pos_window.at(i) = utils::array2vec3d(c_pos[i]);
    }
    for (int i = 0; i < window.rot_window.size(); ++i) {
        window.rot_window.at(i) = utils::array2quat(c_quat[i]);
    }
    for (int i = 0; i < window.vel_window.size(); ++i) {
        window.vel_window.at(i) = utils::array2vec3d(c_vel[i]);
    }
    for (int i = 0; i < window.ba_window.size(); ++i) {
        window.ba_window.at(i) = utils::array2vec3d(c_ba[i]);
    }
    for (int i = 0; i < window.bg_window.size(); ++i) {
        window.bg_window.at(i) = utils::array2vec3d(c_bg[i]);
    }

    for (int i = 0; i < features.size(); ++i) {
        if (features[i].feature_points_.size() < 2 || features[i].start_frame_ >= WINDOW_SIZE - 2) {
            continue;
        }
        features[i].inv_depth = c_inv_depth[i][0];
    }

    tic = utils::array2vec3d(c_tic);
    ric = utils::array2quat(c_ric).toRotationMatrix();
}

void SlideWindowEstimator::optimize(const SlideWindowEstimatorParam &param,
                                    std::vector<FeaturesOfId> &features,
                                    BundleAdjustWindow &window,
                                    Eigen::Vector3d &tic,
                                    Eigen::Matrix3d &ric) {
    eigen2c(window, features, tic, ric);

    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::CauchyLoss(1.0);
    for (int i = 0; i < WINDOW_SIZE + 1; i++) {
        problem.AddParameterBlock(c_pos[i], 3);
        problem.AddParameterBlock(c_quat[i], 4, new ceres::EigenQuaternionManifold);
        problem.AddParameterBlock(c_vel[i], 3);
        problem.AddParameterBlock(c_ba[i], 3);
        problem.AddParameterBlock(c_bg[i], 3);
    }
    problem.AddParameterBlock(c_ric, 4, new ceres::EigenQuaternionManifold);
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
    problem.AddParameterBlock(c_loop_peer_pos, 3);
    problem.AddParameterBlock(c_loop_peer_quat, 4, new ceres::EigenQuaternionManifold);

    /*************** 1:边缘化 **************************/
    if (sp_marginal_info) {
        // todo tiemuhuaguo last_marginal_param_blocks和sp_marginal_info是怎么维护的
        auto *cost_function = new MarginalCost(sp_marginal_info);
        problem.AddResidualBlock(cost_function, nullptr, s_marginal_blocks);
    }

    /*************** 2:IMU **************************/
    for (int i = 0; i < window.pre_int_window.size(); i++) {
        int j = i + 1;
        if (window.pre_int_window.at(i).deltaTime() > 10.0)// todo why???
            continue;
        auto *cost_function = new IMUCost(window.pre_int_window.at(i));
        problem.AddResidualBlock(cost_function, nullptr,
                                 c_pos[i], c_quat[i], c_vel[i], c_ba[i], c_bg[i],
                                 c_pos[j], c_quat[j], c_vel[j], c_ba[j], c_bg[j]);
    }

    /*************** 3:特征点 **************************/
    for (int feature_id = 0; feature_id < features.size(); ++feature_id) {
        const FeaturesOfId &features_of_id = features[feature_id];
        if (features_of_id.feature_points_.size() < 2 || features_of_id.start_frame_ >= WINDOW_SIZE - 2)
            continue;
        int start_frame_id = features_of_id.start_frame_;
        const FeaturePoint2D & point0 = features_of_id.feature_points_[0];
        for (int frame_bias = 0; frame_bias < features_of_id.feature_points_.size(); ++frame_bias) {
            const FeaturePoint2D &point = features_of_id.feature_points_[frame_bias];
            int cur_frame_id = start_frame_id + frame_bias;
            if (param.estimate_time_delay) {
                auto *cost_function = new ProjectTdCost(point0, point);
                problem.AddResidualBlock(cost_function, loss_function,
                                         c_pos[start_frame_id], c_quat[start_frame_id],
                                         c_pos[cur_frame_id], c_quat[cur_frame_id],
                                         c_tic, c_ric, c_inv_depth[feature_id], c_time_delay);
            } else {
                auto *cost_function = new ProjectCost(point0.point, point.point);
                problem.AddResidualBlock(cost_function, loss_function,
                                         c_pos[start_frame_id], c_quat[start_frame_id],
                                         c_pos[cur_frame_id], c_quat[cur_frame_id],
                                         c_tic, c_ric, c_inv_depth[feature_id]);
            }
        }
    }

    /*************** 3:回环 **************************/
    if (sp_loop_match_info) {
        const auto match_points = sp_loop_match_info->match_points;
        for (int feature_id = 0; feature_id < features.size(); ++ feature_id) {
            FeaturesOfId feature = features[feature_id];
            int start = feature.start_frame_;
            if (feature.feature_points_.size() < 2 || start > WINDOW_SIZE - 2 || start < sp_loop_match_info->peer_frame_id) {
                continue;
            }
            auto iter = std::find_if(match_points.begin(), match_points.end(), [&feature](const MatchPoint& mp) {
                return mp.feature_id == feature.feature_id_;
            });
            if (iter == match_points.end()) {
                continue;
            }
            auto *cost_function = new ProjectCost(iter->point, feature.feature_points_[0].point);
            problem.AddResidualBlock(cost_function, loss_function,
                                     c_pos[start], c_quat[start],
                                     c_loop_peer_pos, c_loop_peer_quat,
                                     c_tic, c_ric,
                                     c_inv_depth[feature_id]);
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = param.max_iter_num;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    eigen2c(window, features, RunInfo::Instance().tic, RunInfo::Instance().ric);
}

static void marginalize(const SlideWindowEstimatorParam &param,
                        const std::vector<FeaturesOfId> &features,
                        BundleAdjustWindow &window,
                        std::vector<double *> &reserve_block_origin) {
    auto *marginal_info = new MarginalInfo();

    // 之前的边缘化约束
    if (sp_marginal_info) {
        std::vector<int> marginal_discard_set;
        for (int i = 0; i < static_cast<int>(s_marginal_blocks.size()); i++) {
            if (s_marginal_blocks[i] == c_pos[0] || s_marginal_blocks[i] == c_vel[0])
                marginal_discard_set.push_back(i);
        }
        auto *marginal_cost = new MarginalCost(sp_marginal_info);
        MarginalMetaFactor marginal_factor(marginal_cost, nullptr, s_marginal_blocks, marginal_discard_set);
        marginal_info->addMetaFactor(marginal_factor);
    }

    // 最老帧的imu约束
    auto *imu_cost = new IMUCost(window.pre_int_window.at(0));
    vector<double *> imu_param_blocks = {
            c_pos[0], c_quat[0], c_vel[0], c_ba[0], c_bg[0],
            c_pos[1], c_quat[1], c_vel[1], c_ba[1], c_bg[1],
    };
    const vector<int> imu_discard_set = {0, 1, 2, 3 ,4};
    MarginalMetaFactor imu_factor(imu_cost, nullptr, imu_param_blocks, imu_discard_set);
    marginal_info->addMetaFactor(imu_factor);

    // 最老帧的视觉约束
    const vector<int> visual_discard_set = {0, 1, 6};
    for (int feature_id = 0; feature_id < features.size(); ++feature_id) {
        const FeaturesOfId& feature = features[feature_id];
        if (feature.start_frame_ != window.frame_id_window.at(0)) {
            continue;
        }
        const FeaturePoint2D & point0 = feature.feature_points_[0];
        for (int i = 1; i < feature.feature_points_.size(); ++i) {
            const FeaturePoint2D & point = feature.feature_points_[i];
            ceres::LossFunction *loss_function = new ceres::CauchyLoss(1.0);
            if (param.estimate_time_delay) {
                auto *project_td_cost = new ProjectTdCost(point0, point);
                std::vector<double *> parameter_blocks = {
                        c_pos[0], c_quat[0],
                        c_pos[i], c_quat[i],
                        c_tic, c_ric,
                        c_inv_depth[feature_id],
                        c_time_delay
                };
                MarginalMetaFactor project_td_factor(project_td_cost, loss_function, parameter_blocks, visual_discard_set);
                marginal_info->addMetaFactor(project_td_factor);
            } else {
                auto *project_cost = new ProjectCost(point0.point, point.point);
                std::vector<double *> parameter_blocks = {
                        c_pos[0], c_quat[0],
                        c_pos[i], c_quat[i],
                        c_tic, c_ric,
                        c_inv_depth[feature_id],
                };
                MarginalMetaFactor project_factor(project_cost, loss_function, parameter_blocks, visual_discard_set);
                marginal_info->addMetaFactor(project_factor);
            }
        }
    }

    marginal_info->marginalize(reserve_block_origin);

    delete sp_marginal_info;
    sp_marginal_info = marginal_info;
}

void SlideWindowEstimator::slide(const SlideWindowEstimatorParam &param,
                                 FeatureManager &feature_manager,
                                 BundleAdjustWindow &window) {
    std::vector<double *> reserve_block_origin;
    marginalize(param, feature_manager.features_, window, reserve_block_origin);

    std::unordered_map<int, int> feature_id_2_idx_origin = feature_manager.getFeatureId2Index();
    feature_manager.discardFeaturesOfFrameId(window.frame_id_window.at(0));
    std::unordered_map<int, int> feature_id_2_idx_after_discard = feature_manager.getFeatureId2Index();

    std::unordered_map<double*, double*> slide_addr_map;
    for (const auto &id2idx_origin : feature_id_2_idx_origin) {
        int id = id2idx_origin.first;
        int idx_origin = id2idx_origin.second;
        int idx_after_discard = feature_id_2_idx_after_discard[id];
        slide_addr_map[c_inv_depth[idx_origin]] = c_inv_depth[idx_after_discard];
    }
    for (int frame_idx = 0; frame_idx < window.size - 1; ++frame_idx) {
        slide_addr_map[c_pos[frame_idx]] = c_pos[frame_idx + 1];
        slide_addr_map[c_quat[frame_idx]] = c_quat[frame_idx + 1];
        slide_addr_map[c_vel[frame_idx]] = c_vel[frame_idx + 1];
        slide_addr_map[c_ba[frame_idx]] = c_ba[frame_idx + 1];
        slide_addr_map[c_bg[frame_idx]] = c_bg[frame_idx + 1];
    }
    for (double* origin_addr:reserve_block_origin) {
        s_marginal_blocks.emplace_back(slide_addr_map[origin_addr]);
    }
}
