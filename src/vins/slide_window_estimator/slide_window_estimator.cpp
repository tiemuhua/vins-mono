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

static vins::MarginalInfo *sp_marginal_info;

static vins::LoopMatchInfo* sp_loop_match_info;

using namespace vins;
using namespace std;

static void eigen2c(const std::vector<KeyFrameState>& window,
                    const std::vector<Feature> &feature_window,
                    const Eigen::Vector3d& tic,
                    const Eigen::Matrix3d &ric){
    for (int i = 0; i < window.size(); ++i) {
        utils::vec3d2array(window.at(i).pos, c_pos[i]);
        utils::quat2array(Eigen::Quaterniond(window.at(i).rot), c_quat[i]);
        utils::vec3d2array(window.at(i).vel, c_vel[i]);
        utils::vec3d2array(window.at(i).ba, c_ba[i]);
        utils::vec3d2array(window.at(i).bg, c_bg[i]);
    }

    for (int i = 0; i < feature_window.size(); ++i) {
        if (feature_window[i].points.size() < 2 || feature_window[i].start_kf_idx >= WINDOW_SIZE - 2) {
            continue;
        }
        c_inv_depth[i][0] = feature_window[i].inv_depth;
    }

    utils::vec3d2array(tic, c_tic);
    utils::quat2array(Eigen::Quaterniond(ric), c_ric);
}

static void c2eigen(std::vector<KeyFrameState>& window,
                    std::vector<Feature> &feature_window,
                    Eigen::Vector3d& tic,
                    Eigen::Matrix3d &ric) {
    for (int i = 0; i < window.size(); ++i) {
        window.at(i).pos = utils::array2vec3d(c_pos[i]);
        window.at(i).rot = utils::array2quat(c_quat[i]).toRotationMatrix();
        window.at(i).vel = utils::array2vec3d(c_vel[i]);
        window.at(i).ba = utils::array2vec3d(c_ba[i]);
        window.at(i).bg = utils::array2vec3d(c_bg[i]);
    }

    for (int i = 0; i < feature_window.size(); ++i) {
        if (feature_window[i].points.size() < 2 || feature_window[i].start_kf_idx >= WINDOW_SIZE - 2) {
            continue;
        }
        feature_window[i].inv_depth = c_inv_depth[i][0];
    }

    tic = utils::array2vec3d(c_tic);
    ric = utils::array2quat(c_ric).toRotationMatrix();
}

void SlideWindowEstimator::setLoopMatchInfo(vins::LoopMatchInfo* loop_match_info) {
    sp_loop_match_info = loop_match_info;
}

void SlideWindowEstimator::optimize(const SlideWindowEstimatorParam &param,
                                    std::vector<Feature> &feature_window,
                                    std::vector<KeyFrameState>& state_window,
                                    std::vector<ImuIntegrator>& pre_int_window,
                                    Eigen::Vector3d &tic,
                                    Eigen::Matrix3d &ric) {
    eigen2c(state_window, feature_window, tic, ric);

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
        auto *cost_function = new MarginalCost(sp_marginal_info);
        problem.AddResidualBlock(cost_function, nullptr, sp_marginal_info->marginal_blocks);
    }

    /*************** 2:IMU **************************/
    for (int i = 0; i < pre_int_window.size(); i++) {
        int j = i + 1;
        if (pre_int_window.at(i).deltaTime() > 10.0)// todo why???
            continue;
        auto *cost_function = new IMUCost(pre_int_window.at(i));
        problem.AddResidualBlock(cost_function, nullptr,
                                 c_pos[i], c_quat[i], c_vel[i], c_ba[i], c_bg[i],
                                 c_pos[j], c_quat[j], c_vel[j], c_ba[j], c_bg[j]);
    }

    /*************** 3:特征点 **************************/
    for (int feature_idx = 0; feature_idx < feature_window.size(); ++feature_idx) {
        const Feature &feature = feature_window[feature_idx];
        if (feature.points.size() < 2 || feature.start_kf_idx >= WINDOW_SIZE - 2)
            continue;
        int start_kf_idx = feature.start_kf_idx;
        const cv::Point2f & point0 = feature.points[0];
        const cv::Point2f & vel0 = feature.velocities[0];
        const double time_stamp0 = feature.time_stamps_ms[0];
        for (int frame_idx_shift = 0; frame_idx_shift < feature.points.size(); ++frame_idx_shift) {
            const cv::Point2f &point = feature.points[frame_idx_shift];
            const cv::Point2f &vel = feature.velocities[frame_idx_shift];
            const double time_stamp = feature.time_stamps_ms[frame_idx_shift];
            int cur_frame_id = start_kf_idx + frame_idx_shift;
            if (param.estimate_time_delay) {
                auto *cost_function = new ProjectTdCost(point0, point, vel0, vel, time_stamp0, time_stamp);
                problem.AddResidualBlock(cost_function, loss_function,
                                         c_pos[start_kf_idx], c_quat[start_kf_idx],
                                         c_pos[cur_frame_id], c_quat[cur_frame_id],
                                         c_tic, c_ric, c_inv_depth[feature_idx], c_time_delay);
            } else {
                auto *cost_function = new ProjectCost(point0, point);
                problem.AddResidualBlock(cost_function, loss_function,
                                         c_pos[start_kf_idx], c_quat[start_kf_idx],
                                         c_pos[cur_frame_id], c_quat[cur_frame_id],
                                         c_tic, c_ric, c_inv_depth[feature_idx]);
            }
        }
    }

    /*************** 3:回环 **************************/
    if (sp_loop_match_info) {
        const auto& peer_frame_feature_ids = sp_loop_match_info->peer_frame.feature_ids;
        for (int feature_id = 0; feature_id < feature_window.size(); ++ feature_id) {
            Feature feature = feature_window[feature_id];
            int start = feature.start_kf_idx;
            if (feature.points.size() < 2 || start > WINDOW_SIZE - 2 || start < sp_loop_match_info->peer_frame_id) {
                continue;
            }
            int feature_idx = 0;
            for (; feature_idx < peer_frame_feature_ids.size(); ++feature_idx) {
                if (feature.feature_id == peer_frame_feature_ids[feature_idx]) {
                    break;
                }
            }
            if (feature_idx == peer_frame_feature_ids.size()) {
                continue;
            }
            auto *cost_function = new ProjectCost(sp_loop_match_info->peer_frame.points[feature_idx], feature.points[0]);
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

    eigen2c(state_window, feature_window, tic, ric);
}

static MarginalInfo* marginalize(const SlideWindowEstimatorParam &param,
                                 const std::vector<Feature> &oldest_features,
                                 const ImuIntegrator& oldest_pre_integral,
                                 std::vector<double *> &reserve_block_origin) {
    auto *marginal_info = new MarginalInfo();

    // 之前的边缘化约束
    if (sp_marginal_info) {
        std::vector<int> marginal_discard_set;
        for (int i = 0; i < (int)(sp_marginal_info->marginal_blocks.size()); i++) {
            if (sp_marginal_info->marginal_blocks[i] == c_pos[0] || sp_marginal_info->marginal_blocks[i] == c_vel[0])
                marginal_discard_set.push_back(i);
        }
        auto *marginal_cost = new MarginalCost(sp_marginal_info);
        MarginalMetaFactor marginal_factor(marginal_cost, nullptr, sp_marginal_info->marginal_blocks, marginal_discard_set);
        marginal_info->addMetaFactor(marginal_factor);
    }

    // 最老帧的imu约束
    auto *imu_cost = new IMUCost(oldest_pre_integral);
    vector<double *> imu_param_blocks = {
            c_pos[0], c_quat[0], c_vel[0], c_ba[0], c_bg[0],
            c_pos[1], c_quat[1], c_vel[1], c_ba[1], c_bg[1],
    };
    const vector<int> imu_discard_set = {0, 1, 2, 3 ,4};
    MarginalMetaFactor imu_factor(imu_cost, nullptr, imu_param_blocks, imu_discard_set);
    marginal_info->addMetaFactor(imu_factor);

    // 最老帧的视觉约束
    const vector<int> visual_discard_set = {0, 1, 6};
    for (int feature_idx = 0; feature_idx < oldest_features.size(); ++feature_idx) {
        const Feature& feature = oldest_features[feature_idx];
        const cv::Point2f & point0 = feature.points[0];
        const cv::Point2f & vel0 = feature.velocities[0];
        const double time_stamp0 = feature.time_stamps_ms[0];
        for (int frame_idx_shift = 0; frame_idx_shift < feature.points.size(); ++frame_idx_shift) {
            const cv::Point2f &point = feature.points[frame_idx_shift];
            const cv::Point2f &vel = feature.velocities[frame_idx_shift];
            const double time_stamp = feature.time_stamps_ms[frame_idx_shift];
            int cur_frame_id = feature.start_kf_idx + frame_idx_shift;
            ceres::LossFunction *loss_function = new ceres::CauchyLoss(1.0);
            if (param.estimate_time_delay) {
                auto *project_td_cost = new ProjectTdCost(point0, point, vel0, vel, time_stamp0, time_stamp);
                std::vector<double *> parameter_blocks = {
                        c_pos[0], c_quat[0],
                        c_pos[cur_frame_id], c_quat[cur_frame_id],
                        c_tic, c_ric,
                        c_inv_depth[feature_idx],
                        c_time_delay
                };
                MarginalMetaFactor project_td_factor(project_td_cost, loss_function, parameter_blocks, visual_discard_set);
                marginal_info->addMetaFactor(project_td_factor);
            } else {
                auto *project_cost = new ProjectCost(point0, point);
                std::vector<double *> parameter_blocks = {
                        c_pos[0], c_quat[0],
                        c_pos[cur_frame_id], c_quat[cur_frame_id],
                        c_tic, c_ric,
                        c_inv_depth[feature_idx],
                };
                MarginalMetaFactor project_factor(project_cost, loss_function, parameter_blocks, visual_discard_set);
                marginal_info->addMetaFactor(project_factor);
            }
        }
    }

    marginal_info->marginalize(reserve_block_origin);

    return marginal_info;
}

void SlideWindowEstimator::slide(const Param &param,
                                 const std::vector<Feature> &oldest_features,
                                 const ImuIntegrator& oldest_pre_integral,
                                 const std::unordered_map<int, int> &feature_id_2_idx_origin,
                                 const std::unordered_map<int, int> &feature_id_2_idx_after_discard) {
    std::vector<double *> reserve_block_origin;
    delete sp_marginal_info;
    sp_marginal_info = marginalize(param.slide_window, oldest_features, oldest_pre_integral, reserve_block_origin);

    std::unordered_map<double*, double*> slide_addr_map;
    for (const auto &id2idx_origin : feature_id_2_idx_origin) {
        int id = id2idx_origin.first;
        int idx_origin = id2idx_origin.second;
        int idx_after_discard = feature_id_2_idx_after_discard.at(id);
        slide_addr_map[c_inv_depth[idx_origin]] = c_inv_depth[idx_after_discard];
    }
    for (int frame_idx = 0; frame_idx < param.window_size - 1; ++frame_idx) {
        slide_addr_map[c_pos[frame_idx]] = c_pos[frame_idx + 1];
        slide_addr_map[c_quat[frame_idx]] = c_quat[frame_idx + 1];
        slide_addr_map[c_vel[frame_idx]] = c_vel[frame_idx + 1];
        slide_addr_map[c_ba[frame_idx]] = c_ba[frame_idx + 1];
        slide_addr_map[c_bg[frame_idx]] = c_bg[frame_idx + 1];
    }

    for (double* origin_addr:reserve_block_origin) {
        sp_marginal_info->marginal_blocks.emplace_back(slide_addr_map[origin_addr]);
    }
}
