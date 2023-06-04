//
// Created by gjt on 5/14/23.
//

#include "batch_adjuster.h"

#include <ceres/ceres.h>

#include "log.h"
#include "vins_define_internal.h"
#include "vins_utils.h"

#include "ceres_cost_function_factor/imu_factor.h"
#include "ceres_cost_function_factor/marginal_factor.h"
#include "ceres_cost_function_factor/projection_factor.h"
#include "ceres_cost_function_factor/projection_td_factor.h"

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

static std::vector<double*> s_marginal_param_blocks;
static std::shared_ptr<vins::MarginalInfo> sp_marginal_info;

namespace vins{

    void c2eigen(const BundleAdjustWindow& window,
                 const FeatureManager& feature_manager,
                 const Eigen::Vector3d& tic, const Eigen::Matrix3d &ric){
        for (int i = 0; i < window.pos_window.size(); ++i) {
            utils::vec3d2array(window.pos_window[i], c_pos[i]);
        }
        for (int i = 0; i < window.rot_window.size(); ++i) {
            utils::quat2array(Eigen::Quaterniond(window.rot_window[i]), c_quat[i]);
        }
        for (int i = 0; i < window.vel_window.size(); ++i) {
            utils::vec3d2array(window.vel_window[i], c_vel[i]);
        }
        for (int i = 0; i < window.ba_window.size(); ++i) {
            utils::vec3d2array(window.ba_window[i], c_ba[i]);
        }
        for (int i = 0; i < window.bg_window.size(); ++i) {
            utils::vec3d2array(window.bg_window[i], c_bg[i]);
        }

        std::vector<double> inv_depth_vec = feature_manager.getInvDepth();
        for (int i = 0; i < inv_depth_vec.size(); ++i) {
            c_inv_depth[i][0] = inv_depth_vec[i];
        }

        utils::vec3d2array(tic, c_tic);
        utils::quat2array(Eigen::Quaterniond(ric), c_ric);
    }

    void eigen2c() {

    }

    void BatchAdjuster::optimize(const BatchAdjustParam& param,
                                 const FeatureManager &feature_manager,
                                 BundleAdjustWindow &window) {
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
        for (int i = 0; i < WINDOW_SIZE; i++) {
            int j = i + 1;
            if (window.pre_int_window[i].deltaTime() > 10.0)// todo why???
                continue;
            auto *cost_function = new IMUFactor(&window.pre_int_window[i]);
            problem.AddResidualBlock(cost_function, nullptr,
                                     c_pos[i], c_quat[i], c_vel[i], c_ba[i], c_bg[i],
                                     c_pos[j], c_quat[j], c_vel[j], c_ba[j], c_bg[j]);
        }

        /*************** 3:特征点 **************************/
        for (int feature_id = 0; feature_id < feature_manager.features_.size(); ++feature_id) {
            const FeaturesOfId &features_of_id = feature_manager.features_[feature_id];
            if (features_of_id.feature_points_.size() < 2 || features_of_id.start_frame_ >= WINDOW_SIZE - 2)
                continue;
            int start_frame_id = features_of_id.start_frame_;
            const FeaturePoint & point0 = features_of_id.feature_points_[0];
            for (int frame_bias = 0; frame_bias < features_of_id.feature_points_.size(); ++frame_bias) {
                const FeaturePoint &point = features_of_id.feature_points_[frame_bias];
                int cur_frame_id = start_frame_id + frame_bias;
                if (param.estimate_time_delay) {
                    auto *cost_function = new ProjectionTdFactor(point0, point);
                    problem.AddResidualBlock(cost_function, loss_function,
                                             c_pos[start_frame_id], c_quat[start_frame_id],
                                             c_pos[cur_frame_id], c_quat[cur_frame_id],
                                             c_tic, c_ric, c_inv_depth[feature_id], c_time_delay);
                } else {
                    auto *cost_function = new ProjectionFactor(point0.unified_point, point.unified_point);
                    problem.AddResidualBlock(cost_function, loss_function,
                                             c_pos[start_frame_id], c_quat[start_frame_id],
                                             c_pos[cur_frame_id], c_quat[cur_frame_id],
                                             c_tic, c_ric, c_inv_depth[feature_id]);
                }
            }
        }

        /*************** 4:回环 **************************/
        if (param.is_re_localize) {
            ceres::Manifold *local_parameterization = new ceres::SE3Manifold();
            problem.AddParameterBlock(re_local_Pose, 7);
            int retrive_feature_index = 0;
            feature_index = -1;
            for (const FeaturesOfId &features_of_id: feature_manager_.features_) {
                if (features_of_id.feature_points_.size() < 2 || features_of_id.start_frame_ >= WINDOW_SIZE - 2)
                    continue;
                ++feature_index;
                int start = features_of_id.start_frame_;
                if (start <= re_local_frame_local_index) {
                    while ((int) match_points_[retrive_feature_index].feature_id < features_of_id.feature_id_) {
                        retrive_feature_index++;
                    }
                    if (match_points_[retrive_feature_index].feature_id == features_of_id.feature_id_) {
                        auto *cost_function = new ProjectionFactor(match_points_[retrive_feature_index].point,
                                                                   features_of_id.feature_points_[0].unified_point);
                        problem.AddResidualBlock(cost_function, loss_function,
                                                 para_Pose[start], re_local_Pose, para_Ex_Pose, para_FeatureInvDepth[feature_index]);
                        retrive_feature_index++;
                    }
                }
            }
        }

        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.trust_region_strategy_type = ceres::DOGLEG;
        options.max_num_iterations = param.max_iter_num;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
    }
}