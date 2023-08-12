//
// Created by gjt on 6/24/23.
//

#include "initiate.h"
#include <vector>
#include "impl/visual_inertial_aligner.h"
#include "impl/visual_initiator.h"
#include "log.h"
using namespace vins;
using namespace std;

static bool isAccVariantBigEnough(const std::vector<Frame> &all_image_frame_) {
    //check imu observability
    Eigen::Vector3d sum_acc;
    // todo tiemuhuaguo 原始代码很奇怪，all_image_frame隔一个用一个，而且all_image_frame.size() - 1是什么意思？
    for (const Frame &frame: all_image_frame_) {
        double dt = frame.pre_integral_->deltaTime();
        Eigen::Vector3d tmp_acc = frame.pre_integral_->deltaVel() / dt;
        sum_acc += tmp_acc;
    }
    Eigen::Vector3d avg_acc = sum_acc / (double )all_image_frame_.size();
    double var = 0;
    for (const Frame &frame:all_image_frame_) {
        double dt = frame.pre_integral_->deltaTime();
        Eigen::Vector3d tmp_acc = frame.pre_integral_->deltaVel() / dt;
        var += (tmp_acc - avg_acc).transpose() * (tmp_acc - avg_acc);
    }

    var = sqrt(var / (double )all_image_frame_.size());
    LOG_I("IMU acc variant:%f", var);
    return var > 0.25;
}

bool Initiate::initiate(int frame_cnt,
                        RunInfo &run_info,
                        FeatureManager &feature_manager) {
    if (!isAccVariantBigEnough(run_info.all_frames)) {
        return false;
    }

    bool visual_succ = VisualInitiator::initialStructure(feature_manager, frame_cnt, run_info.all_frames);
    if (!visual_succ) {
        return false;
    }

    Eigen::Vector3d delta_bg;
    Eigen::Matrix3d rot_diff;
    std::vector<Eigen::Vector3d> velocities;
    double scale;
    bool align_succ = VisualInertialAligner::visualInitialAlignImpl(run_info.tic,
                                                                    run_info.ric,
                                                                    run_info.all_frames,
                                                                    run_info.gravity,
                                                                    delta_bg,
                                                                    rot_diff,
                                                                    velocities,
                                                                    scale);
    if (!align_succ) {
        return false;
    }

    auto &state_window = run_info.state_window;
    auto &all_frames = run_info.all_frames;
    auto &TIC = run_info.tic;
    auto RIC = run_info.ric;
    for (int i = 0; i < state_window.size(); ++i) {
        state_window.at(i).bg = state_window.at(i).bg + delta_bg;
    }
    int frame_size = (int )all_frames.size();
    for (int frame_id = 0, key_frame_id = 0; frame_id < frame_size; frame_id++) {
        if (!all_frames[frame_id].is_key_frame_) {
            continue;
        }
        state_window.at(key_frame_id).rot = rot_diff * all_frames.at(key_frame_id).R;
        state_window.at(key_frame_id).pos = rot_diff * all_frames.at(key_frame_id).T;
        state_window.at(key_frame_id).vel = rot_diff * velocities.at(key_frame_id);
        key_frame_id++;
    }
    for (int i = 0; i < (int) state_window.size(); ++i) {
        state_window.at(i).pos =
                (scale * state_window.at(i).pos - state_window.at(i).rot * TIC) -
                (scale * state_window.at(0).pos - state_window.at(0).rot * TIC);
    }

    for (int i = 0; i < (int) run_info.pre_int_window.size(); ++i) {
        run_info.pre_int_window.at(i).rePredict(Eigen::Vector3d::Zero(), state_window.at(i).bg);
    }

    //triangulate on cam pose , no tic
    feature_manager.triangulate(state_window.pos_window, state_window.rot_window, Eigen::Vector3d::Zero(), RIC);
    for (Feature &feature: feature_manager.features_) {
        if (feature.points.size() < 2 || feature.start_frame >= Param::Instance().window_size - 2) {
            continue;
        }
        feature.inv_depth *= scale; // todo tiemuhuaguo 这里是乘还是除？？
    }
    return true;
}
