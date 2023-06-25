//
// Created by gjt on 6/24/23.
//

#include "initiate.h"
#include <vector>
#include "impl/visual_inertial_aligner.h"
#include "impl/visual_initiator.h"

using namespace vins;
bool Initiate::initiate(const int frame_cnt,
                        ConstVec3dRef TIC,
                        ConstMat3dRef RIC,
                        BundleAdjustWindow& window,
                        std::vector<ImageFrame> &all_frames,
                        Eigen::Vector3d& gravity,
                        FeatureManager &feature_manager) {
    bool visual_succ = VisualInitiator::initialStructure(feature_manager, frame_cnt, all_frames);
    if (!visual_succ) {
        return false;
    }

    Eigen::Vector3d delta_bg;
    Eigen::Matrix3d rot_diff;
    std::vector<Eigen::Vector3d> velocities;
    double scale;
    bool align_succ = VisualInertialAligner::visualInitialAlignImpl(TIC,
                                                                    RIC,
                                                                    all_frames,
                                                                    gravity,
                                                                    delta_bg,
                                                                    rot_diff,
                                                                    velocities,
                                                                    scale);
    if (!align_succ) {
        return false;
    }

    for (int i = 0; i < window.bg_window.size(); ++i) {
        window.bg_window.at(i) = window.bg_window.at(i) + delta_bg;
    }
    int frame_size = (int )all_frames.size();
    for (int frame_id = 0, key_frame_id = 0; frame_id < frame_size; frame_id++) {
        if (!all_frames[frame_id].is_key_frame_) {
            continue;
        }
        window.rot_window.at(key_frame_id) = rot_diff * all_frames.at(key_frame_id).R;
        window.pos_window.at(key_frame_id) = rot_diff * all_frames.at(key_frame_id).T;
        window.vel_window.at(key_frame_id) = rot_diff * velocities.at(key_frame_id);
        key_frame_id++;
    }
    for (int i = 0; i < (int) window.pos_window.size(); ++i) {
        window.pos_window.at(i) =
                scale * window.pos_window.at(i)
                - window.rot_window.at(i) * TIC
                - scale * window.pos_window.at(0)
                + window.rot_window.at(0) * TIC;
    }

    for (int i = 0; i < (int) window.pre_int_window.size(); ++i) {
        window.pre_int_window.at(i).rePredict(Eigen::Vector3d::Zero(), window.bg_window.at(i));
    }

    //triangulate on cam pose , no tic
    feature_manager.clearDepth();
    feature_manager.triangulate(window.pos_window, window.rot_window, Eigen::Vector3d::Zero(), RIC);
    for (FeaturesOfId &features_of_id: feature_manager.features_) {
        if (features_of_id.feature_points_.size() < 2 || features_of_id.start_frame_ >= Param::Instance().window_size - 2) {
            continue;
        }
        features_of_id.inv_depth *= scale; // todo tiemuhuaguo 这里是乘还是除？？
    }
    return true;
}
