//
// Created by gjt on 6/24/23.
//

#include "initiate.h"
#include "impl/visual_inertial_aligner.h"

bool visualInitialAlign(const double gravity_norm, ConstVec3dRef TIC, ConstMat3dRef RIC,
                                               Window<Eigen::Vector3d> &bg_window, Window<Eigen::Vector3d>& pos_window,
                                               Window<Eigen::Matrix3d> &rot_window, Window<Eigen::Vector3d> &vel_window,
                                               Window<ImuIntegrator> &pre_integrate_window,
                                               std::vector<ImageFrame> &all_frames, Eigen::Vector3d& gravity,
                                               FeatureManager &feature_manager) {

    for (int i = 0; i < bg_window.size(); ++i) {
        bg_window.at(i) = bg_window.at(i) + delta_bg;
    }
    for (int frame_id = 0, key_frame_id = 0; frame_id < frame_size; frame_id++) {
        if (!all_frames[frame_id].is_key_frame_) {
            continue;
        }
        rot_window.at(key_frame_id) = rot_diff * all_frames.at(key_frame_id).R;
        pos_window.at(key_frame_id) = rot_diff * all_frames.at(key_frame_id).T;
        vel_window.at(key_frame_id) = rot_diff * velocities.at(key_frame_id);
        key_frame_id++;
    }
    for (int i = 0; i < (int) pos_window.size(); ++i) {
        pos_window.at(i) = s * pos_window.at(i) - rot_window.at(i) * TIC - (s * pos_window.at(i) - rot_window.at(i) * TIC);
    }

    for (int i = 0; i < (int) pre_integrate_window.size(); ++i) {
        pre_integrate_window.at(i).rePredict(Eigen::Vector3d::Zero(), bg_window.at(i));
    }

    //triangulate on cam pose , no tic
    feature_manager.clearDepth();
    feature_manager.triangulate(pos_window, rot_window, Eigen::Vector3d::Zero(), RIC);
    for (FeaturesOfId &features_of_id: feature_manager.features_) {
        if (features_of_id.feature_points_.size() < 2 || features_of_id.start_frame_ >= Param::Instance().window_size - 2) {
            continue;
        }
        features_of_id.inv_depth *= s; // todo tiemuhuaguo 这里是乘还是除？？
    }
}