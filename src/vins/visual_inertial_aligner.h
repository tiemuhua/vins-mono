//
// Created by gjt on 5/14/23.
//

#ifndef VINS_VISUAL_INERTIAL_ALIGNER_H
#define VINS_VISUAL_INERTIAL_ALIGNER_H

#include <Eigen/Eigen>
#include "vins_define_internal.h"
#include "imu_integrator.h"
#include "feature_manager.h"

namespace vins {
    class VisualInertialAligner {
    public:
        static bool visualInitialAlign(const double gravity_norm, ConstVec3dRef TIC, ConstMat3dRef RIC,
                                       Window<Eigen::Vector3d> &bg_window, Window<Eigen::Vector3d>& pos_window,
                                       Window<Eigen::Matrix3d> &rot_window, Window<Eigen::Vector3d> &vel_window,
                                       Window<ImuIntegrator> &pre_integrate_window,
                                       std::vector<ImageFrame> &all_frames, Eigen::Vector3d& gravity,
                                       FeatureManager &feature_manager);
    };
}

#endif //VINS_VISUAL_INERTIAL_ALIGNER_H
