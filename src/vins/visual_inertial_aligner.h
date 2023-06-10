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
                                       BgWindow &bg_window, PosWindow& pos_window, RotWindow &rot_window,
                                       VelWindow &vel_window, PreIntegrateWindow &pre_integrate_window,
                                       std::vector<ImageFrame> &all_frames, Eigen::Vector3d& gravity,
                                       FeatureManager &feature_manager);
    };
}

#endif //VINS_VISUAL_INERTIAL_ALIGNER_H
