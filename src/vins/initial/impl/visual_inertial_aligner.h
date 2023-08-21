//
// Created by gjt on 5/14/23.
//

#ifndef VINS_VISUAL_INERTIAL_ALIGNER_H
#define VINS_VISUAL_INERTIAL_ALIGNER_H

#include <Eigen/Eigen>
#include "vins_run_info.h"

namespace vins {
    bool alignVisualAndInertial(double gravity_norm,
                                ConstVec3dRef TIC,
                                ConstMat3dRef RIC,
                                std::vector<Frame> &all_frames,
                                Eigen::Vector3d& gravity,
                                Eigen::Vector3d& delta_bg,
                                Eigen::Matrix3d& rot_diff,
                                std::vector<Eigen::Vector3d> &velocities,
                                double &scale);
}

#endif //VINS_VISUAL_INERTIAL_ALIGNER_H
