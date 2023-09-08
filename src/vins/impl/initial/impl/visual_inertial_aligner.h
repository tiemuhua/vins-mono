//
// Created by gjt on 5/14/23.
//

#ifndef VINS_VISUAL_INERTIAL_ALIGNER_H
#define VINS_VISUAL_INERTIAL_ALIGNER_H

#include <Eigen/Eigen>
#include "vins/impl/vins_run_info.h"

namespace vins {
    Eigen::Vector3d solveGyroBias(const std::vector<Frame> &all_image_frame);
    bool linearAlignment(const std::vector<Frame> &all_frames,
                         ConstVec3dRef TIC,
                         double gravity_norm,
                         Eigen::Vector3d &g);
    void refineGravity(const std::vector<Frame> &all_frames,
                       double gravity_norm,
                       ConstVec3dRef TIC,
                       Eigen::Vector3d &g,
                       double &s,
                       std::vector<Eigen::Vector3d> &vel);
    Eigen::Matrix3d rotGravityToZAxis(ConstVec3dRef gravity, ConstMat3dRef R0);
}

#endif //VINS_VISUAL_INERTIAL_ALIGNER_H
