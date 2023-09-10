//
// Created by gjt on 5/14/23.
//

#ifndef VINS_VISUAL_INERTIAL_ALIGNER_H
#define VINS_VISUAL_INERTIAL_ALIGNER_H

#include <Eigen/Eigen>
#include "vins/impl/vins_run_info.h"

namespace vins {
    Eigen::Vector3d estimateGyroBias(const std::vector<Eigen::Matrix3d> &imu_delta_rots,
                                     const std::vector<Eigen::Matrix3d> &img_delta_rots,
                                     const std::vector<Eigen::Matrix3d> &jacobians_bg_2_rot);
    bool estimateRIC(const std::vector<Eigen::Matrix3d> &img_rots,
                     const std::vector<Eigen::Matrix3d> &imu_rots,
                     Eigen::Matrix3d &calib_ric_result);

    bool estimateGravityScaleVelocity(const std::vector<Frame> &all_frames,
                                      ConstVec3dRef TIC,
                                      Eigen::Vector3d &gravity,
                                      double &scale,
                                      std::vector<Eigen::Vector3d> &vel);
    Eigen::Matrix3d rotGravityToZAxis(ConstVec3dRef gravity, ConstMat3dRef R0);
}

#endif //VINS_VISUAL_INERTIAL_ALIGNER_H
