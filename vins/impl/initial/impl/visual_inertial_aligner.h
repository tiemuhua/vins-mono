//
// Created by gjt on 5/14/23.
//

#ifndef VINS_VISUAL_INERTIAL_ALIGNER_H
#define VINS_VISUAL_INERTIAL_ALIGNER_H

#include <Eigen/Eigen>
#include "impl/vins_run_info.h"

namespace vins {
    Eigen::Vector3d estimateGyroBias(const std::vector<Eigen::Matrix3d> &imu_delta_rots,
                                     const std::vector<Eigen::Matrix3d> &img_delta_rots,
                                     const std::vector<Eigen::Matrix3d> &jacobians_bg_2_rot);

    bool estimateRIC(const std::vector<Eigen::Matrix3d> &img_rots,
                     const std::vector<Eigen::Matrix3d> &imu_rots,
                     Eigen::Matrix3d &calib_ric_result);

    bool estimateTICGravityScaleVelocity(const std::vector<Eigen::Matrix3d> &frames_img_rot,
                                         const std::vector<Eigen::Vector3d> &img_delta_poses,
                                         const std::vector<Eigen::Vector3d> &imu_delta_poses,
                                         const std::vector<Eigen::Vector3d> &imu_delta_velocities,
                                         const std::vector<double> &delta_times,
                                         const Eigen::Matrix3d &RIC,
                                         Eigen::Vector3d &TIC,
                                         Eigen::Vector3d &gravity,
                                         double &scale,
                                         std::vector<Eigen::Vector3d> &vel);

    Eigen::Matrix3d rotGravityToZAxis(const Eigen::Vector3d& gravity, const Eigen::Matrix3d& R0);
}

#endif //VINS_VISUAL_INERTIAL_ALIGNER_H
