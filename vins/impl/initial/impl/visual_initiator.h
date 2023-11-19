//
// Created by gjt on 5/14/23.
//

#ifndef VINS_VISUAL_INITIATOR_H
#define VINS_VISUAL_INITIATOR_H

#include "impl/vins_model.h"

namespace vins {
    bool initiateByVisual(int window_size,
                          const std::vector<Feature> &feature_window,
                          const std::vector<Frame> &all_frames,
                          const cv::Mat &camera_matrix,
                          std::vector<Eigen::Matrix3d> &kf_img_rot,
                          std::vector<Eigen::Vector3d> &kf_img_pos,
                          std::vector<Eigen::Matrix3d> &frames_img_rot,
                          std::vector<Eigen::Vector3d> &frames_img_pos);
}

#endif //VINS_VISUAL_INITIATOR_H
