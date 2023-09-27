//
// Created by gjt on 5/14/23.
//

#ifndef VINS_BATCH_ADJUSTER_H
#define VINS_BATCH_ADJUSTER_H

#include "vins/param.h"
#include "impl/vins_define_internal.h"
#include "impl/feature_helper.h"

namespace vins::FrontEndOptimize {
    void optimize(const FrontEndOptimizeParam &param,
                  const std::vector<ImuIntegratorPtr> &pre_int_window,
                  const std::vector<vins::LoopMatchInfo> &loop_match_infos,
                  std::vector<Feature> &feature_window,
                  std::vector<KeyFrameState> &state_window,
                  Eigen::Vector3d &tic,
                  Eigen::Matrix3d &ric);

    void slide(const Param &param,
               const std::vector<Feature> &oldest_features,
               const ImuIntegrator &oldest_pre_integral,
               const std::unordered_map<int, int> &feature_id_2_idx_origin,
               const std::unordered_map<int, int> &feature_id_2_idx_after_discard);
}

#endif //VINS_BATCH_ADJUSTER_H
