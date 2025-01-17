//
// Created by gjt on 5/14/23.
//

#ifndef VINS_BATCH_ADJUSTER_H
#define VINS_BATCH_ADJUSTER_H

#include "param.h"
#include "impl/vins_model.h"
#include "impl/feature_helper.h"

namespace vins { namespace FrontEndOptimize {
    void optimize(const FrontEndOptimizeParam &param,
                  const std::vector<ImuIntegralUniPtr> &pre_int_window,
                  const std::vector<vins::LoopMatchInfo> &loop_match_infos,
                  std::vector<Feature> &feature_window,
                  std::vector<KeyFrameState> &state_window,
                  Eigen::Vector3d &tic,
                  Eigen::Matrix3d &ric);

    void slide(const Param &param,
               const std::vector<Feature> &oldest_features,
               const ImuIntegral &oldest_pre_integral,
               const std::unordered_map<int, int> &feature_id_2_idx_before_discard,
               const std::unordered_map<int, int> &feature_id_2_idx_after_discard);
} }

#endif //VINS_BATCH_ADJUSTER_H
