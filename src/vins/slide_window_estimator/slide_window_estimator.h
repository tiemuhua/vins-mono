//
// Created by gjt on 5/14/23.
//

#ifndef VINS_BATCH_ADJUSTER_H
#define VINS_BATCH_ADJUSTER_H

#include "vins/vins_define_internal.h"
#include "vins/vins_run_info.h"
#include "vins/parameters.h"
#include "vins/feature_helper.h"

namespace vins::SlideWindowEstimator{
    void setLoopMatchInfo(vins::LoopMatchInfo*);
    void optimize(const SlideWindowEstimatorParam &param,
                  std::vector<Feature> &features,
                  Window<EstimateState>& state_window,
                  Window<ImuIntegrator>& pre_int_window,
                  Eigen::Vector3d &tic,
                  Eigen::Matrix3d &ric);
    void slide(const SlideWindowEstimatorParam &param,
               int oldest_key_frame_id,
               FeatureHelper &feature_manager,
               Window<EstimateState>& state_window,
               Window<ImuIntegrator>& pre_int_window);
}

#endif //VINS_BATCH_ADJUSTER_H
