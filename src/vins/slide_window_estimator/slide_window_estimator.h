//
// Created by gjt on 5/14/23.
//

#ifndef VINS_BATCH_ADJUSTER_H
#define VINS_BATCH_ADJUSTER_H

#include "vins/vins_define_internal.h"
#include "vins/vins_run_info.h"
#include "vins/parameters.h"

namespace vins::SlideWindowEstimator{
    void optimize(const SlideWindowEstimatorParam &param,
                  std::vector<FeaturesOfId> &features_,
                  BundleAdjustWindow &window,
                  Eigen::Vector3d &tic,
                  Eigen::Matrix3d &ric);
    void marginalize(std::vector<FeaturesOfId> &features_,
                     BundleAdjustWindow &window);
}

#endif //VINS_BATCH_ADJUSTER_H
