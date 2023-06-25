//
// Created by gjt on 6/24/23.
//

#ifndef GJT_VINS_INITIATE_H
#define GJT_VINS_INITIATE_H
#include "vins_define_internal.h"
#include "parameters.h"
#include "vins_run_info.h"
#include "feature_manager.h"

namespace vins {
    class Initiate {
    public:
        static bool initiate(int frame_cnt,
                             ConstVec3dRef TIC,
                             ConstMat3dRef RIC,
                             BundleAdjustWindow& window,
                             std::vector<ImageFrame> &all_frames,
                             Eigen::Vector3d& gravity,
                             FeatureManager &feature_manager);
    };
}


#endif //GJT_VINS_INITIATE_H
