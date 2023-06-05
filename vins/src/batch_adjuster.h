//
// Created by gjt on 5/14/23.
//

#ifndef VINS_BATCH_ADJUSTER_H
#define VINS_BATCH_ADJUSTER_H

#include "vins_define_internal.h"
#include "feature_manager.h"

namespace vins{

    struct BatchAdjustParam{
        bool fix_extrinsic = false;
        bool estimate_time_delay = true;
        int max_iter_num = 100;
    };

    class BatchAdjuster {
    public:
        void optimize(const BatchAdjustParam &param,
                      const FeatureManager &feature_manager,
                      BundleAdjustWindow &window);
    };
}


#endif //VINS_BATCH_ADJUSTER_H
