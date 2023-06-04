//
// Created by gjt on 5/14/23.
//

#ifndef VINS_BATCH_ADJUSTER_H
#define VINS_BATCH_ADJUSTER_H

#include "vins_define_internal.h"
#include "feature_manager.h"

namespace vins{

    struct BatchAdjustParam{
        bool fix_extrinsic;
        bool estimate_time_delay;
        bool is_re_localize;
        int max_iter_num;
    };

    class BatchAdjuster {
    public:
        void optimize(const BatchAdjustParam& param,
                      const FeatureManager &feature_manager,
                      BundleAdjustWindow &window);
    };
}


#endif //VINS_BATCH_ADJUSTER_H
