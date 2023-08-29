//
// Created by gjt on 6/24/23.
//

#ifndef GJT_VINS_INITIATE_H
#define GJT_VINS_INITIATE_H
#include "vins_define_internal.h"
#include "parameters.h"
#include "vins_run_info.h"
#include "feature_helper.h"

namespace vins {
    class Initiate {
    public:
        static bool initiate(const double gravity_norm, RunInfo &run_info);
    };
}


#endif //GJT_VINS_INITIATE_H
