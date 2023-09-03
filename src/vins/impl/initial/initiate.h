//
// Created by gjt on 6/24/23.
//

#ifndef GJT_VINS_INITIATE_H
#define GJT_VINS_INITIATE_H

#include "vins/param.h"

#include "vins/impl/vins_run_info.h"

namespace vins {
    class Initiate {
    public:
        static bool initiate(double gravity_norm, RunInfo &run_info);
    };
}


#endif //GJT_VINS_INITIATE_H
