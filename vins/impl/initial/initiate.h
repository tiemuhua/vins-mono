//
// Created by gjt on 6/24/23.
//

#ifndef GJT_VINS_INITIATE_H
#define GJT_VINS_INITIATE_H

#include "param.h"

#include "impl/vins_model.h"

namespace vins {
    class Initiate {
    public:
        static bool initiate(VinsModel &run_info);
    };
}


#endif //GJT_VINS_INITIATE_H
