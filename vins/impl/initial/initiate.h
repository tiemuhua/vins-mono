//
// Created by gjt on 6/24/23.
//

#ifndef GJT_VINS_INITIATE_H
#define GJT_VINS_INITIATE_H

#include "param.h"

#include "impl/vins_model.h"

namespace vins {
    namespace Initiate {
        bool initiate(const cv::Mat &camera_matrix, VinsModel &run_info);
    }
}


#endif //GJT_VINS_INITIATE_H
