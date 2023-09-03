//
// Created by gjt on 5/14/23.
//

#ifndef VINS_VISUAL_INITIATOR_H
#define VINS_VISUAL_INITIATOR_H
#include "vins_define_internal.h"
#include "feature_helper.h"
namespace vins{
    bool initiateByVisual(int cur_window_size,
                          const std::vector<Feature>& feature_window,
                          std::vector<Frame> &all_frames);
}

#endif //VINS_VISUAL_INITIATOR_H
