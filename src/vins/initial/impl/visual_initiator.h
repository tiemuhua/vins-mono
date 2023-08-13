//
// Created by gjt on 5/14/23.
//

#ifndef VINS_VISUAL_INITIATOR_H
#define VINS_VISUAL_INITIATOR_H
#include "vins_define_internal.h"
#include "feature_helper.h"
namespace vins{
    bool initiateByVisual(int key_frame_num,
                          const std::vector<Feature>& features,
                          std::vector<Frame> &all_frames);
}

#endif //VINS_VISUAL_INITIATOR_H
