//
// Created by gjt on 5/14/23.
//

#ifndef VINS_VISUAL_INITIATOR_H
#define VINS_VISUAL_INITIATOR_H
#include "vins_define_internal.h"
#include "feature_helper.h"
namespace vins{
    class VisualInitiator {
    public:
        static bool initialStructure(const std::vector<Feature>& features,
                                     int key_frame_num,
                                     std::vector<Frame> &all_frames);
    };
}

#endif //VINS_VISUAL_INITIATOR_H
