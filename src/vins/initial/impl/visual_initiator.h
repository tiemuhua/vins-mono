//
// Created by gjt on 5/14/23.
//

#ifndef VINS_VISUAL_INITIATOR_H
#define VINS_VISUAL_INITIATOR_H
#include "vins_define_internal.h"
#include "feature_manager.h"
namespace vins{
    class VisualInitiator {
    public:
        static bool initialStructure(const FeatureManager& feature_manager,
                                     int key_frame_num,
                                     std::vector<Frame> &all_frames);
    };
}

#endif //VINS_VISUAL_INITIATOR_H
