//
// Created by gjt on 6/9/23.
//

#ifndef VINS_FRAME_MATCHER_H
#define VINS_FRAME_MATCHER_H
#include "keyframe.h"

namespace vins{
    namespace MatchFrame{
        bool findLoop(ConstKeyFramePtr old_kf,
                      const int old_kf_id,
                      ConstKeyFramePtr new_kf,
                      LoopInfo &loop_info);
    }
}

#endif //VINS_FRAME_MATCHER_H
