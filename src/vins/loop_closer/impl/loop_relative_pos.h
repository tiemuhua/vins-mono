//
// Created by gjt on 6/9/23.
//

#ifndef VINS_FRAME_MATCHER_H
#define VINS_FRAME_MATCHER_H
#include "keyframe.h"

namespace vins::LoopRelativePos{
    /**
     * 若成功建立回环，返回true并设置new_kf->loop_relative_pose_
     * 否则返回false
     * */
    bool find4DofLoopDrift(ConstKeyFramePtr old_kf,
                           int old_kf_id,
                           KeyFramePtr new_kf);
}

#endif //VINS_FRAME_MATCHER_H