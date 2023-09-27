//
// Created by gjt on 6/9/23.
//

#ifndef VINS_FRAME_MATCHER_H
#define VINS_FRAME_MATCHER_H

#include "impl/loop_closer/keyframe.h"

namespace vins {
    /**
     * 若成功建立回环，返回true并设置new_kf->loop_relative_pose_、status、old_frame_pts2d
     * 否则返回false
     * */
    bool buildLoopRelation(ConstKeyFramePtr &old_kf,
                           int old_kf_id,
                           const KeyFramePtr &new_kf,
                           std::vector<uint8_t> &status,
                           std::vector<cv::Point2f> &old_frame_pts2d);
}

#endif //VINS_FRAME_MATCHER_H
