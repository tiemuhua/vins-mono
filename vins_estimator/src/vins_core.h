//
// Created by gjt on 5/14/23.
//

#ifndef VINS_VINS_CORE_H
#define VINS_VINS_CORE_H

#include <vector>
#include <mutex>
#include "Eigen/Eigen"
#include "initial/initial_alignment.h"
#include "feature_tracker/src/feature_tracker.h"

namespace vins{
    typedef const Eigen::Vector3d & ConstVec3dRef;
    class VinsCore {
    public:
        void handleImage(const FeatureTracker::FeaturesPerImage& image_features, double time_stamp);
        void handleIMU(ConstVec3dRef acc, ConstVec3dRef gyr, double time_stamp);

    private:
        enum EVinsState {
            kVinsStateEstimateExtrinsic,    // 估计相机外参
            kVinsStateInitial,              // 初始化
            kVinsStateNormal,               // 正常优化
        } vins_state_ = kVinsStateEstimateExtrinsic;
        EVinsState _handleEstimateExtrinsic(const FeatureTracker::FeaturesPerImage& image_features);
        EVinsState _handleInitial(const FeatureTracker::FeaturesPerImage& image_features);
        EVinsState _handleNormal(const FeatureTracker::FeaturesPerImage& image_features);

    private:
        std::mutex emplace_image_frame_mutex_;
        std::vector<ImageFrame> all_image_frames_;
    };
}


#endif //VINS_VINS_CORE_H
