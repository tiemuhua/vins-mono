//
// Created by gjt on 5/14/23.
//

#include "vins_core.h"

namespace vins{

    void VinsCore::handleIMU(ConstVec3dRef acc, ConstVec3dRef gyr, double time_stamp) {
        std::lock_guard<std::mutex> guard(emplace_image_frame_mutex_);
        all_image_frames_.back().pre_integrate_.predict(time_stamp, acc, gyr);
    }

    void VinsCore::handleImage(const FeatureTracker::FeaturesPerImage& image_features, double time_stamp) {
        switch (vins_state_) {
            case kVinsStateEstimateExtrinsic:
                vins_state_ = _handleEstimateExtrinsic(image_features);
                break;
            case kVinsStateInitial:
                vins_state_ = _handleInitial(image_features);
                break;
            case kVinsStateNormal:
                vins_state_ = _handleNormal(image_features);
                break;
        }
    }

    VinsCore::EVinsState VinsCore::_handleEstimateExtrinsic(const FeatureTracker::FeaturesPerImage& image_features){

    }
    VinsCore::EVinsState VinsCore::_handleInitial(const FeatureTracker::FeaturesPerImage& image_features){

    }
    VinsCore::EVinsState VinsCore::_handleNormal(const FeatureTracker::FeaturesPerImage& image_features){

    }

}