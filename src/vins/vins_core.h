//
// Created by gjt on 5/14/23.
//

#ifndef VINS_VINS_CORE_H
#define VINS_VINS_CORE_H

#include <vector>
#include <mutex>
#include <Eigen/Eigen>

#include "feature_tracker.h"
#include "vins_define_internal.h"
#include "rotation_extrinsic_estimator.h"
#include "feature_manager.h"

namespace vins{
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
        EVinsState _handleEstimateExtrinsic();
        EVinsState _handleInitial(double time_stamp);
        EVinsState _handleNormal(double time_stamp);

    private:
        std::mutex read_imu_buf_mutex_;
        std::queue<Eigen::Vector3d> acc_buf_;
        std::queue<Eigen::Vector3d> gyr_buf_;
        std::queue<double> time_stamp_buf_;

        double last_init_time_stamp_ = 0.0;

        std::vector<ImageFrame> all_frames_;

        RotationExtrinsicEstimator rotation_extrinsic_estimator_;
        FeatureManager feature_manager_;

        Eigen::Matrix3d ric_;
        Eigen::Vector3d tic_;
    };
}


#endif //VINS_VINS_CORE_H
