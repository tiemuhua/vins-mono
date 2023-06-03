//
// Created by gjt on 5/14/23.
//

#include "vins_core.h"

namespace vins{

    void VinsCore::handleIMU(ConstVec3dRef acc, ConstVec3dRef gyr, double time_stamp) {
        Synchronized(read_imu_buf_mutex_) {
            acc_buf_.push(acc);
            gyr_buf_.push(gyr);
            time_stamp_buf_.push(time_stamp);
        }
    }

    void VinsCore::handleImage(const FeatureTracker::FeaturesPerImage& image, double time_stamp) {
        Eigen::Vector3d zero;
        ImuIntegrator imu_integrator(0,0,0,0,0,zero,zero,zero,zero,zero);
        Synchronized(read_imu_buf_mutex_) {
            while (!time_stamp_buf_.empty() && time_stamp_buf_.front() < time_stamp) {
                imu_integrator.predict(time_stamp_buf_.front(), acc_buf_.front(), gyr_buf_.front());
                time_stamp_buf_.pop();
                acc_buf_.pop();
                gyr_buf_.pop();
            }
        }

        std::vector<FeaturePoint> feature_points;
        for (int i = 0; i < image.feature_ids.size(); ++i) {
            FeaturePoint point;
            point.unified_point = image.unified_points[i];
            point.point = image.points[i];
            point.point_velocity = image.points_velocity[i];
            point.feature_id = image.feature_ids[i];
            feature_points.emplace_back(std::move(point));
        }
        bool is_key_frame = feature_manager_.addFeatureCheckParallax((int) all_frames_.size(), feature_points);
        all_frames_.emplace_back(std::move(feature_points),
                                 time_stamp,
                                 std::move(imu_integrator),
                                 is_key_frame);

        switch (vins_state_) {
            case kVinsStateEstimateExtrinsic:
                vins_state_ = _handleEstimateExtrinsic();
                break;
            case kVinsStateInitial:
                vins_state_ = _handleInitial();
                break;
            case kVinsStateNormal:
                vins_state_ = _handleNormal();
                break;
        }
    }

    VinsCore::EVinsState VinsCore::_handleEstimateExtrinsic(){
        int frames_cnt = (int )all_frames_.size();
        if (frames_cnt < 2) {
            return kVinsStateEstimateExtrinsic;
        }
        PointCorrespondences correspondences =
                feature_manager_.getCorresponding( frames_cnt - 2, frames_cnt - 1);
        Eigen::Matrix3d calibrated_ric;
        bool succ = rotation_extrinsic_estimator_.calibrateRotationExtrinsic(correspondences,
                                                                             all_frames_.back().pre_integrate_.deltaQuat(),
                                                                             calibrated_ric);
        if (!succ) {
            return kVinsStateEstimateExtrinsic;
        }
        ric_ = calibrated_ric;
        return kVinsStateInitial;
    }

    VinsCore::EVinsState VinsCore::_handleInitial(double time_stamp){
        if (time_stamp - last_init_time_stamp_ < 0.1) {
            return kVinsStateInitial;
        }
        last_init_time_stamp_ = time_stamp;
        bool succ = ()
    }

    VinsCore::EVinsState VinsCore::_handleNormal(const FeatureTracker::FeaturesPerImage& image_features){

    }

}