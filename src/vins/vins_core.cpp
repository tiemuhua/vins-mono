//
// Created by gjt on 5/14/23.
//

#include "vins_core.h"
#include "vins_define_internal.h"
#include "initial/initiate.h"
#include "feature_tracker.h"
#include "ric_estimator.h"
#include "feature_manager.h"

namespace vins{

    void VinsCore::handleIMU(ConstVec3dRef acc, ConstVec3dRef gyr, double time_stamp) {
        Synchronized(read_imu_buf_mutex_) {
            acc_buf_.push(acc);
            gyr_buf_.push(gyr);
            time_stamp_buf_.push(time_stamp);
        }
    }

    static int frame_id = 0;
    static Eigen::Vector3d zero = Eigen::Vector3d::Zero();

    void VinsCore::handleImage(const cv::Mat &_img, double time_stamp) {
        auto imu_integrator = std::make_shared<ImuIntegrator>(0,0,0,0,0,zero,zero,zero,zero,zero);
        Synchronized(read_imu_buf_mutex_) {
            while (!time_stamp_buf_.empty() && time_stamp_buf_.front() < time_stamp) {
                imu_integrator->predict(time_stamp_buf_.front(), acc_buf_.front(), gyr_buf_.front());
                time_stamp_buf_.pop();
                acc_buf_.pop();
                gyr_buf_.pop();
            }
        }

        std::vector<FeaturePoint2D> feature_points = feature_tracker_->extractFeatures(_img, time_stamp);
        frame_id++;
        bool is_key_frame = feature_manager_->isKeyFrame(frame_id, feature_points);
        feature_manager_->addFeatures(frame_id, feature_points);
        all_frames_.emplace_back(std::move(feature_points),
                                 time_stamp,
                                 std::move(imu_integrator),
                                 is_key_frame);

        switch (vins_state_) {
            case kVinsStateEstimateExtrinsic:
                vins_state_ = _handleEstimateExtrinsic();
                break;
//            case kVinsStateInitial:
//                vins_state_ = _handleInitial();
//                break;
//            case kVinsStateNormal:
//                vins_state_ = _handleNormal();
//                break;
        }
    }

    VinsCore::EVinsState VinsCore::_handleEstimateExtrinsic(int frame_id){
        if (frames_cnt < 2) {
            return kVinsStateEstimateExtrinsic;
        }
        PointCorrespondences correspondences =
                feature_manager_->getCorrespondences(frames_cnt - 2, frames_cnt - 1);
        Eigen::Matrix3d calibrated_ric;
        bool succ = ric_estimator_->calibrateRotationExtrinsic(correspondences,
                                                              all_frames_.back().pre_integral_->deltaQuat(),
                                                              calibrated_ric);
        if (!succ) {
            return kVinsStateEstimateExtrinsic;
        }
        ric_ = calibrated_ric;
        return kVinsStateInitial;
    }

    VinsCore::EVinsState VinsCore::_handleInitial(int frame_id, double time_stamp){
        if (time_stamp - last_init_time_stamp_ < 0.1) {
            return kVinsStateInitial;
        }
        last_init_time_stamp_ = time_stamp;
        if (Initiate::initiate(frame_id, RunInfo::Instance().tic, RunInfo::Instance().ric, RunInfo::Instance().window, all_frames_, ))
    }

//    VinsCore::EVinsState VinsCore::_handleNormal(const FeatureTracker::FeaturesPerImage& image_features){
//
//    }

}