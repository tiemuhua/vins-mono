//
// Created by gjt on 5/14/23.
//

#include "vins_core.h"
#include "vins_define_internal.h"
#include "initial/initiate.h"
#include "feature_tracker.h"
#include "ric_estimator.h"
#include "feature_manager.h"
#include "slide_window_estimator/slide_window_estimator.h"

namespace vins{

    void VinsCore::handleIMU(ConstVec3dRef acc, ConstVec3dRef gyr, double time_stamp) {
        Synchronized(read_imu_buf_mutex_) {
            acc_buf_.push(acc);
            gyr_buf_.push(gyr);
            time_stamp_buf_.push(time_stamp);
        }
    }

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
        cur_frame_id_++;
        bool is_key_frame = feature_manager_->isKeyFrame(cur_frame_id_, feature_points);
        feature_manager_->addFeatures(cur_frame_id_, feature_points);
        RunInfo::Instance().all_frames.emplace_back(std::move(feature_points),
                                                    time_stamp,
                                                    std::move(imu_integrator),
                                                    is_key_frame);

        switch (vins_state_) {
            case kVinsStateEstimateExtrinsic:
                vins_state_ = _handleEstimateExtrinsic();
                break;
            case kVinsStateInitial:
                vins_state_ = _handleInitial(time_stamp);
                break;
            case kVinsStateNormal:
                vins_state_ = _handleNormal(is_key_frame);
                break;
        }
    }

    VinsCore::EVinsState VinsCore::_handleEstimateExtrinsic(){
        if (RunInfo::Instance().all_frames.size() < 2) {
            return kVinsStateEstimateExtrinsic;
        }
        PointCorrespondences correspondences =
                feature_manager_->getCorrespondences(cur_frame_id_ - 1, cur_frame_id_);
        Eigen::Quaterniond imu_quat = RunInfo::Instance().all_frames.back().pre_integral_->deltaQuat();
        bool succ = ric_estimator_->calibrateRotationExtrinsic(correspondences, imu_quat, RunInfo::Instance().ric);
        if (!succ) {
            return kVinsStateEstimateExtrinsic;
        }
        return kVinsStateInitial;
    }

    VinsCore::EVinsState VinsCore::_handleInitial(double time_stamp){
        if (time_stamp - last_init_time_stamp_ < 0.1) {
            return kVinsStateInitial;
        }
        last_init_time_stamp_ = time_stamp;
        bool rtn = Initiate::initiate(cur_frame_id_, RunInfo::Instance(), *feature_manager_);
        if (!rtn) {
            return kVinsStateInitial;
        }
        return kVinsStateNormal;
    }

    VinsCore::EVinsState VinsCore::_handleNormal(bool is_key_frame){
        if (!is_key_frame) {
            return kVinsStateNormal;
        }
        SlideWindowEstimator::slide(Param::Instance().slide_window,
                                    RunInfo::Instance().frame_id_window.at(0),
                                    *feature_manager_,
                                    RunInfo::Instance().state_window,
                                    RunInfo::Instance().pre_int_window);
        // todo 什么时候往Window里面塞东西？
        SlideWindowEstimator::optimize(Param::Instance().slide_window,
                                       feature_manager_->features_,
                                       RunInfo::Instance().state_window,
                                       RunInfo::Instance().pre_int_window,
                                       RunInfo::Instance().tic,
                                       RunInfo::Instance().ric);
        // todo window push
    }
}