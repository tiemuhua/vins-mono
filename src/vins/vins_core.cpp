//
// Created by gjt on 5/14/23.
//

#include "vins_core.h"
#include "vins_utils.h"
#include "vins_define_internal.h"
#include "initial/initiate.h"
#include "feature_tracker.h"
#include "ric_estimator.h"
#include "feature_helper.h"
#include "slide_window_estimator/slide_window_estimator.h"
#include "loop_closer/loop_closer.h"

namespace vins{
    VinsCore::VinsCore(Param* param) {
        run_info_ = new RunInfo(param_->window_size);
        param_ = param;
        ric_estimator_ = new RICEstimator(param->window_size);
        feature_tracker_ = new FeatureTracker(param);
        loop_closer_ = new LoopCloser();
    }

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
        bool is_key_frame = FeatureHelper::isKeyFrame(cur_frame_id_, feature_points, run_info_->features);
        FeatureHelper::addFeatures(cur_frame_id_, time_stamp, feature_points, run_info_->features);
        run_info_->all_frames.emplace_back(std::move(feature_points),
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
                vins_state_ = _handleNormal(_img, is_key_frame);
                break;
        }
    }

    VinsCore::EVinsState VinsCore::_handleEstimateExtrinsic(){
        if (run_info_->all_frames.size() < 2) {
            return kVinsStateEstimateExtrinsic;
        }
        PointCorrespondences correspondences =
                FeatureHelper::getCorrespondences(cur_frame_id_ - 1, cur_frame_id_, run_info_->features);
        Eigen::Quaterniond imu_quat = run_info_->all_frames.back().pre_integral_->deltaQuat();
        bool succ = ric_estimator_->estimate(correspondences, imu_quat, run_info_->ric);
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
        bool rtn = Initiate::initiate(param_->gravity_norm, param_->window_size, cur_frame_id_, *run_info_);
        if (!rtn) {
            return kVinsStateInitial;
        }
        return kVinsStateNormal;
    }

    VinsCore::EVinsState VinsCore::_handleNormal(const cv::Mat &_img, bool is_key_frame){
        if (!is_key_frame) {
            return kVinsStateNormal;
        }
        SlideWindowEstimator::slide(param_->slide_window,
                                    run_info_->frame_id_window.at(0),
                                    run_info_->features,
                                    run_info_->state_window,
                                    run_info_->pre_int_window);
        // todo 什么时候往Window里面塞东西？
        SlideWindowEstimator::optimize(param_->slide_window,
                                       run_info_->features,
                                       run_info_->state_window,
                                       run_info_->pre_int_window,
                                       run_info_->tic,
                                       run_info_->ric);
        // todo 失败检测与状态恢复
        bool fail;
        if (fail) {
            return kVinsStateInitial;
        }

        // 如果有个特征点第一帧不是关键帧，应该怎么办？
        vector<cv::Point3f> key_pts_3d;
        for (const cv::Point2f &p2d:run_info_->all_frames.back().points) {
            Eigen::Vector3d p3d = utils::cvPoint2dToEigenVec3d(p2d, );
        }

        const int fast_th = 20; // corner detector response threshold
        std::vector<cv::KeyPoint> external_key_points_un_normalized;
        cv::FAST(_img, external_key_points_un_normalized, fast_th, true);
        std::vector<cv::Point2f> external_key_pts2d;
        for (const cv::KeyPoint & keypoint : external_key_points_un_normalized) {
            external_key_pts2d.push_back(feature_tracker_->rawPoint2UniformedPoint(keypoint.pt));
        }

        loop_closer_->addKeyFrame(run_info_->all_frames.back(),
                                  _img,
                                  key_pts_3d,
                                  external_key_points_un_normalized,
                                  external_key_pts2d);
        return kVinsStateNormal;
    }
}