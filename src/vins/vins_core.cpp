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
#include "camera_wrapper.h"
#include "log.h"

namespace vins{
    VinsCore::VinsCore(Param* param) {
        run_info_ = new RunInfo();
        param_ = param;
        ric_estimator_ = new RICEstimator(param->window_size);
        camera_wrapper_ = new CameraWrapper(param);
        feature_tracker_ = new FeatureTracker(param, camera_wrapper_);
        loop_closer_ = new LoopCloser();
    }

    static PrevIMUInput s_prev_imu_input;

    void VinsCore::handleIMU(ConstVec3dRef acc, ConstVec3dRef gyr, double time_stamp) {
        Synchronized(read_imu_buf_mutex_) {
            if (vins_state_ == EVinsState::kNoIMUData) {
                s_prev_imu_input.acc = acc;
                s_prev_imu_input.gyr = gyr;
                s_prev_imu_input.time_stamp = time_stamp;
                vins_state_ = EVinsState::kNoImgData;
            } else if (vins_state_ == EVinsState::kNoImgData) {
                s_prev_imu_input.acc = acc;
                s_prev_imu_input.gyr = gyr;
                s_prev_imu_input.time_stamp = time_stamp;
            } else {
                cur_pre_integral_ptr_->predict(time_stamp, acc, gyr);
            }
        }
    }

    static const Eigen::Vector3d zero = Eigen::Vector3d::Zero();

    static State recurseByImu(const State& prev_state, const ImuIntegrator& pre_integral) {
        const Eigen::Vector3d& delta_pos = pre_integral.deltaPos();
        const Eigen::Vector3d& delta_vel = pre_integral.deltaVel();
        const Eigen::Quaterniond& delta_quat = pre_integral.deltaQuat();

        Eigen::Vector3d prev_pos = prev_state.pos;
        Eigen::Vector3d prev_vel = prev_state.vel;
        Eigen::Matrix3d prev_rot = prev_state.rot;
        Eigen::Vector3d prev_ba = prev_state.ba;
        Eigen::Vector3d prev_bg = prev_state.bg;

        Eigen::Vector3d cur_pos = prev_pos + prev_rot * delta_pos;
        Eigen::Vector3d cur_vel = prev_vel + prev_rot * delta_vel;
        Eigen::Matrix3d cur_rot = delta_quat.toRotationMatrix() * prev_rot;

        State cur_state = {cur_pos, cur_rot, cur_vel, prev_ba, prev_bg};
        return cur_state;
    }

    void VinsCore::handleImage(const cv::Mat &_img, double time_stamp) {
        if (vins_state_ == EVinsState::kNoIMUData) {
            return;
        }

        std::vector<FeaturePoint2D> feature_points = feature_tracker_->extractFeatures(_img, time_stamp);
        bool is_key_frame = FeatureHelper::isKeyFrame(cur_frame_id_, feature_points, run_info_->feature_window);
        run_info_->all_frames.emplace_back(feature_points, cur_pre_integral_ptr_, is_key_frame);

        if (vins_state_ == EVinsState::kNoImgData) {
            FeatureHelper::addFeatures(cur_frame_id_, time_stamp, feature_points, run_info_->feature_window);
            run_info_->state_window.push_back({});
            cur_pre_integral_ptr_ = std::make_shared<ImuIntegrator>(param_->imu_param, s_prev_imu_input, zero);

            vins_state_ = EVinsState::kEstimateExtrinsic;
            return;
        }

        if (is_key_frame) {
            FeatureHelper::addFeatures(cur_frame_id_, time_stamp, std::move(feature_points), run_info_->feature_window);
            run_info_->state_window.emplace_back(recurseByImu(run_info_->state_window.back(), *cur_pre_integral_ptr_));
            run_info_->pre_int_window.emplace_back(*cur_pre_integral_ptr_);
            cur_pre_integral_ptr_ = std::make_shared<ImuIntegrator>(param_->imu_param, s_prev_imu_input, zero);
        }

        switch (vins_state_) {
            case EVinsState::kEstimateExtrinsic:
                vins_state_ = _handleEstimateExtrinsic();
                break;
            case EVinsState::kInitial:
                vins_state_ = _handleInitial(time_stamp);
                break;
            case EVinsState::kNormal:
                vins_state_ = _handleNormal(_img, is_key_frame);
                break;
        }
    }

    VinsCore::EVinsState VinsCore::_handleEstimateExtrinsic(){
        if (run_info_->all_frames.size() < 2) {
            return EVinsState::kEstimateExtrinsic;
        }
        PointCorrespondences correspondences =
                FeatureHelper::getCorrespondences(cur_frame_id_ - 1, cur_frame_id_, run_info_->feature_window);
        Eigen::Quaterniond imu_quat = run_info_->all_frames.back().pre_integral_->deltaQuat();
        bool succ = ric_estimator_->estimate(correspondences, imu_quat, run_info_->ric);
        if (!succ) {
            LOG_E("estimate extrinsic false, please rotate rapidly");
            return EVinsState::kEstimateExtrinsic;
        }
        return EVinsState::kInitial;
    }

    VinsCore::EVinsState VinsCore::_handleInitial(double time_stamp){
        if (time_stamp - last_init_time_stamp_ < 0.1) {
            return EVinsState::kInitial;
        }
        last_init_time_stamp_ = time_stamp;
        bool rtn = Initiate::initiate(param_->gravity_norm, *run_info_);
        if (!rtn) {
            return EVinsState::kInitial;
        }
        return EVinsState::kNormal;
    }

    VinsCore::EVinsState VinsCore::_handleNormal(const cv::Mat &_img,
                                                 bool is_key_frame,
                                                 const ImuIntegrator& pre_integral){
        if (!is_key_frame) {
            return EVinsState::kNormal;
        }
        SlideWindowEstimator::slide(param_->slide_window,
                                    run_info_->feature_window,
                                    run_info_->state_window,
                                    run_info_->pre_int_window);
        SlideWindowEstimator::optimize(param_->slide_window,
                                       run_info_->feature_window,
                                       run_info_->state_window,
                                       run_info_->pre_int_window,
                                       run_info_->tic,
                                       run_info_->ric);
        // todo 失败检测与状态恢复
        bool fail;
        if (fail) {
            return EVinsState::kInitial;
        }

        // todo 如果有个特征点第一帧不是关键帧，应该怎么办？
        std::vector<cv::Point3f> key_pts_3d;
        for (const cv::Point2f &p2d:run_info_->all_frames.back().points) {
            double depth = -1; // todo
            key_pts_3d.emplace_back(utils::cvPoint2fToCvPoint3f(p2d, depth));
        }

        const int fast_th = 20; // corner detector response threshold
        std::vector<cv::KeyPoint> external_key_points_un_normalized;
        cv::FAST(_img, external_key_points_un_normalized, fast_th, true);
        std::vector<cv::Point2f> external_key_pts2d;
        for (const cv::KeyPoint & keypoint : external_key_points_un_normalized) {
            external_key_pts2d.push_back(camera_wrapper_->rawPoint2UniformedPoint(keypoint.pt));
        }

        loop_closer_->addKeyFrame(run_info_->all_frames.back(),
                                  _img,
                                  key_pts_3d,
                                  external_key_points_un_normalized,
                                  external_key_pts2d);
        return EVinsState::kNormal;
    }
}