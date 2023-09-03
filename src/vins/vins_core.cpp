//
// Created by gjt on 5/14/23.
//
#include <sys/time.h>

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

namespace vins {
    VinsCore::VinsCore(Param *param) {
        run_info_ = new RunInfo();
        param_ = param;
        ric_estimator_ = new RICEstimator(param->window_size);
        camera_wrapper_ = new CameraWrapper(param);
        feature_tracker_ = new FeatureTracker(param, camera_wrapper_);
        loop_closer_ = new LoopCloser();
        std::thread([this]() {
            struct timeval tv1{}, tv2{};
            gettimeofday(&tv1, nullptr);
            _handleData();
            gettimeofday(&tv2, nullptr);
            int cost_us = (tv2.tv_sec - tv1.tv_sec) * 1000000 + tv2.tv_usec - tv1.tv_usec;
            if ( cost_us < 1 * 1000) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }).detach();
    }

    void VinsCore::handleIMU(ConstVec3dRef acc, ConstVec3dRef gyr, double time_stamp) {
        Synchronized(read_imu_buf_mutex_) {
            if (vins_state_ == EVinsState::kNoIMUData) {
                run_info_->prev_imu_state.acc = acc;
                run_info_->prev_imu_state.gyr = gyr;
                run_info_->prev_imu_state.time_stamp = time_stamp;
                vins_state_ = EVinsState::kNoImgData;
            } else if (vins_state_ == EVinsState::kNoImgData) {
                run_info_->prev_imu_state.acc = acc;
                run_info_->prev_imu_state.gyr = gyr;
                run_info_->prev_imu_state.time_stamp = time_stamp;
            } else {
                acc_buf_.emplace(acc);
                gyr_buf_.emplace(gyr);
                imu_time_stamp_buf_.emplace(time_stamp);
            }
        }
    }

    static const Eigen::Vector3d zero = Eigen::Vector3d::Zero();

    static KeyFrameState recurseByImu(const KeyFrameState &prev_state, const ImuIntegrator &pre_integral) {
        const Eigen::Vector3d &delta_pos = pre_integral.deltaPos();
        const Eigen::Vector3d &delta_vel = pre_integral.deltaVel();
        const Eigen::Quaterniond &delta_quat = pre_integral.deltaQuat();

        Eigen::Vector3d prev_pos = prev_state.pos;
        Eigen::Vector3d prev_vel = prev_state.vel;
        Eigen::Matrix3d prev_rot = prev_state.rot;
        Eigen::Vector3d prev_ba = prev_state.ba;
        Eigen::Vector3d prev_bg = prev_state.bg;

        Eigen::Vector3d cur_pos = prev_pos + prev_rot * delta_pos;
        Eigen::Vector3d cur_vel = prev_vel + prev_rot * delta_vel;
        Eigen::Matrix3d cur_rot = delta_quat.toRotationMatrix() * prev_rot;

        KeyFrameState cur_state = {cur_pos, cur_rot, cur_vel, prev_ba, prev_bg};
        return cur_state;
    }

    void VinsCore::handleImage(const std::shared_ptr<cv::Mat> &_img, double time_stamp) {
        img_buf_.emplace(_img);
        img_time_stamp_buf_.emplace(time_stamp);
    }

    void VinsCore::_handleData() {
        if (vins_state_ == EVinsState::kNoIMUData) {
            return;
        }

        /******************从缓冲区中读取图像数据*******************/
        if (img_buf_.empty()) {
            return;
        }
        auto img_ptr = img_buf_.front();
        double img_time_stamp = img_time_stamp_buf_.front();
        img_buf_.pop();
        img_time_stamp_buf_.pop();

        /******************提取特征点*******************/
        int prev_kf_window_size = run_info_->kf_state_window.size();
        std::vector<FeaturePoint2D> feature_points = feature_tracker_->extractFeatures(*img_ptr, img_time_stamp);

        /******************首帧图像加入滑动窗口*******************/
        if (vins_state_ == EVinsState::kNoImgData) {
            run_info_->kf_state_window.push_back({});
            run_info_->frame_window.emplace_back(feature_points, nullptr, true);
            FeatureHelper::addFeatures(prev_kf_window_size, img_time_stamp, feature_points, run_info_->feature_window);
            vins_state_ = EVinsState::kEstimateExtrinsic;
            return;
        }

        /******************从缓冲区中读取惯导数据*******************/
        auto frame_pre_integral =
                std::make_shared<ImuIntegrator>(param_->imu_param, run_info_->prev_imu_state, run_info_->gravity);
        run_info_->prev_imu_state.acc = acc_buf_.back();
        run_info_->prev_imu_state.gyr = gyr_buf_.back();
        run_info_->prev_imu_state.time_stamp = imu_time_stamp_buf_.back();
        while (!acc_buf_.empty() && imu_time_stamp_buf_.front() <= img_time_stamp) {
            frame_pre_integral->predict(imu_time_stamp_buf_.front(), acc_buf_.front(), gyr_buf_.front());
            imu_time_stamp_buf_.pop();
            acc_buf_.pop();
            gyr_buf_.pop();
        }

        /******************非首帧图像加入滑动窗口*******************/
        bool is_key_frame = FeatureHelper::isKeyFrame(prev_kf_window_size, feature_points, run_info_->feature_window);
        run_info_->frame_window.emplace_back(feature_points, frame_pre_integral, is_key_frame);
        if (!is_key_frame) {
            return;
        }
        FeatureHelper::addFeatures(prev_kf_window_size, img_time_stamp, feature_points, run_info_->feature_window);
        if (kf_pre_integral_ptr_ == nullptr) {
            kf_pre_integral_ptr_ =
                    std::make_shared<ImuIntegrator>(param_->imu_param, run_info_->prev_imu_state, run_info_->gravity);
        }
        kf_pre_integral_ptr_->jointLaterIntegrator(*frame_pre_integral);
        run_info_->kf_state_window.emplace_back(
                recurseByImu(run_info_->kf_state_window.back(), *kf_pre_integral_ptr_));
        run_info_->pre_int_window.emplace_back(kf_pre_integral_ptr_);
        kf_pre_integral_ptr_ = nullptr;

        /******************扔掉最老的关键帧并边缘化*******************/
        if (run_info_->kf_state_window.size() == param_->window_size + 1) {
            std::unordered_map<int, int> feature_id_2_idx_origin =
                    FeatureHelper::getFeatureId2Index(run_info_->feature_window);
            auto oldest_features_begin = std::remove_if(run_info_->feature_window.begin(),
                                                        run_info_->feature_window.end(), [](const Feature &feature) {
                        return feature.start_kf_window_idx == 0;
                    });
            std::vector<Feature> oldest_feature(oldest_features_begin, run_info_->feature_window.end());
            run_info_->feature_window.erase(oldest_features_begin, run_info_->feature_window.end());
            std::unordered_map<int, int> feature_id_2_idx_after_discard =
                    FeatureHelper::getFeatureId2Index(run_info_->feature_window);
            for (Feature &feature: run_info_->feature_window) {
                feature.start_kf_window_idx--;
            }
            auto it = std::find_if(run_info_->frame_window.begin(), run_info_->frame_window.end(), [&](auto &frame) {
                return frame.time_stamp >= run_info_->kf_state_window.begin()->time_stamp;
            });
            run_info_->frame_window.erase(run_info_->frame_window.begin(), it);
            run_info_->kf_state_window.erase(run_info_->kf_state_window.begin());
            run_info_->pre_int_window.erase(run_info_->pre_int_window.begin());
            if (vins_state_ == EVinsState::kNormal) {
                SlideWindowEstimator::slide(*param_,
                                            oldest_feature,
                                            *run_info_->pre_int_window.front(),
                                            feature_id_2_idx_origin,
                                            feature_id_2_idx_after_discard);
            }
        }

        /******************初始化RIC*******************/
        if (vins_state_ == EVinsState::kEstimateExtrinsic) {
            if (run_info_->frame_window.size() < 2) {
                return;
            }
            int cur_kf_window_size = run_info_->kf_state_window.size();
            PointCorrespondences correspondences = FeatureHelper::getCorrespondences(cur_kf_window_size - 1,
                                                                                     cur_kf_window_size,
                                                                                     run_info_->feature_window);
            Eigen::Quaterniond imu_quat = run_info_->pre_int_window.back()->deltaQuat();
            bool succ = ric_estimator_->estimate(correspondences, imu_quat, run_info_->ric);
            if (!succ) {
                LOG_E("estimate extrinsic false, please rotate rapidly");
                return;
            }
            vins_state_ = EVinsState::kInitial;
            return;
        }

        /******************初始化系统状态、机体坐标系*******************/
        if (vins_state_ == EVinsState::kInitial) {
            if (img_time_stamp - last_init_time_stamp_ < 0.1) {
                return;
            }
            last_init_time_stamp_ = img_time_stamp;
            bool rtn = Initiate::initiate(param_->gravity_norm, *run_info_);
            if (!rtn) {
                return;
            }
            vins_state_ = EVinsState::kNormal;
            return;
        }

        /******************滑窗优化*******************/
        SlideWindowEstimator::optimize(param_->slide_window,
                                       run_info_->pre_int_window,
                                       run_info_->loop_match_infos,
                                       run_info_->feature_window,
                                       run_info_->kf_state_window,
                                       run_info_->tic,
                                       run_info_->ric);
        run_info_->prev_imu_state.ba = run_info_->kf_state_window.back().ba;
        run_info_->prev_imu_state.bg = run_info_->kf_state_window.back().bg;

        /******************错误检测*******************/
        bool fail;
        if (fail) {
            delete run_info_;
            run_info_ = new RunInfo;
            vins_state_ = EVinsState::kInitial;
            return;
        }

        /******************准备回环*******************/
        std::vector<cv::Point3f> key_pts_3d;
        int key_pts_num = run_info_->frame_window.back().points.size();
        for (int i = 0; i < key_pts_num; ++i) {
            cv::Point2f p2d = run_info_->frame_window.back().points[i];
            double depth = FeatureHelper::featureIdToDepth(run_info_->frame_window.back().feature_ids[i],
                                                           run_info_->feature_window);
            key_pts_3d.emplace_back(utils::cvPoint2fToCvPoint3f(p2d, depth));
        }

        const int fast_th = 20; // corner detector response threshold
        std::vector<cv::KeyPoint> external_key_points_un_normalized;
        cv::FAST(*img_ptr, external_key_points_un_normalized, fast_th, true);
        std::vector<cv::Point2f> external_key_pts2d;
        for (const cv::KeyPoint &keypoint: external_key_points_un_normalized) {
            external_key_pts2d.push_back(camera_wrapper_->rawPoint2UniformedPoint(keypoint.pt));
        }

        loop_closer_->addKeyFrame(run_info_->frame_window.back(),
                                  *img_ptr,
                                  key_pts_3d,
                                  external_key_points_un_normalized,
                                  external_key_pts2d);
    }
}