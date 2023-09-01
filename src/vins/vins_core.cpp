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

namespace vins {
    VinsCore::VinsCore(Param *param) {
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

    void VinsCore::handleImage(const cv::Mat &_img, double time_stamp) {
        if (vins_state_ == EVinsState::kNoIMUData) {
            return;
        }

        int prev_kf_window_size = run_info_->kf_state_window.size();
        std::vector<FeaturePoint2D> feature_points = feature_tracker_->extractFeatures(_img, time_stamp);
        bool is_key_frame = FeatureHelper::isKeyFrame(prev_kf_window_size, feature_points, run_info_->feature_window);
        run_info_->frame_window.emplace_back(feature_points, cur_pre_integral_ptr_, is_key_frame);

        if (vins_state_ == EVinsState::kNoImgData) {
            FeatureHelper::addFeatures(prev_kf_window_size, time_stamp, feature_points, run_info_->feature_window);
            run_info_->kf_state_window.push_back({});
            cur_pre_integral_ptr_ = std::make_shared<ImuIntegrator>(param_->imu_param, s_prev_imu_input, zero);

            vins_state_ = EVinsState::kEstimateExtrinsic;
            return;
        }

        if (is_key_frame) {
            FeatureHelper::addFeatures(prev_kf_window_size, time_stamp, std::move(feature_points),
                                       run_info_->feature_window);
            run_info_->kf_state_window.emplace_back(
                    recurseByImu(run_info_->kf_state_window.back(), *cur_pre_integral_ptr_));
            run_info_->pre_int_window.emplace_back(*cur_pre_integral_ptr_);
            cur_pre_integral_ptr_ = std::make_shared<ImuIntegrator>(param_->imu_param, s_prev_imu_input, zero);
        }

        /******************扔掉最老的关键帧并边缘化*******************/
        std::unordered_map<int, int> feature_id_2_idx_origin =
                FeatureHelper::getFeatureId2Index(run_info_->feature_window);
        auto oldest_features_begin = std::remove_if(run_info_->feature_window.begin(),
                                                    run_info_->feature_window.end(), [](const Feature &feature) {
                    return feature.start_kf_idx == 0;
                });
        std::vector<Feature> oldest_feature(oldest_features_begin, run_info_->feature_window.end());
        run_info_->feature_window.erase(oldest_features_begin, run_info_->feature_window.end());
        std::unordered_map<int, int> feature_id_2_idx_after_discard =
                FeatureHelper::getFeatureId2Index(run_info_->feature_window);
        for (Feature &feature: run_info_->feature_window) {
            feature.start_kf_idx--;
        }
        run_info_->frame_window.erase(
                run_info_->frame_window.begin(),
                std::find_if(run_info_->frame_window.begin(), run_info_->frame_window.end(), [&](const Frame &frame) {
                    return frame.time_stamp > run_info_->kf_state_window.begin()->time_stamp;
                }));
        run_info_->kf_state_window.erase(run_info_->kf_state_window.begin());
        run_info_->pre_int_window.erase(run_info_->pre_int_window.begin());
        if (vins_state_ == EVinsState::kNormal) {
            SlideWindowEstimator::slide(*param_,
                                        oldest_feature,
                                        run_info_->pre_int_window.front(),
                                        feature_id_2_idx_origin,
                                        feature_id_2_idx_after_discard);
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
            Eigen::Quaterniond imu_quat = run_info_->pre_int_window.back().deltaQuat();
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
            if (time_stamp - last_init_time_stamp_ < 0.1) {
                return;
            }
            last_init_time_stamp_ = time_stamp;
            bool rtn = Initiate::initiate(param_->gravity_norm, *run_info_);
            if (!rtn) {
                return;
            }
            vins_state_ = EVinsState::kNormal;
            return;
        }

        /******************滑窗优化*******************/
        SlideWindowEstimator::optimize(param_->slide_window,
                                       run_info_->feature_window,
                                       run_info_->kf_state_window,
                                       run_info_->pre_int_window,
                                       run_info_->tic,
                                       run_info_->ric);

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
        cv::FAST(_img, external_key_points_un_normalized, fast_th, true);
        std::vector<cv::Point2f> external_key_pts2d;
        for (const cv::KeyPoint &keypoint: external_key_points_un_normalized) {
            external_key_pts2d.push_back(camera_wrapper_->rawPoint2UniformedPoint(keypoint.pt));
        }

        loop_closer_->addKeyFrame(run_info_->frame_window.back(),
                                  _img,
                                  key_pts_3d,
                                  external_key_points_un_normalized,
                                  external_key_pts2d);
        // todo SlideWindowEstimator::setLoopMatchInfo
    }
}