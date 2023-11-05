//
// Created by gjt on 5/14/23.
//
#include <sys/time.h>

#include "vins_core.h"
#include "vins_utils.h"
#include "vins_define_internal.h"
#include "initial/initiate.h"
#include "feature_tracker.h"
#include "feature_helper.h"
#include "impl/front_end_optimize/front_end_optimize.h"
#include "loop_closer/loop_closer.h"
#include "camera_wrapper.h"

namespace vins {
    VinsCore::VinsCore(const Param& param, std::weak_ptr<Callback> cb) {
        run_info_ = new RunInfo();
        param_ = param;
        camera_wrapper_ = new CameraWrapper(param);
        feature_tracker_ = new FeatureTracker(param, camera_wrapper_);
        cb_ = std::move(cb);
        std::thread([this]() {
            while(true) {
                struct timeval tv1{}, tv2{};
                gettimeofday(&tv1, nullptr);
                if (vins_state_ != EVinsState::kNoIMUData) {
                    _handleData();
                }
                gettimeofday(&tv2, nullptr);
                int cost_us = (tv2.tv_sec - tv1.tv_sec) * 1000000 + tv2.tv_usec - tv1.tv_usec;
                if (cost_us < 1 * 1000) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
            }
        }).detach();
    }

    void VinsCore::handleIMU(const Eigen::Vector3d& acc, const Eigen::Vector3d& gyr, double time_stamp) {
        Synchronized(io_mutex_) {
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

    void VinsCore::handleImage(const std::shared_ptr<cv::Mat> &_img, double time_stamp) {
        Synchronized(io_mutex_) {
            img_buf_.emplace(_img);
            img_time_stamp_buf_.emplace(time_stamp);
        }
    }

    void VinsCore::handleDriftCalibration(const Eigen::Vector3d &t_drift, const Eigen::Matrix3d &r_drift) {
        Synchronized(io_mutex_) {
            t_drift_ = t_drift;
            r_drift_ = r_drift;
        }
    }

    static KeyFrameState recurseByImu(const KeyFrameState &prev_state, const ImuIntegrator &pre_integral) {
        const Eigen::Vector3d &delta_pos = pre_integral.deltaPos();
        const Eigen::Vector3d &delta_vel = pre_integral.deltaVel();
        const Eigen::Quaterniond &delta_quat = pre_integral.deltaQuat();

        const Eigen::Vector3d &prev_pos = prev_state.pos;
        const Eigen::Vector3d &prev_vel = prev_state.vel;
        const Eigen::Matrix3d &prev_rot = prev_state.rot;
        const Eigen::Vector3d &prev_ba = prev_state.ba;
        const Eigen::Vector3d &prev_bg = prev_state.bg;

        const Eigen::Vector3d cur_pos = prev_pos + prev_rot * delta_pos;
        const Eigen::Vector3d cur_vel = prev_vel + prev_rot * delta_vel;
        const Eigen::Matrix3d cur_rot = delta_quat.toRotationMatrix() * prev_rot;

        KeyFrameState cur_state = {cur_pos, cur_rot, cur_vel, prev_ba, prev_bg};
        return cur_state;
    }

    void VinsCore::_handleData() {
        /******************从缓冲区中读取图像数据*******************/
        double img_time_stamp = -1;
        std::shared_ptr<cv::Mat> img_ptr = nullptr;
        Synchronized(io_mutex_) {
            if (img_buf_.empty()) {
                return;
            }
            img_ptr = img_buf_.front();
            img_time_stamp = img_time_stamp_buf_.front();
            img_buf_.pop();
            img_time_stamp_buf_.pop();
        }

        /******************提取特征点*******************/
        std::vector<FeaturePoint2D> feature_pts;
        std::vector<cv::KeyPoint> feature_raw_pts;
        feature_tracker_->extractFeatures(*img_ptr, img_time_stamp, feature_pts, feature_raw_pts);

        /******************首帧图像加入滑动窗口*******************/
        if (vins_state_ == EVinsState::kNoImgData) {
            run_info_->kf_state_window.push_back({});
            run_info_->frame_window.emplace_back(feature_pts, nullptr, true);
            FeatureHelper::addFeatures(0, img_time_stamp, feature_pts, run_info_->feature_window);
            vins_state_ = EVinsState::kInitial;
            return;
        }

        /******************从缓冲区中读取惯导数据*******************/
        auto frame_pre_integral =
                std::make_shared<ImuIntegrator>(param_.imu_param, run_info_->prev_imu_state, run_info_->gravity);
        Synchronized(io_mutex_) {
            while (!acc_buf_.empty() && imu_time_stamp_buf_.front() <= img_time_stamp) {
                run_info_->prev_imu_state.acc = acc_buf_.front();
                run_info_->prev_imu_state.gyr = gyr_buf_.front();
                run_info_->prev_imu_state.time_stamp = imu_time_stamp_buf_.front();
                frame_pre_integral->predict(imu_time_stamp_buf_.front(), acc_buf_.front(), gyr_buf_.front());
                imu_time_stamp_buf_.pop();
                acc_buf_.pop();
                gyr_buf_.pop();
            }
        }

        /******************非首帧图像加入滑动窗口*******************/
        int prev_kf_window_size = run_info_->kf_state_window.size();
        bool is_key_frame = FeatureHelper::isKeyFrame(param_.key_frame_parallax_threshold,
                                                      prev_kf_window_size,
                                                      feature_pts,
                                                      run_info_->feature_window);
        LOG(INFO) << "input feature: " << feature_pts.size() << "\t"
                  << "num of feature: " << run_info_->feature_window.size() << "\t"
                  << "is key frame: " << (is_key_frame ? "true" : "false");

        run_info_->frame_window.emplace_back(feature_pts, frame_pre_integral, is_key_frame);
        if (!is_key_frame) {
            return;
        }
        FeatureHelper::addFeatures(prev_kf_window_size, img_time_stamp, feature_pts, run_info_->feature_window);
        if (kf_pre_integral_ptr_ == nullptr) {
            kf_pre_integral_ptr_ =
                    std::make_unique<ImuIntegrator>(param_.imu_param, run_info_->prev_imu_state, run_info_->gravity);
        }
        kf_pre_integral_ptr_->jointLaterIntegrator(*frame_pre_integral);
        run_info_->kf_state_window.emplace_back(
                recurseByImu(run_info_->kf_state_window.back(), *kf_pre_integral_ptr_));
        run_info_->pre_int_window.emplace_back(std::move(kf_pre_integral_ptr_));
        kf_pre_integral_ptr_ = nullptr;

        /******************滑动窗口塞满后再进行后续操作*******************/
        if (run_info_->kf_state_window.size() < param_.window_size) {
            return;
        }

        /******************扔掉最老的关键帧并边缘化*******************/
        if (run_info_->kf_state_window.size() == param_.window_size + 1) {
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
            for (LoopMatchInfo &info: run_info_->loop_match_infos) {
                info.window_idx--;
            }
            if (!run_info_->loop_match_infos.empty() && run_info_->loop_match_infos[0].window_idx == -1) {
                run_info_->loop_match_infos.erase(run_info_->loop_match_infos.begin());
            }
            if (vins_state_ == EVinsState::kNormal) {
                FrontEndOptimize::slide(param_,
                                        oldest_feature,
                                        *run_info_->pre_int_window.front(),
                                        feature_id_2_idx_origin,
                                        feature_id_2_idx_after_discard);
            }
        }

        /******************初始化系统状态、机体坐标系*******************/
        if (vins_state_ == EVinsState::kInitial) {
            if (img_time_stamp - last_init_time_stamp_ < 0.1) {
                return;
            }
            last_init_time_stamp_ = img_time_stamp;
            bool rtn = Initiate::initiate(*run_info_);
            if (!rtn) {
                return;
            }
            vins_state_ = EVinsState::kNormal;
            return;
        }

        /******************滑窗优化*******************/
        FrontEndOptimize::optimize(param_.slide_window,
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

        /******************寻找回环，加入滑动窗口，加入后端优化队列*******************/
        // 额外的特征点
        const int fast_th = 20; // corner detector response threshold
        std::vector<cv::KeyPoint> external_raw_pts;
        cv::FAST(*img_ptr, external_raw_pts, fast_th, true);

        // 描述符
        std::vector<DVision::BRIEF::bitset> descriptors;
        brief_extractor_->brief_.compute(*img_ptr, feature_raw_pts, descriptors);
        std::vector<DVision::BRIEF::bitset> external_descriptors;
        brief_extractor_->brief_.compute(*img_ptr, external_raw_pts, external_descriptors);

        //.特征点三维坐标.
        std::vector<cv::Point3f> key_pts_3d;
        size_t key_pts_num = run_info_->frame_window.back().points.size();
        for (int i = 0; i < key_pts_num; ++i) {
            int feature_id = run_info_->frame_window.back().feature_ids[i];
            std::unordered_map<int, int> feature_id_2_idx = FeatureHelper::getFeatureId2Index(
                    run_info_->feature_window);
            cv::Point2f p2d = run_info_->feature_window[feature_id_2_idx[feature_id]].points[0];
            double depth = FeatureHelper::featureIdToDepth(run_info_->frame_window.back().feature_ids[i],
                                                           run_info_->feature_window);
            key_pts_3d.emplace_back(utils::cvPoint2fToCvPoint3f(p2d, depth));
        }

        auto cur_kf = std::make_shared<KeyFrame>(run_info_->frame_window.back(), key_pts_3d, descriptors,
                                                 external_descriptors);
        LoopMatchInfo info;
        info.window_idx = run_info_->kf_state_window.size() - 1;
        if (loop_closer_.findLoop(cur_kf, info)) {
            run_info_->loop_match_infos.emplace_back(std::move(info));
        }
        loop_closer_.addKeyFrame(cur_kf);

        /******************后端线程算出结果后，前端线程相应校正*******************/
        Synchronized(io_mutex_) {
            if (r_drift_.norm() < 0.001) {
                return;
            }
            for (KeyFrameState &state: run_info_->kf_state_window) {
                state.pos = r_drift_ * state.pos + t_drift_;
                state.rot = r_drift_ * state.rot;
            }
            r_drift_ = Eigen::Matrix3d::Zero();
        }

        std::shared_ptr<Callback> cb = cb_.lock();
        if (cb) {
            std::vector<PosAndTimeStamp> pos_and_time_stamps;
            for (const KeyFrameState &state:run_info_->kf_state_window) {
                PosAndTimeStamp pos_and_time_stamp;
                pos_and_time_stamp.time_stamp = state.time_stamp;
                pos_and_time_stamp.pos = state.pos;
                pos_and_time_stamp.rot = state.rot;
                pos_and_time_stamps.emplace_back(pos_and_time_stamp);
            }
            cb->onPosSolved(pos_and_time_stamps);
        }
    }
}
