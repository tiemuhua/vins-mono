//
// Created by gjt on 5/14/23.
//
#include <sys/time.h>

#include "vins_controller.h"
#include "vins_utils.h"
#include "vins_model.h"
#include "initial/initiate.h"
#include "feature_tracker.h"
#include "feature_helper.h"
#include "front_end_optimize/front_end_optimize.h"
#include "loop_closer/loop_closer.h"
#include "camera_wrapper.h"

namespace vins {
    VinsController::VinsController(const Param& param, std::weak_ptr<Callback> cb) {
        param_ = param;
        camera_wrapper_ = new CameraWrapper(param);
        cb_ = std::move(cb);
        brief_extractor_ = new DVision::BRIEF();
        brief_extractor_->importPairs(param.brief_param.x1,
                                      param.brief_param.y1,
                                      param.brief_param.x2,
                                      param.brief_param.y2);

        std::thread([this]() {
            _handleData();
        }).detach();
    }

    // 外部IO线程调用
    void VinsController::handleIMU(const Eigen::Vector3d& acc, const Eigen::Vector3d& gyr, double time_stamp) {
        Synchronized(io_mutex_) {
            acc_buf_.emplace(acc);
            gyr_buf_.emplace(gyr);
            imu_time_buf_.emplace(time_stamp);
        }
    }

    // 外部IO线程调用
    void VinsController::handleImage(const std::shared_ptr<cv::Mat> &_img, double time_stamp) {
        Synchronized(io_mutex_) {
            img_buf_.emplace(_img);
            img_time_stamp_buf_.emplace(time_stamp);
        }
    }

    // 回环线程调用
    void VinsController::handleDriftCalibration(const Eigen::Vector3d &t_drift, const Eigen::Matrix3d &r_drift) {
        Synchronized(io_mutex_) {
            t_drift_ = t_drift;
            r_drift_ = r_drift;
        }
    }

    // _handleData调用
    static KeyFrameState _recurseByImu(const KeyFrameState &prev_state, const ImuIntegral &imu_integral) {
        const Eigen::Vector3d &delta_pos = imu_integral.deltaPos();
        const Eigen::Vector3d &delta_vel = imu_integral.deltaVel();
        const Eigen::Quaterniond &delta_quat = imu_integral.deltaQuat();

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

    // vins工作线程调用
    constexpr int wait_data_interval_us = 100;
    [[noreturn]] void VinsController::_handleData(){
        while(true) {
            RawFrameData raw_frame_data;
            bool has_data = false;
            Synchronized(io_mutex_) {
                has_data = readRawFrameDataUnsafe(raw_frame_data);
            }
            if (has_data) {
                _handleDataImpl(std::move(raw_frame_data));
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds (wait_data_interval_us));
            }
        }
    }

    /**
     * @brief 从缓冲区中读取数据
     * @return 若读到数据需要处理则返回true，否则返回false
     * */
    bool VinsController::readRawFrameDataUnsafe(RawFrameData& raw_frame_data) {
        // 首次循环
        if (vins_state_ == EVinsState::kNoIMUData) {
            if (acc_buf_.empty()) {
                return false;
            }
            vins_state_ = EVinsState::kNoImgData;
            vins_model_.prev_imu_state.acc = acc_buf_.back();
            vins_model_.prev_imu_state.gyr = gyr_buf_.back();
            vins_model_.prev_imu_state.time = imu_time_buf_.back();
            acc_buf_ = std::queue<Eigen::Vector3d>();
            gyr_buf_ = std::queue<Eigen::Vector3d>();
            imu_time_buf_ = std::queue<double>();
            img_buf_ = std::queue<std::shared_ptr<cv::Mat>>();
            img_time_stamp_buf_ = std::queue<double>();
            return false;
        }

        // 读取图像数据
        if (img_buf_.empty()) {
            return false;
        }
        raw_frame_data.img_time_stamp_ms = img_time_stamp_buf_.front();
        raw_frame_data.img = img_buf_.front();
        img_buf_.pop();
        img_time_stamp_buf_.pop();

        // 首帧特殊处理
        if (vins_state_ == EVinsState::kNoImgData) {
            while (!acc_buf_.empty() && imu_time_buf_.front() <= raw_frame_data.img_time_stamp_ms) {
                vins_model_.prev_imu_state.acc = acc_buf_.front();
                vins_model_.prev_imu_state.gyr = gyr_buf_.front();
                vins_model_.prev_imu_state.time = imu_time_buf_.front();
                imu_time_buf_.pop();
                acc_buf_.pop();
                gyr_buf_.pop();
            }
            vins_state_ = EVinsState::kInitial;
            return true;
        }

        // 非首帧正常读取IMU数据
        raw_frame_data.imu_integral = std::make_unique<ImuIntegral>(param_.imu_param,
                                                                    vins_model_.prev_imu_state,
                                                                    vins_model_.gravity);
        while (!acc_buf_.empty() && imu_time_buf_.front() <= raw_frame_data.img_time_stamp_ms) {
            vins_model_.prev_imu_state.acc = acc_buf_.front();
            vins_model_.prev_imu_state.gyr = gyr_buf_.front();
            vins_model_.prev_imu_state.time = imu_time_buf_.front();
            raw_frame_data.imu_integral->predict(imu_time_buf_.front(),
                                                 acc_buf_.front(),
                                                 gyr_buf_.front());
            imu_time_buf_.pop();
            acc_buf_.pop();
            gyr_buf_.pop();
        }
        return true;
    }

    // vins工作线程调用
    void VinsController::_handleDataImpl(RawFrameData raw_frame_sensor_data) {
        /******************提取特征点*******************/
        std::vector<FeaturePoint2D> feature_pts;
        std::vector<cv::KeyPoint> feature_raw_pts;
        FeatureTracker::extractFeatures(raw_frame_sensor_data.img,
                                        raw_frame_sensor_data.img_time_stamp_ms,
                                        *camera_wrapper_,
                                        param_.frame_tracker,
                                        feature_pts,
                                        feature_raw_pts,
                                        vins_model_.prev_img_feature_info);

        // 首帧不存在对应的IMU数据，需要特殊处理。
        if (raw_frame_sensor_data.imu_integral == nullptr) {
            vins_model_.frame_window.emplace_back(feature_pts,
                                                  nullptr,
                                                  true,
                                                  raw_frame_sensor_data.img_time_stamp_ms);
            FeatureHelper::addFeatures(vins_model_.kf_state_window.size(),
                                       raw_frame_sensor_data.img_time_stamp_ms,
                                       feature_pts,
                                       vins_model_.feature_window);
            // addFeatures用到了kf_state_window.size，顺序不能反
            vins_model_.kf_state_window.emplace_back(KeyFrameState());
            return;
        }

        // 当前帧IMU预积分汇入当前关键帧IMU预积分
        if (vins_model_.kf_imu_integral == nullptr) {
            vins_model_.kf_imu_integral = std::make_unique<ImuIntegral>(param_.imu_param,
                                                                        vins_model_.prev_imu_state,
                                                                        vins_model_.gravity);
        }
        vins_model_.kf_imu_integral->jointLaterIntegrator(*raw_frame_sensor_data.imu_integral);

        /******************当前帧加入滑动窗口*******************/
        bool is_key_frame = vins_model_.kf_state_window.size() < 2 ||
                            FeatureHelper::isKeyFrame(param_.key_frame_parallax_threshold,
                                                      feature_pts,
                                                      vins_model_.feature_window);
        LOG(INFO) << "input feature: " << feature_pts.size() << "\t"
                  << "num of feature: " << vins_model_.feature_window.size() << "\t"
                  << "is key frame: " << (is_key_frame ? "true" : "false");
        vins_model_.frame_window.emplace_back(feature_pts,
                                               std::move(raw_frame_sensor_data.imu_integral),
                                               is_key_frame,
                                              raw_frame_sensor_data.img_time_stamp_ms);
        if (!is_key_frame) {
            return;
        }
        FeatureHelper::addFeatures(vins_model_.kf_state_window.size(),
                                   raw_frame_sensor_data.img_time_stamp_ms,
                                   feature_pts,
                                   vins_model_.feature_window);
        KeyFrameState kf_state = _recurseByImu(vins_model_.kf_state_window.back(), *vins_model_.kf_imu_integral);
        kf_state.time_stamp = raw_frame_sensor_data.img_time_stamp_ms;
        // addFeatures用到了kf_state_window.size，顺序不能反
        vins_model_.kf_state_window.emplace_back(kf_state);
        vins_model_.pre_int_window.emplace_back(std::move(vins_model_.kf_imu_integral));
        vins_model_.kf_imu_integral = nullptr;

        /******************滑动窗口塞满后再进行后续操作*******************/
        if (vins_model_.kf_state_window.size() < param_.window_size) {
            return;
        }

        /******************扔掉最老的关键帧并边缘化*******************/
        if (vins_model_.kf_state_window.size() == param_.window_size + 1) {
            // 边缘化所涉及的特征点
            std::vector<Feature> marginal_features;
            for (const Feature &feature: vins_model_.feature_window) {
                if (feature.start_kf_window_idx == 0) {
                    marginal_features.emplace_back(feature);
                }
            }
            LOG(INFO) << "margin features size:" << marginal_features.size();

            // 删掉即将溜出滑动窗口的帧所对应的特征点
            for (Feature &feature: vins_model_.feature_window) {
                if (feature.start_kf_window_idx == 0) {
                    feature.points.erase(feature.points.begin());
                    feature.velocities.erase(feature.velocities.begin());
                } else {
                    feature.start_kf_window_idx--;
                }
                // feature.points.size() == 1的时候就被删了
                assert(!feature.points.empty());
            }

            // 可以三角化的特征必然包括两个以上的特征点，删除所有无法三角化的特征
            // 并记录删除特征前后的feature_id_2_idx
            LOG(INFO) << "features size before discard:" << vins_model_.feature_window.size();
            std::unordered_map<int, int> feature_id_2_idx_before_discard =
                    FeatureHelper::getFeatureId2Index(vins_model_.feature_window);
            utils::erase_if_wrapper(vins_model_.feature_window, [&](const Feature &feature) -> bool {
                // 后继无帧，该特征要溜出滑动窗口了
                bool is_too_old = feature.start_kf_window_idx == 0 && feature.points.size() == 1;
                // 相邻帧中没有出现该特征，无法三角化，大概率是计算误差导致的离群点
                bool is_outline_feature =
                        feature.start_kf_window_idx != param_.window_size - 1 && feature.points.size() == 1;
                return is_too_old || is_outline_feature;
            });
            std::unordered_map<int, int> feature_id_2_idx_after_discard =
                    FeatureHelper::getFeatureId2Index(vins_model_.feature_window);
            LOG(INFO) << "features size after discard:" << vins_model_.feature_window.size();

            // 边缘化
            if (vins_state_ == EVinsState::kNormal) {
                FrontEndOptimize::slide(param_,
                                        marginal_features,
                                        *vins_model_.pre_int_window.front(),
                                        feature_id_2_idx_before_discard,
                                        feature_id_2_idx_after_discard);
            }

            // 移除滑动窗口中过期的帧、关键帧状态、关键帧IMU积分、回环匹配
            LOG(INFO) << "frame_window size before discard:" << vins_model_.frame_window.size();
            utils::erase_if_wrapper(vins_model_.frame_window, [&](const Frame& frame) ->bool {
                return frame.time_stamp < vins_model_.kf_state_window.begin()->time_stamp;
            });
            LOG(INFO) << "frame_window size before discard:" << vins_model_.frame_window.size();
            vins_model_.kf_state_window.erase(vins_model_.kf_state_window.begin());
            vins_model_.pre_int_window.erase(vins_model_.pre_int_window.begin());
            for (LoopMatchInfo &info: vins_model_.loop_match_infos) {
                info.window_idx--;
            }
            if (!vins_model_.loop_match_infos.empty() && vins_model_.loop_match_infos[0].window_idx == -1) {
                vins_model_.loop_match_infos.erase(vins_model_.loop_match_infos.begin());
            }
        }

        /******************初始化系统状态、机体坐标系*******************/
        if (vins_state_ == EVinsState::kInitial) {
            if (raw_frame_sensor_data.img_time_stamp_ms - vins_model_.last_init_time < 0.1) {
                return;
            }
            vins_model_.last_init_time = raw_frame_sensor_data.img_time_stamp_ms;
            bool rtn = Initiate::initiate(vins_model_);
            if (!rtn) {
                return;
            }
            vins_state_ = EVinsState::kNormal;
            return;
        }

        /******************滑窗优化*******************/
        FrontEndOptimize::optimize(param_.slide_window,
                                   vins_model_.pre_int_window,
                                   vins_model_.loop_match_infos,
                                   vins_model_.feature_window,
                                   vins_model_.kf_state_window,
                                   vins_model_.tic,
                                   vins_model_.ric);
        vins_model_.prev_imu_state.ba = vins_model_.kf_state_window.back().ba;
        vins_model_.prev_imu_state.bg = vins_model_.kf_state_window.back().bg;

        /******************错误检测*******************/
        bool fail = vins_model_.prev_imu_state.ba.norm() > 1e2 || vins_model_.prev_imu_state.bg.norm() > 1e1;
        if (fail) {
            vins_model_ = VinsModel();
            vins_state_ = EVinsState::kInitial;
            auto cb = cb_.lock();
            if (cb) {
                cb->onFail();
            }
            return;
        }

        /******************寻找回环，加入滑动窗口，加入后端优化队列*******************/
        // 额外的特征点
        const int fast_th = 20; // corner detector response threshold
        std::vector<cv::KeyPoint> external_raw_pts;
        cv::FAST(*raw_frame_sensor_data.img, external_raw_pts, fast_th, true);

        // 描述符
        std::vector<DVision::BRIEF::bitset> descriptors;
        brief_extractor_->compute(*raw_frame_sensor_data.img, feature_raw_pts, descriptors);
        std::vector<DVision::BRIEF::bitset> external_descriptors;
        brief_extractor_->compute(*raw_frame_sensor_data.img, external_raw_pts, external_descriptors);

        //.特征点三维坐标.
        std::vector<cv::Point3f> key_pts_3d;
        size_t key_pts_num = vins_model_.frame_window.back().points.size();
        for (int i = 0; i < key_pts_num; ++i) {
            int feature_id = vins_model_.frame_window.back().feature_ids[i];
            std::unordered_map<int, int> feature_id_2_idx = FeatureHelper::getFeatureId2Index(
                    vins_model_.feature_window);
            cv::Point2f p2d = vins_model_.feature_window[feature_id_2_idx[feature_id]].points[0];
            double depth = FeatureHelper::featureIdToDepth(vins_model_.frame_window.back().feature_ids[i],
                                                           vins_model_.feature_window);
            key_pts_3d.emplace_back(utils::cvPoint2fToCvPoint3f(p2d, depth));
        }

        KeyFrameUniPtr cur_kf = std::make_unique<KeyFrame>(vins_model_.frame_window.back(),
                                                           key_pts_3d,
                                                           descriptors,
                                                           external_descriptors);
        LoopMatchInfo info;
        info.window_idx = vins_model_.kf_state_window.size() - 1;
        if (loop_closer_.findLoop(*cur_kf, info)) {
            vins_model_.loop_match_infos.emplace_back(std::move(info));
        }
        loop_closer_.addKeyFrame(std::move(cur_kf));

        /******************后端线程算出结果后，前端线程相应校正*******************/
        Synchronized(io_mutex_) {
            if (r_drift_.norm() < 0.001) {
                return;
            }
            for (KeyFrameState &state: vins_model_.kf_state_window) {
                state.pos = r_drift_ * state.pos + t_drift_;
                state.rot = r_drift_ * state.rot;
            }
            r_drift_ = Eigen::Matrix3d::Zero();
        }

        std::shared_ptr<Callback> cb = cb_.lock();
        if (cb) {
            std::vector<PosAndTimeStamp> pos_and_time_stamps;
            for (const KeyFrameState &state:vins_model_.kf_state_window) {
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
