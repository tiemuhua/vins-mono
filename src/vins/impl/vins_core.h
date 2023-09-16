//
// Created by gjt on 5/14/23.
//

#ifndef VINS_VINS_CORE_H
#define VINS_VINS_CORE_H

#include <vector>
#include <queue>
#include <mutex>

#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>

#include "feature_tracker.h"
#include "imu_integrator.h"
#include "vins_define_internal.h"
#include "vins_logic.h"

namespace vins {
    class Frame;

    class BriefExtractor;

    class FeatureTracker;

    class LoopCloser;

    class RunInfo;

    class Param;

    class CameraWrapper;

    class ImuIntegrator;

    class VinsCore {
    public:
        explicit VinsCore(Param *param);

        void handleImage(const std::shared_ptr<cv::Mat> &_img, double time_stamp);

        void handleIMU(const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr, double time_stamp);

        void handleDriftCalibration(const Eigen::Vector3d &t_drift, const Eigen::Matrix3d &r_drift);

        vins::Param *getParam() { return param_; }

    private:
        void _handleData();

    private:
        enum class EVinsState : int {
            kNoIMUData,             // 尚未收到IMU数据
            kNoImgData,             // 尚未收到图片数据
            kInitial,               // 初始化
            kNormal,                // 正常优化
        } vins_state_ = EVinsState::kNoIMUData;

        //.数据缓冲区.
        std::queue<std::shared_ptr<cv::Mat>> img_buf_;
        std::queue<double> img_time_stamp_buf_;
        std::queue<Eigen::Vector3d> acc_buf_;
        std::queue<Eigen::Vector3d> gyr_buf_;
        std::queue<double> imu_time_stamp_buf_;
        Eigen::Vector3d t_drift_;
        Eigen::Matrix3d r_drift_;
        std::mutex io_mutex_;


        //.子求解器.
        FeatureTracker *feature_tracker_ = nullptr;
        CameraWrapper *camera_wrapper_ = nullptr;
        LoopCloser *loop_closer_ = nullptr;
        BriefExtractor *brief_extractor_ = nullptr;

        //.运行时信息.
        RunInfo *run_info_ = nullptr;
        Param *param_ = nullptr;
        std::shared_ptr<ImuIntegrator> kf_pre_integral_ptr_ = nullptr;
        double last_init_time_stamp_ = 0.0;
    };
}


#endif //VINS_VINS_CORE_H
