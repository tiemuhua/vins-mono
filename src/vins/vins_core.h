//
// Created by gjt on 5/14/23.
//

#ifndef VINS_VINS_CORE_H
#define VINS_VINS_CORE_H

#include <vector>
#include <mutex>
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>

namespace vins{
    class Frame;
    class RICEstimator;
    class BriefExtractor;
    class FeatureTracker;
    class LoopCloser;
    class RunInfo;
    class Param;
    class CameraWrapper;
    class ImuIntegrator;

    class VinsCore {
    public:
        VinsCore(Param* param);
        void handleImage(const std::shared_ptr<cv::Mat> &_img, double time_stamp);
        void handleIMU(const Eigen::Vector3d &acc, const Eigen::Vector3d & gyr, double time_stamp);

    private:
        void _handleData();

    private:
        enum class EVinsState : int {
            kNoIMUData,             // 尚未收到IMU数据
            kNoImgData,             // 尚未收到图片数据
            kEstimateExtrinsic,     // 估计相机外参
            kInitial,               // 初始化
            kNormal,                // 正常优化
        } vins_state_ = EVinsState::kNoIMUData;

        std::mutex read_imu_buf_mutex_;
        std::shared_ptr<ImuIntegrator> kf_pre_integral_ptr_;

        double last_init_time_stamp_ = 0.0;

        //.数据缓冲区.
        std::queue<std::shared_ptr<cv::Mat>> img_buf_;
        std::queue<double> img_time_stamp_buf_;
        std::queue<Eigen::Vector3d> acc_buf_;
        std::queue<Eigen::Vector3d> gyr_buf_;
        std::queue<double> imu_time_stamp_buf_;

        //.子求解器.
        RICEstimator *ric_estimator_;
        FeatureTracker *feature_tracker_;
        CameraWrapper *camera_wrapper_;
        LoopCloser *loop_closer_;

        RunInfo* run_info_;
        Param* param_;
    };
}


#endif //VINS_VINS_CORE_H
