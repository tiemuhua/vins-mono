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

    class VinsCore {
    public:
        VinsCore(Param* param);
        void handleImage(const cv::Mat &_img, double time_stamp);
        void handleIMU(const Eigen::Vector3d &acc, const Eigen::Vector3d & gyr, double time_stamp);

    private:
        enum EVinsState {
            kVinsStateEstimateExtrinsic,    // 估计相机外参
            kVinsStateInitial,              // 初始化
            kVinsStateNormal,               // 正常优化
        } vins_state_ = kVinsStateEstimateExtrinsic;
        EVinsState _handleEstimateExtrinsic();
        EVinsState _handleInitial(double time_stamp);
        EVinsState _handleNormal(const cv::Mat &_img, bool is_key_frame);

    private:
        std::mutex read_imu_buf_mutex_;
        std::queue<Eigen::Vector3d> acc_buf_;
        std::queue<Eigen::Vector3d> gyr_buf_;
        std::queue<double> time_stamp_buf_;

        double last_init_time_stamp_ = 0.0;

        int cur_frame_id_;

        RICEstimator *ric_estimator_;
        FeatureTracker *feature_tracker_;
        CameraWrapper *camera_wrapper_;
        LoopCloser *loop_closer_;

        RunInfo* run_info_;
        Param* param_;
    };
}


#endif //VINS_VINS_CORE_H
