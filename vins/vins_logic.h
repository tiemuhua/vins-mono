//
// Created by gjt on 9/3/23.
//

#ifndef GJT_VINS_VINS_LOGIC_H
#define GJT_VINS_VINS_LOGIC_H

#include <vector>
#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>
#include "param.h"

namespace vins {

    struct PosAndTimeStamp{
        double time_stamp;
        Eigen::Vector3d pos;
        Eigen::Matrix3d rot;
    };
    class Callback {
    public:
        virtual void onPosSolved(const std::vector<PosAndTimeStamp> & pos_and_time_stamps) = 0;
    };

    void init(const vins::Param& param, const std::shared_ptr<Callback> &cb);
    const vins::Param &getParam();
    void handleImage(const std::shared_ptr<cv::Mat> &_img, double time_stamp);
    void handleIMU(const Eigen::Vector3d &acc, const Eigen::Vector3d & gyr, double time_stamp);
    void handleDriftCalibration(const Eigen::Vector3d &t_drift, const Eigen::Matrix3d &r_drift);
}

#endif //GJT_VINS_VINS_LOGIC_H
