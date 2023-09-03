//
// Created by gjt on 9/3/23.
//

#include "vins_logic.h"
#include "impl/vins_core.h"

namespace vins {
    VinsCore* sp_vins_core;
    void init(vins::Param *param) {
        sp_vins_core = new VinsCore(param);
    }
    void handleImage(const std::shared_ptr<cv::Mat> &_img, double time_stamp) {
        sp_vins_core->handleImage(_img, time_stamp);
    }
    void handleIMU(const Eigen::Vector3d &acc, const Eigen::Vector3d & gyr, double time_stamp) {
        sp_vins_core->handleIMU(acc, gyr, time_stamp);
    }
    void handleDriftCalibration(const Eigen::Vector3d &t_drift, const Eigen::Matrix3d &r_drift) {
        sp_vins_core->handleDriftCalibration(t_drift, r_drift);
    }
}
