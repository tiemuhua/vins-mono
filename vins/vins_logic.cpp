//
// Created by gjt on 9/3/23.
//

#include "vins_logic.h"
#include "impl/vins_core.h"

namespace vins {
    static VinsCore *sp_vins_core;

    void init(std::unique_ptr<vins::Param> param, const std::shared_ptr<Callback> &cb) {
        sp_vins_core = new VinsCore(std::move(param), cb);
    }

    vins::Param *getParam() {
        return sp_vins_core->getParam();
    }

    void handleImage(const std::shared_ptr<cv::Mat> &_img, double time_stamp) {
        LOG(INFO) << "handleImage ts:" << time_stamp;
        sp_vins_core->handleImage(_img, time_stamp);
    }

    void handleIMU(const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr, double time_stamp) {
        LOG(INFO) << "handleIMU ts:" << time_stamp;
        sp_vins_core->handleIMU(acc, gyr, time_stamp);
    }

    void handleDriftCalibration(const Eigen::Vector3d &t_drift, const Eigen::Matrix3d &r_drift) {
        sp_vins_core->handleDriftCalibration(t_drift, r_drift);
    }
}
