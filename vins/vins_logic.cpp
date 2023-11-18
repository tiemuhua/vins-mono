//
// Created by gjt on 9/3/23.
//

#include "vins_logic.h"
#include "impl/vins_controller.h"

namespace vins {
    static VinsController *sp_vins_core;

    void init(const vins::Param& param, const std::shared_ptr<Callback> &cb) {
        sp_vins_core = new VinsController(param, cb);
    }

    const vins::Param &getParam() {
        assert(sp_vins_core != nullptr);
        return sp_vins_core->getParam();
    }

    void handleImage(const std::shared_ptr<cv::Mat> &_img, double time_stamp) {
//        LOG(INFO) << "handleImage ts:" << time_stamp;
        assert(_img->rows == getParam().camera.row && _img->cols == getParam().camera.col);
        sp_vins_core->handleImage(_img, time_stamp);
    }

    void handleIMU(const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr, double time_stamp) {
//        LOG(INFO) << "handleIMU ts:" << time_stamp
//            << ", acc:" << acc.transpose()
//            << ", gyr:" << gyr.transpose();
        sp_vins_core->handleIMU(acc, gyr, time_stamp);
    }

    void handleDriftCalibration(const Eigen::Vector3d &t_drift, const Eigen::Matrix3d &r_drift) {
        sp_vins_core->handleDriftCalibration(t_drift, r_drift);
    }
}
