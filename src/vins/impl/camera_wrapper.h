//
// Created by gjt on 8/29/23.
//

#ifndef GJT_VINS_CAMERA_WRAPPER_H
#define GJT_VINS_CAMERA_WRAPPER_H

#include "opencv2/opencv.hpp"
#include <Eigen/Eigen>

#include "camodocal/camera_models/CameraFactory.h"
#include "vins/impl/param.h"

namespace vins {
    class CameraWrapper {
    public:
        CameraWrapper(Param *param) {
            camera_ = camodocal::CameraFactory::instance()->generateCameraFromYamlFile(param->camera.calib_file);
            param_ = param;
        }
        cv::Point2f rawPoint2NormPoint(const cv::Point2f &p) {
            Eigen::Vector3d tmp_p;
            camera_->liftProjective(Eigen::Vector2d(p.x, p.y), tmp_p);
            float col = param_->camera.focal * tmp_p.x() / tmp_p.z() + param_->camera.col / 2.0;
            float row = param_->camera.focal * tmp_p.y() / tmp_p.z() + param_->camera.row / 2.0;
            return {col, row};
        }

    private:
        camodocal::CameraPtr camera_;
        Param *param_;
    };
}

#endif //GJT_VINS_CAMERA_WRAPPER_H
