//
// Created by gjt on 8/29/23.
//

#ifndef GJT_VINS_CAMERA_WRAPPER_H
#define GJT_VINS_CAMERA_WRAPPER_H

#include "opencv2/opencv.hpp"
#include <Eigen/Eigen>

#include "camodocal/camera_models/CameraFactory.h"
#include "param.h"

namespace vins {
    class CameraWrapper {
    public:
        CameraWrapper(const Param &param) {
            camera_ = camodocal::CameraFactory::instance()->generateCamera(
                    camodocal::Camera::ModelType::PINHOLE,
                    "vins_camera",
                    cv::Size(param.camera.col, param.camera.row)
            );
            camera_param_ = param.camera;
        }

        cv::Point2f rawPoint2NormPoint(const cv::Point2f &p) const {
            return p;
//            Eigen::Vector3d tmp_p;
//            camera_->liftProjective(Eigen::Vector2d(p.x, p.y), tmp_p);
//            float col = focal_ * tmp_p.x() / tmp_p.z() + camera_->imageWidth() / 2.0;
//            float row = focal_ * tmp_p.y() / tmp_p.z() + camera_->imageHeight() / 2.0;
//            return {col, row};
        }
        cv::Mat getCameraMatrix() const {
            cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) <<
                    camera_param_.f_x, 0, camera_param_.cx,
                    0, camera_param_.f_y, camera_param_.cy,
                    0, 0, 1);
            return camera_matrix;
        }

        camodocal::CameraPtr camera_ = nullptr;
        CameraParam camera_param_;
    };
}

#endif //GJT_VINS_CAMERA_WRAPPER_H
