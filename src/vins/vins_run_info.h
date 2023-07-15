//
// Created by gjt on 6/10/23.
//

#ifndef GJT_VINS_VINS_DATA_H
#define GJT_VINS_VINS_DATA_H

#include <string>
#include <opencv2/opencv.hpp>
#include <Eigen/Eigen>
#include "DVision/BRIEF.h"
#include <camodocal/camera_models/Camera.h>
#include "imu_integrator.h"

namespace vins {
    class RunInfo {
    public:
        Eigen::Vector3d tic;
        Eigen::Matrix3d ric;
        camodocal::CameraPtr camera_ptr;
        static RunInfo& Instance() {
            return run_info_;
        }
    private:
        static RunInfo run_info_;
    };

    struct FeaturePoint2D {
        cv::Point2f point;
        cv::Point2f velocity;
        double time_stamp;
        int feature_id;
    };

    class FeaturesOfId {
    public:
        int feature_id_         = -1;
        int start_frame_        = -1;
        bool is_outlier_        = false;
        double inv_depth        = -1;
        std::vector<FeaturePoint2D> feature_points_;

        enum {
            FeatureHaveNotSolved,
            FeatureSolvedSucc,
            FeatureSolveFail,
        }solve_flag_;

        FeaturesOfId(int _feature_id, int _start_frame)
                : feature_id_(_feature_id), start_frame_(_start_frame),
                  inv_depth(-1.0), solve_flag_(FeatureHaveNotSolved) {
        }

        [[nodiscard]] int endFrame() const {
            return start_frame_ + (int )feature_points_.size() - 1;
        }
    };

    class ImageFrame {
    public:
        ImageFrame() = delete;

        ImageFrame(std::vector<FeaturePoint2D> _points, double _t, ImuIntegrator pre_integrate_, bool _is_key_frame):
                points{std::move(_points)},
                t{_t},
                pre_integrate_(std::move(pre_integrate_)),
                is_key_frame_(_is_key_frame){};
        std::vector<FeaturePoint2D> points;
        double t{};
        Eigen::Matrix3d R;
        Eigen::Vector3d T;
        ImuIntegrator pre_integrate_;
        bool is_key_frame_ = false;
    };

    template<typename T>
    class Window {
    public:
        void push(T t) {
            queue_[(begin_ + size_) % capacity_] = std::move(t);
            if (size_ < capacity_) {
                size_++;
            } else {
                begin_ = (begin_ + 1) % capacity_;
            }
        }
        [[nodiscard]] int size() const {
            return size_;
        }
        [[nodiscard]] const T& at(int i) const {
            assert(i < size_);
            return queue_[(begin_ + i) % capacity_];
        }
        [[nodiscard]] T& at(int i) {
            assert(i < size_);
            return queue_[(begin_ + i) % capacity_];
        }

    private:
        std::vector<T> queue_;
        int capacity_ = 0;
        int begin_ = 0;
        int size_ = 0;
    };

    struct BundleAdjustWindow {
        Window<ImuIntegrator> pre_int_window;
        Window<double> time_stamp_window;
        Window<Eigen::Vector3d> ba_window;
        Window<Eigen::Vector3d> bg_window;
        Window<Eigen::Vector3d> pos_window;
        Window<Eigen::Vector3d> vel_window;
        Window<Eigen::Matrix3d> rot_window;
        Window<int> frame_id_window;
    };

}

#endif //GJT_VINS_VINS_DATA_H
