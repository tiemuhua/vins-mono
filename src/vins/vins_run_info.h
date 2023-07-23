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

    template<typename T>
    class Window {
    public:
        Window(int capacity) {
            capacity_ = capacity;
        }
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
        int size;
    };

    camodocal::CameraPtr CameraInstance();


    class RunInfo {
    public:
        Eigen::Vector3d tic;
        Eigen::Matrix3d ric;
        Eigen::Vector3d gravity;
        BundleAdjustWindow window;
        std::vector<ImageFrame> all_frames;
        static RunInfo& Instance() {
            return run_info_;
        }
    private:
        static RunInfo run_info_;
    };
}

#endif //GJT_VINS_VINS_DATA_H
