//
// Created by gjt on 5/14/23.
//

#ifndef VINS_VINS_H
#define VINS_VINS_H

#include "imu_integrator.h"
#include <opencv2/opencv.hpp>

namespace vins {
#define Synchronized(mutex_)  for(ScopedLocker locker(mutex_); locker.cnt < 1; locker.cnt++)

    class ScopedLocker {
    public:
        explicit ScopedLocker(std::mutex& mutex) : guard(mutex) {}
        std::lock_guard<std::mutex> guard;
        int cnt = 0;
    };

    struct FeaturePoint {
        cv::Point2f point;
        cv::Point2f unified_point;
        cv::Point2f point_velocity;
        int feature_id;
    };

    class FeaturesOfId {
    public:
        const int feature_id_;
        int start_frame_;
        std::vector<FeaturePoint> feature_points_;

        bool is_outlier_{};
        double estimated_depth;
        enum {
            FeatureHaveNotSolved,
            FeatureSolvedSucc,
            FeatureSolveFail,
        }solve_flag_;

        FeaturesOfId(int _feature_id, int _start_frame)
                : feature_id_(_feature_id), start_frame_(_start_frame),
                  estimated_depth(-1.0), solve_flag_(FeatureHaveNotSolved) {
        }

        [[nodiscard]] int endFrame() const {
            return start_frame_ + (int )feature_points_.size() - 1;
        }
    };

    class ImageFrame {
    public:
        ImageFrame() = delete;

        ImageFrame(std::vector<FeaturePoint> _points, double _t, ImuIntegrator pre_integrate_, bool _is_key_frame):
                points{std::move(_points)},
                t{_t},
                pre_integrate_(std::move(pre_integrate_)),
                is_key_frame_(_is_key_frame){};
        std::vector<FeaturePoint> points;
        double t{};
        Eigen::Matrix3d R;
        Eigen::Vector3d T;
        ImuIntegrator pre_integrate_;
        bool is_key_frame_ = false;
    };

    typedef std::vector<ImuIntegrator> PreIntegrateWindow;
    typedef std::vector<double> TimeStampWindow;
    typedef std::vector<Eigen::Vector3d> BaWindow;
    typedef std::vector<Eigen::Vector3d> BgWindow;
    typedef std::vector<Eigen::Vector3d> PosWindow;
    typedef std::vector<Eigen::Vector3d> VelWindow;
    typedef std::vector<Eigen::Matrix3d> RotWindow;
    typedef std::vector<Eigen::Matrix3d> RotWindow;
}


#endif //VINS_VINS_H
