//
// Created by gjt on 5/14/23.
//

#ifndef VINS_VINS_H
#define VINS_VINS_H

#include <Eigen/Eigen>
#include <opencv2/opencv.hpp>

namespace vins {
#define Synchronized(mutex_)  for(ScopedLocker locker(mutex_); locker.cnt < 1; locker.cnt++)

    class ScopedLocker {
    public:
        explicit ScopedLocker(std::mutex& mutex) : guard(mutex) {}
        std::lock_guard<std::mutex> guard;
        int cnt = 0;
    };

    // point和velocity已经投影到归一化平面
    struct FeaturePoint2D {
        cv::Point2f point;
        cv::Point2f velocity;
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
        }solve_flag_ = FeatureHaveNotSolved;

        FeaturesOfId(int _feature_id, int _start_frame)
                : feature_id_(_feature_id), start_frame_(_start_frame) {}

        [[nodiscard]] int endFrame() const {
            return start_frame_ + (int )feature_points_.size() - 1;
        }
    };

    class ImuIntegrator;
    class ImageFrame {
    public:
        ImageFrame() = delete;

        ImageFrame(std::vector<FeaturePoint2D> _points, double _t, std::shared_ptr<ImuIntegrator> _pre_integral, bool _is_key_frame):
                points{std::move(_points)},
                t{_t},
                pre_integral_(std::move(_pre_integral)),
                is_key_frame_(_is_key_frame){};
        std::vector<FeaturePoint2D> points;
        double t{};
        Eigen::Matrix3d R;
        Eigen::Vector3d T;
        std::shared_ptr<ImuIntegrator> pre_integral_;
        bool is_key_frame_ = false;
    };


    typedef const Eigen::Matrix3d & ConstMat3dRef;
    typedef const Eigen::Vector3d & ConstVec3dRef;
    typedef const Eigen::Quaterniond & ConstQuatRef;
    typedef Eigen::Matrix3d & Mat3dRef;
    typedef Eigen::Vector3d & Vec3dRef;
    typedef Eigen::Quaterniond & QuatRef;

    typedef std::vector<std::pair<cv::Point2f , cv::Point2f>> Correspondences;

    struct MatchPoint{
        cv::Point2f point;
        int feature_id;
    };
    struct LoopMatchInfo {
        int peer_frame_id = -1;
        std::vector<MatchPoint> match_points;
    };

    constexpr double pi = 3.1415926;
}


#endif //VINS_VINS_H
