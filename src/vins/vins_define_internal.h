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
