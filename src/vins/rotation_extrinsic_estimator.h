#pragma once

#include "vins_define_internal.h"
#include <vector>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

namespace vins {
    typedef std::pair<cv::Point2f, cv::Point2f> PointCorrespondence;
    typedef std::vector<PointCorrespondence> PointCorrespondences;
    class RotationExtrinsicEstimator {
    public:
        RotationExtrinsicEstimator(int window_size){
            window_size_ = window_size;
        }

        bool calibrateRotationExtrinsic(const PointCorrespondences& correspondences, ConstQuatRef delta_q_imu,
                                        Eigen::Matrix3d &calib_ric_result);

    private:
        static Eigen::Matrix3d solveRelativeR(const PointCorrespondences &correspondences);

        static double testTriangulation(const std::vector<cv::Point2f> &l,
                                        const std::vector<cv::Point2f> &r,
                                        cv::Mat_<double> R, cv::Mat_<double> t);

        static void decomposeE(const cv::Mat& E,
                               cv::Mat_<double> &R1, cv::Mat_<double> &R2,
                               cv::Mat_<double> &t1, cv::Mat_<double> &t2);

        std::vector<Eigen::Matrix3d> rot_visual_que_;
        std::vector<Eigen::Matrix3d> rot_imu_que_;
        std::vector<Eigen::Matrix3d> rot_imu_in_world_frame_que_;
        Eigen::Matrix3d ric_ = Eigen::Matrix3d::Identity();
        int window_size_ = 10;
    };
}


