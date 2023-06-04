#pragma once

#include <vector>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "ThirdParty/DBoW/DBoW2.h"
#include "ThirdParty/DVision/DVision.h"

#define MIN_LOOP_NUM 25

namespace vins{

    class BriefExtractor
    {
    public:
        virtual void operator()(const cv::Mat &im, vector<cv::KeyPoint> &keys, vector<DVision::BRIEF::bitset> &descriptors) const;
        BriefExtractor(const std::string &pattern_file);

        DVision::BRIEF m_brief;
    };

    class KeyFrame
    {
    public:
        KeyFrame(double _time_stamp, int _index, Eigen::Vector3d &_vio_T_w_i, Eigen::Matrix3d &_vio_R_w_i, cv::Mat &_image,
                 vector<cv::Point3f> &_point_3d, vector<cv::Point2f> &_point_2d_uv, vector<cv::Point2f> &_point_2d_normal,
                 vector<double> &_point_id, int _sequence);
        KeyFrame(double _time_stamp, int _index, Eigen::Vector3d &_vio_T_w_i, Eigen::Matrix3d &_vio_R_w_i, Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i,
                 cv::Mat &_image, int _loop_index, Eigen::Matrix<double, 8, 1 > &_loop_info,
                 vector<cv::KeyPoint> &_keypoints, vector<cv::KeyPoint> &_keypoints_norm, vector<DVision::BRIEF::bitset> &_brief_descriptors);
        bool findConnection(KeyFrame* old_kf);
        void computeWindowBRIEFPoint();
        void computeBRIEFPoint();
        //void extractBrief();
        static int HammingDis(const DVision::BRIEF::bitset &a, const DVision::BRIEF::bitset &b);
        bool searchInAera(const DVision::BRIEF::bitset& window_descriptor,
                          const std::vector<DVision::BRIEF::bitset> &descriptors_old,
                          const std::vector<cv::KeyPoint> &keypoints_old,
                          const std::vector<cv::KeyPoint> &keypoints_old_norm,
                          cv::Point2f &best_match,
                          cv::Point2f &best_match_norm);
        void searchByBRIEFDes(std::vector<cv::Point2f> &matched_2d_old,
                              std::vector<cv::Point2f> &matched_2d_old_norm,
                              std::vector<uchar> &status,
                              const std::vector<DVision::BRIEF::bitset> &descriptors_old,
                              const std::vector<cv::KeyPoint> &keypoints_old,
                              const std::vector<cv::KeyPoint> &keypoints_old_norm);
        void PnPRANSAC(const vector<cv::Point2f> &matched_2d_old_norm,
                       const std::vector<cv::Point3f> &matched_3d,
                       std::vector<uchar> &status,
                       Eigen::Vector3d &PnP_T_old, Eigen::Matrix3d &PnP_R_old);
        void getVioPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i) const;
        void getPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i);
        void updatePose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i);
        void updateVioPose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i);
        void updateLoop(Eigen::Matrix<double, 8, 1 > &_loop_info);

        Eigen::Vector3d getLoopRelativeT();
        double getLoopRelativeYaw();
        Eigen::Quaterniond getLoopRelativeQ();



        double time_stamp;
        int index;
        int local_index;
        Eigen::Vector3d vio_T_w_i;
        Eigen::Matrix3d vio_R_w_i;
        Eigen::Vector3d T_w_i;
        Eigen::Matrix3d R_w_i;
        Eigen::Vector3d origin_vio_T;
        Eigen::Matrix3d origin_vio_R;
        cv::Mat image;
        cv::Mat thumbnail;
        vector<cv::Point3f> point_3d;
        vector<cv::Point2f> point_2d_uv;
        vector<cv::Point2f> point_2d_norm;
        vector<double> point_id;
        vector<cv::KeyPoint> keypoints;
        vector<cv::KeyPoint> keypoints_norm;
        vector<cv::KeyPoint> window_keypoints;
        vector<DVision::BRIEF::bitset> brief_descriptors;
        vector<DVision::BRIEF::bitset> window_brief_descriptors;
        bool has_fast_point;
        int sequence;

        bool has_loop;
        int loop_index;
        Eigen::Matrix<double, 8, 1 > loop_info;
    };
}
