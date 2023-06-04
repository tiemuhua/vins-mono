#pragma once

#include <vector>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "ThirdParty/DBoW/DBoW2.h"
#include "ThirdParty/DVision/DVision.h"

#define MIN_LOOP_NUM 25

namespace vins{

    class BriefExtractor {
    public:
        BriefExtractor(const std::string &pattern_file);
        DVision::BRIEF m_brief;
    };

    class KeyFrame
    {
    public:
        struct LoopInfo{
            Eigen::Vector3d relative_t;
            Eigen::Quaterniond relative_q;
            double relative_yaw;
        };

        KeyFrame(double _time_stamp, Eigen::Vector3d &_vio_T_w_i, Eigen::Matrix3d &_vio_R_w_i, cv::Mat &_image,
                 vector<cv::Point3f> &_point_3d, vector<cv::Point2f> &_point_2d_uv, vector<cv::Point2f> &_point_2d_normal,
                 vector<double> &_point_id, int _sequence);
        KeyFrame(double _time_stamp, Eigen::Vector3d &_vio_T_w_i, Eigen::Matrix3d &_vio_R_w_i, Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i,
                 cv::Mat &_image, int _loop_index, const LoopInfo &_loop_info,
                 vector<cv::KeyPoint> &_keypoints, vector<cv::KeyPoint> &_keypoints_norm, vector<DVision::BRIEF::bitset> &_brief_descriptors);
        bool findConnection(KeyFrame *old_kf, int old_kf_id);

        void computeWindowBRIEFPoint(const std::string &pattern_file);
        void computeBRIEFPoint(const std::string &pattern_file);

        static int HammingDis(const DVision::BRIEF::bitset &a, const DVision::BRIEF::bitset &b);
        void getVioPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i) const;
        void getPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i);
        void updatePose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i);
        void updateVioPose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i);
        void updateLoop(Eigen::Matrix<double, 8, 1 > &_loop_info);

        Eigen::Vector3d getLoopRelativeT();
        double getLoopRelativeYaw();
        Eigen::Quaterniond getLoopRelativeQ();

        double time_stamp;
        Eigen::Vector3d vio_T_w_i;
        Eigen::Matrix3d vio_R_w_i;
        Eigen::Vector3d T_w_i;
        Eigen::Matrix3d R_w_i;
        Eigen::Vector3d origin_vio_T;
        Eigen::Matrix3d origin_vio_R;
        cv::Mat image;
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
        int loop_peer_id_;
        LoopInfo loop_info_;
    private:
        bool searchInAera(const DVision::BRIEF::bitset& window_descriptor,
                          const KeyFrame* old_kf,
                          cv::Point2f &best_match,
                          cv::Point2f &best_match_norm);
        void searchByBRIEFDes(std::vector<cv::Point2f> &matched_2d_old,
                              std::vector<cv::Point2f> &matched_2d_old_norm,
                              std::vector<uchar> &status,
                              const KeyFrame* old_kf);
        void PnPRANSAC(const vector<cv::Point2f> &matched_2d_old_norm,
                       const std::vector<cv::Point3f> &matched_3d,
                       std::vector<uchar> &status,
                       Eigen::Vector3d &PnP_T_old, Eigen::Matrix3d &PnP_R_old);
    };
}
