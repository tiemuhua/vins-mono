#pragma once

#include <vector>
#include <eigen3/Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include "ThirdParty/DBoW/DBoW2.h"
#include "ThirdParty/DVision/DVision.h"
#include "../vins_define_internal.h"
#define MIN_LOOP_NUM 25

namespace vins{

    class BriefExtractor {
    public:
        BriefExtractor(const std::string &pattern_file);
        DVision::BRIEF m_brief;
    };

    struct LoopInfo {
        int peer_frame_id = -1;
        Eigen::Vector3d relative_pos = Eigen::Vector3d::Zero();
        double relative_yaw = 0.0;
    };

    class KeyFrame {
    public:
        KeyFrame(double _time_stamp, Eigen::Vector3d &_vio_T_w_i, Eigen::Matrix3d &_vio_R_w_i, cv::Mat &_image,
                 vector<cv::Point3f> &_point_3d, vector<cv::Point2f> &_point_2d_uv, vector<cv::Point2f> &_point_2d_normal,
                 vector<double> &_point_id, int _sequence);
        bool findConnection(const KeyFrame *old_kf, int old_kf_id);

        void computeWindowBRIEFPoint(const std::string &pattern_file);
        void computeBRIEFPoint(const std::string &pattern_file);

        void getVioPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i) const;
        void getPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i) const;
        void getPosRotDrift(const Eigen::Vector3d &pos, const Eigen::Vector3d &euler,
                            Eigen::Vector3d &pos_drift, Eigen::Matrix3d &rot_drift) const;
        void updatePose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i);
        void updatePoseByDrift(const Eigen::Vector3d &t_drift, const Eigen::Matrix3d &r_drift);
        void updateVioPose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i);

        Eigen::Vector3d getLoopRelativeT();
        double getLoopRelativeYaw();
        Eigen::Quaterniond getLoopRelativeQ();

        double time_stamp;
        Eigen::Vector3d vio_T_i_w_;
        Eigen::Matrix3d vio_R_i_w_;
        Eigen::Vector3d T_i_w_;
        Eigen::Matrix3d R_i_w_;
        Eigen::Vector3d origin_vio_T;
        Eigen::Matrix3d origin_vio_R;
        cv::Mat image;
        vector<cv::KeyPoint> external_key_points_;
        vector<DVision::BRIEF::bitset> external_brief_descriptors;
        vector<cv::Point3f> key_points_pos_;
        vector<DVision::BRIEF::bitset> descriptors;

        LoopInfo loop_info_;
    private:
        void PnPRANSAC(const vector<cv::Point2f> &matched_2d_old_norm,
                       const std::vector<cv::Point3f> &matched_3d,
                       std::vector<uchar> &status,
                       Eigen::Vector3d &PnP_T_old, Eigen::Matrix3d &PnP_R_old);
    };
    typedef std::shared_ptr<KeyFrame> KeyFramePtr;
    typedef const std::shared_ptr<const KeyFrame> ConstKeyFramePtr;
}
