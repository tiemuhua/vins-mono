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

    struct KeyFrame {
        KeyFrame(double _time_stamp, Eigen::Vector3d &t, Eigen::Matrix3d &r, cv::Mat &_image,
                 std::vector<cv::Point3f> &_point_3d, std::vector<cv::Point2f> &_point_2d_uv);

        void getPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i) const;
        void getVioPose(Eigen::Vector3d &_T_w_i, Eigen::Matrix3d &_R_w_i) const;
        void getPosRotDrift(const Eigen::Vector3d &pos, const Eigen::Vector3d &euler,
                            Eigen::Vector3d &pos_drift, Eigen::Matrix3d &rot_drift) const;

        void updatePose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i);
        void updateVioPose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i);
        void updatePoseByDrift(const Eigen::Vector3d &t_drift, const Eigen::Matrix3d &r_drift);

        double time_stamp;
        Eigen::Vector3d vio_T_i_w_;
        Eigen::Matrix3d vio_R_i_w_;
        Eigen::Vector3d T_i_w_;
        Eigen::Matrix3d R_i_w_;

        // pnp匹配的时候新帧提供key_pts3d_和descriptors_
        vector<cv::Point3f> key_pts3d_;
        vector<DVision::BRIEF::bitset> descriptors_;
        // pnp匹配的时候老帧提供external_key_pts2d_和external_descriptors_
        vector<cv::KeyPoint> external_key_pts2d_;
        vector<DVision::BRIEF::bitset> external_descriptors_;

        LoopInfo loop_info_;
    };
    typedef std::shared_ptr<KeyFrame> KeyFramePtr;
    typedef const std::shared_ptr<const KeyFrame> ConstKeyFramePtr;
}
