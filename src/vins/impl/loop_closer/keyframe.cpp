#include "keyframe.h"
#include <opencv2/imgproc/types_c.h>
#include "vins/impl/vins_run_info.h"
#include "vins/impl/vins_utils.h"

using namespace vins;
using namespace Eigen;
using namespace std;

// create keyframe online
KeyFrame::KeyFrame(const Frame &_base_frame,
                   const std::vector<cv::Point3f> &_point_3d,
                   const std::vector<DVision::BRIEF::bitset> &descriptors,
                   const std::vector<DVision::BRIEF::bitset> &external_descriptors) {
    base_frame_ = _base_frame;
    T_i_w_ = vio_T_i_w_;
    R_i_w_ = vio_R_i_w_;
    key_pts3d_ = _point_3d;
    descriptors_ = descriptors;
    external_descriptors_ = external_descriptors;
}

void KeyFrame::getVioPose(Eigen::Vector3d &_T_i_w, Eigen::Matrix3d &_R_i_w) const {
    _T_i_w = vio_T_i_w_;
    _R_i_w = vio_R_i_w_;
}

void calculatePoseRotDrift(
        const Eigen::Vector3d &pos1, const Eigen::Vector3d &euler1,
        const Eigen::Vector3d &pos2, const Eigen::Vector3d &euler2,
        Eigen::Vector3d &pos_drift, Eigen::Matrix3d &rot_drift) {
    double yaw_drift = euler1.x() - euler2.x();
    rot_drift = utils::ypr2rot(Vector3d(yaw_drift, 0, 0));
    pos_drift = pos1 - pos2;
}

void KeyFrame::getLoopedPose(Eigen::Vector3d &_T_i_w, Eigen::Matrix3d &_R_i_w) const {
    _T_i_w = T_i_w_;
    _R_i_w = R_i_w_;
}

void KeyFrame::updateLoopedPose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i) {
    T_i_w_ = _T_w_i;
    R_i_w_ = _R_w_i;
}

void KeyFrame::updatePoseByDrift(const Eigen::Vector3d &t_drift, const Eigen::Matrix3d &r_drift) {
    T_i_w_ = r_drift * T_i_w_ + t_drift;
    R_i_w_ = r_drift * R_i_w_;
}
