#include "keyframe.h"
#include <opencv2/imgproc/types_c.h>
#include "vins/vins_run_info.h"
#include "vins/vins_utils.h"

using namespace vins;
using namespace Eigen;
using namespace std;
using namespace DVision;

// create keyframe online
KeyFrame::KeyFrame(double _time_stamp, Vector3d &t, Matrix3d &r, cv::Mat &_image,
                   vector<cv::Point3f> &_point_3d, vector<cv::Point2f> &_point_2d_uv) {
    time_stamp = _time_stamp;
    vio_T_i_w_ = t;
    vio_R_i_w_ = r;
    T_i_w_ = vio_T_i_w_;
    R_i_w_ = vio_R_i_w_;
    key_pts3d_ = _point_3d;

    vector<cv::KeyPoint> key_points;
    for (auto & i : _point_2d_uv) {
        cv::KeyPoint key;
        key.pt = i;
        key_points.push_back(key);
    }
    RunInfo::Instance().extractor.m_brief.compute(_image, key_points, descriptors_);

    const int fast_th = 20; // corner detector response threshold
    vector<cv::KeyPoint> external_key_points_un_normalized;
    cv::FAST(_image, external_key_points_un_normalized, fast_th, true);
    RunInfo::Instance().extractor.m_brief.compute(_image, external_key_points_un_normalized, external_descriptors_);
    for (auto & keypoint : external_key_points_un_normalized) {
        Eigen::Vector3d tmp_p;
        RunInfo::Instance().camera_ptr->liftProjective(Eigen::Vector2d(keypoint.pt.x, keypoint.pt.y), tmp_p);
        cv::KeyPoint tmp_norm;
        tmp_norm.pt = cv::Point2f(tmp_p.x() / tmp_p.z(), tmp_p.y() / tmp_p.z());
        external_key_pts2d_.push_back(tmp_norm);
    }
}

void KeyFrame::getVioPose(Eigen::Vector3d &_T_i_w, Eigen::Matrix3d &_R_i_w) const {
    _T_i_w = vio_T_i_w_;
    _R_i_w = vio_R_i_w_;
}

void KeyFrame::getPosRotDrift(const Eigen::Vector3d &pos, const Eigen::Vector3d &euler,
                              Eigen::Vector3d &pos_drift, Eigen::Matrix3d &rot_drift) const {
    double yaw_drift = euler.x() - utils::rot2ypr(vio_R_i_w_).x();
    rot_drift = utils::ypr2rot(Vector3d(yaw_drift, 0, 0));
    pos_drift = pos - rot_drift * vio_T_i_w_;
}

void KeyFrame::getPose(Eigen::Vector3d &_T_i_w, Eigen::Matrix3d &_R_i_w) const {
    _T_i_w = T_i_w_;
    _R_i_w = R_i_w_;
}

void KeyFrame::updatePose(const Eigen::Vector3d &_T_w_i, const Eigen::Matrix3d &_R_w_i) {
    T_i_w_ = _T_w_i;
    R_i_w_ = _R_w_i;
}

void KeyFrame::updatePoseByDrift(const Eigen::Vector3d &t_drift, const Eigen::Matrix3d &r_drift) {
    T_i_w_ = r_drift * T_i_w_ + t_drift;
    R_i_w_ = r_drift * R_i_w_;
}
