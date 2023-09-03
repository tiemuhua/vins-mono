//
// Created by gjt on 6/9/23.
//

#include "loop_relative_pos.h"
#include <vector>
#include <opencv2/core/eigen.hpp>
#include "vins/impl/vins_utils.h"
#include "vins/impl/loop_closer/keyframe.h"

using namespace vins;
using namespace DVision;
using namespace Eigen;
using namespace std;

inline int HammingDis(const BRIEF::bitset &a, const BRIEF::bitset &b) {
    return (a ^ b).count();
}

static bool searchInAreaForBestIdx(const DVision::BRIEF::bitset& new_descriptor,
                                   const std::vector<DVision::BRIEF::bitset>& old_descriptors) {
    int bestDist = 128;
    int bestIndex = -1;
    for (int i = 0; i < (int) old_descriptors.size(); i++) {
        int dis = HammingDis(new_descriptor, old_descriptors[i]);
        if (dis < bestDist) {
            bestDist = dis;
            bestIndex = i;
        }
    }
    if (bestDist > 80) {
        return -1;
    }
    return bestIndex;
}

static void searchByBRIEFDes(const std::vector<DVision::BRIEF::bitset> &new_descriptors,
                             const std::vector<DVision::BRIEF::bitset>& old_descriptors,
                             const std::vector<cv::Point2f> &old_pts2d_without_order,
                             std::vector<cv::Point2f> &old_pts2d,
                             std::vector<uchar> &status) {
    for (const auto & new_descriptor : new_descriptors) {
        int idx = searchInAreaForBestIdx(new_descriptor, old_descriptors);
        if (idx == -1) {
            status.push_back(0);
            old_pts2d.emplace_back();
        } else {
            status.push_back(0);
            old_pts2d.push_back(old_pts2d_without_order[idx]);
        }
    }
}

/**
 * @param pts2d_in_old_frame 特征点在旧帧中的归一化像素坐标
 * @param pts3d_in_new_frame 特征点在新帧IMU坐标系中的三维坐标
 * @param R_init @param T_init VIO前端得到的旧帧相机在新帧IMU坐标系的位姿
 * @param R_pnp @param T_pnp pnp得到的旧帧相机在新帧IMU坐标系中的位姿
 * */
static void PnpRANSAC(const vector<cv::Point2f> &pts2d_in_old_frame,
                      const std::vector<cv::Point3f> &pts3d_in_new_frame,
                      const Eigen::Matrix3d &R_init,
                      const Eigen::Vector3d &T_init,
                      std::vector<uchar> &status,
                      Eigen::Matrix3d &R_pnp,
                      Eigen::Vector3d &T_pnp) {
    cv::Mat r_cv, r_vec, t_cv, D;
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0);
    cv::eigen2cv(R_init, r_cv);
    cv::Rodrigues(r_cv, r_vec);
    cv::eigen2cv(T_init, t_cv);
    cv::Mat inliers;

    solvePnPRansac(pts3d_in_new_frame, pts2d_in_old_frame, K, D, r_vec, t_cv,
                   true, 100, 10.0 / 460.0, 0.99, inliers);

    status = std::vector<uchar>(pts3d_in_new_frame.size(), 0);
    for (int i = 0; i < inliers.rows; i++) {
        int n = inliers.at<int>(i);
        status[n] = 1;
    }

    cv::Rodrigues(r_vec, r_cv);
    cv::cv2eigen(r_cv, R_pnp);
    cv::cv2eigen(t_cv, T_pnp);
}

static constexpr int min_loop_key_points_num = 25;
bool vins::buildLoopRelation(ConstKeyFramePtr &old_kf,
                             int old_kf_id,
                             const KeyFramePtr &new_kf,
                             std::vector<uint8_t> &status,
                             std::vector<cv::Point2f> &old_frame_pts2d) {
    searchByBRIEFDes(new_kf->descriptors_, old_kf->descriptors_, old_kf->base_frame_.points, old_frame_pts2d, status);
    utils::reduceVector(old_frame_pts2d, status);
    vector<cv::Point3f> new_frame_pts3d = new_kf->key_pts3d_;
    utils::reduceVector(new_frame_pts3d, status);
    if (old_frame_pts2d.size() < min_loop_key_points_num) {
        return false;
    }

    // R_n_w: new帧在world坐标系中的姿态
    // R_o_w: old帧在world坐标系中的姿态
    // R_o_n: old帧在new帧坐标系中的姿态
    // 则有R_o_n = R_o_w * R_w_n
    Eigen::Matrix3d R_o_n_vio = old_kf->vio_R_i_w_ * new_kf->vio_R_i_w_.transpose();
    Eigen::Vector3d T_o_n_vio = new_kf->vio_R_i_w_.transpose() * (old_kf->vio_T_i_w_ - new_kf->vio_T_i_w_);
    Eigen::Matrix3d R_o_n_pnp;
    Eigen::Vector3d T_o_n_pnp;
    status.clear();
    PnpRANSAC(old_frame_pts2d, new_frame_pts3d, R_o_n_vio, T_o_n_vio, status, R_o_n_pnp, T_o_n_pnp);
    utils::reduceVector(old_frame_pts2d, status);
    utils::reduceVector(new_frame_pts3d, status);
    if (old_frame_pts2d.size() < min_loop_key_points_num) {
        return false;
    }
    PnpRANSAC(old_frame_pts2d, new_frame_pts3d, R_o_n_vio, T_o_n_vio, status, R_o_n_pnp, T_o_n_pnp);

    new_kf->loop_relative_pose_.relative_pos = T_o_n_pnp;
    double old_yaw = utils::rot2ypr(R_o_n_vio * new_kf->vio_R_i_w_).x();
    double new_yaw = utils::rot2ypr(new_kf->vio_R_i_w_).x();
    new_kf->loop_relative_pose_.relative_yaw = utils::normalizeAnglePi(old_yaw - new_yaw);
    if (abs(new_kf->loop_relative_pose_.relative_yaw) < pi / 6 && new_kf->loop_relative_pose_.relative_pos.norm() < 20.0) {
        new_kf->loop_relative_pose_.peer_frame_id = old_kf_id;
        return true;
    }
    return false;
}