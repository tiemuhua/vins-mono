//
// Created by gjt on 6/9/23.
//

#include "feature_retriever.h"
#include <vector>
#include "vins/vins_utils.h"

using namespace vins::FeatureRetriever;
using namespace vins;
using namespace DVision;
using namespace Eigen;
using namespace std;

inline int HammingDis(const BRIEF::bitset &a, const BRIEF::bitset &b) {
    return (a ^ b).count();
}

static bool searchInArea(const DVision::BRIEF::bitset& descriptor,
                         ConstKeyFramePtr old_kf,
                         cv::Point2f &best_match_norm) {
    int bestDist = 128;
    int bestIndex = -1;
    for (int i = 0; i < (int) old_kf->external_descriptors_.size(); i++) {
        int dis = HammingDis(descriptor, old_kf->external_descriptors_[i]);
        if (dis < bestDist) {
            bestDist = dis;
            bestIndex = i;
        }
    }
    if (bestIndex != -1 && bestDist < 80) {
        best_match_norm = old_kf->external_key_pts2d_[bestIndex].pt;
        return true;
    } else
        return false;
}

static void searchByBRIEFDes(ConstKeyFramePtr old_kf,
                             const std::vector<BRIEF::bitset> &descriptors,
                             std::vector<cv::Point2f> &pts2d_in_old_frame,
                             std::vector<uchar> &status) {
    for (const auto & descriptor : descriptors) {
        cv::Point2f pt(0.f, 0.f);
        if (searchInArea(descriptor, old_kf, pt))
            status.push_back(1);
        else
            status.push_back(0);
        pts2d_in_old_frame.push_back(pt);
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
bool findLoop(ConstKeyFramePtr old_kf,
              const int old_kf_id,
              ConstKeyFramePtr new_kf,
              LoopInfo &loop_info) {
    vector<cv::Point3f> pts3d_in_new_frame = new_kf->key_pts3d_;
    vector<uint8_t> status;
    vector<cv::Point2f> pts2d_in_old_frame;
    searchByBRIEFDes(old_kf, new_kf->descriptors_, pts2d_in_old_frame, status);
    utils::reduceVector(pts2d_in_old_frame, status);
    utils::reduceVector(pts3d_in_new_frame, status);
    if (pts2d_in_old_frame.size() < min_loop_key_points_num) {
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
    PnpRANSAC(pts2d_in_old_frame, pts3d_in_new_frame, R_o_n_vio, T_o_n_vio, status, R_o_n_pnp, T_o_n_pnp);
    utils::reduceVector(pts2d_in_old_frame, status);
    utils::reduceVector(pts3d_in_new_frame, status);
    if (pts2d_in_old_frame.size() < min_loop_key_points_num) {
        return false;
    }

    loop_info.relative_pos = T_o_n_pnp;
    double old_yaw = utils::rot2ypr(R_o_n_vio * new_kf->vio_R_i_w_).x();
    double new_yaw = utils::rot2ypr(new_kf->vio_R_i_w_).x();
    loop_info.relative_yaw = utils::normalizeAnglePi(old_yaw - new_yaw);
    if (abs(loop_info.relative_yaw) < pi/6 && loop_info.relative_pos.norm() < 20.0) {
        loop_info.peer_frame_id = old_kf_id;
        return true;
    }
    return false;
}