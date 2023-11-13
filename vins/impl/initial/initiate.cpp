//
// Created by gjt on 6/24/23.
//

#include "initiate.h"
#include <vector>
#include <glog/logging.h>
#include "impl/visual_initiator.h"
#include "impl/visual_inertial_aligner.h"

using namespace vins;
using namespace std;

static bool isAccVariantBigEnough(const std::vector<Frame> &all_image_frame_) {
    //check imu observability
    Eigen::Vector3d sum_acc;
    for (int i = 1; i < all_image_frame_.size(); ++i) {
        const Frame &frame = all_image_frame_[i];
        double dt = frame.imu_integral_->deltaTime();
        if (abs(dt) < 1e-5) {
            LOG(INFO) << "dt is too small";
            continue;
        }
        Eigen::Vector3d tmp_acc = frame.imu_integral_->deltaVel() / dt;
        sum_acc += tmp_acc;
    }
    Eigen::Vector3d avg_acc = sum_acc / (double) all_image_frame_.size();
    assert(avg_acc.norm() < 1e4);
    double var = 0;
    for (int i = 1; i < all_image_frame_.size(); ++i) {
        const Frame &frame = all_image_frame_[i];
        double dt = frame.imu_integral_->deltaTime();
        if (abs(dt) < 1e-5) {
            LOG(INFO) << "dt is too small";
            continue;
        }
        Eigen::Vector3d tmp_acc = frame.imu_integral_->deltaVel() / dt;
        var += (tmp_acc - avg_acc).norm();
    }

    var = sqrt(var / (double) all_image_frame_.size());
    assert(abs(var) < 1e4);
    LOG(INFO) << "acc average:" << avg_acc.transpose() << ", acc variant:" << var;
    return var > 0.25;
}

bool Initiate::initiate(VinsModel &run_info) {
    if (!isAccVariantBigEnough(run_info.frame_window)) {
        return false;
    }

    std::vector<Eigen::Matrix3d> kf_img_rot(run_info.kf_state_window.size());
    std::vector<Eigen::Vector3d> kf_img_pos(run_info.kf_state_window.size());
    std::vector<Eigen::Matrix3d> frames_img_rot(run_info.frame_window.size());
    std::vector<Eigen::Vector3d> frames_img_pos(run_info.frame_window.size());

    bool visual_succ = initiateByVisual((int) run_info.kf_state_window.size(),
                                        run_info.feature_window,
                                        run_info.frame_window,
                                        kf_img_rot,
                                        kf_img_pos,
                                        frames_img_rot,
                                        frames_img_pos);
    if (!visual_succ) {
        return false;
    }

    //.求解bg，bg只与两帧之间的相对旋转有关，与绝对姿态无关，因此与ric无关，需要先求解bg后求解ric.
    std::vector<Eigen::Matrix3d> imu_delta_rots;
    std::vector<Eigen::Matrix3d> img_delta_rots;
    std::vector<Eigen::Matrix3d> jacobians_bg_2_rot;
    for (int i = 0; i < kf_img_rot.size() - 1; ++i) {
        img_delta_rots.emplace_back(kf_img_rot[i + 1] * kf_img_rot[i].transpose());
    }
    Eigen::Vector3d bg = Eigen::Vector3d::Zero();
    for (int i = 0; i < 4; ++i) {
        imu_delta_rots.clear();
        jacobians_bg_2_rot.clear();
        for (const auto &it: run_info.pre_int_window) {
            imu_delta_rots.emplace_back(it->deltaQuat().toRotationMatrix());
            jacobians_bg_2_rot.emplace_back(it->getJacobian().block<3, 3>(kOrderRot, kOrderBG));
        }
        Eigen::Vector3d bg_step = estimateGyroBias(imu_delta_rots, img_delta_rots, jacobians_bg_2_rot);
        if (bg_step.norm() > 1e4) {
            return false;
        }
        bg += bg_step;
        for (Frame &frame: run_info.frame_window) {
            frame.imu_integral_->rePredict(Eigen::Vector3d::Zero(), bg);
        }
        for (auto &pre_integrate: run_info.pre_int_window) {
            pre_integrate->rePredict(Eigen::Vector3d::Zero(), bg);
        }
    }
    for (KeyFrameState &state: run_info.kf_state_window) {
        state.bg = bg;
    }

    //.求解ric.
    imu_delta_rots.clear();
    jacobians_bg_2_rot.clear();
    for (const auto &it: run_info.pre_int_window) {
        imu_delta_rots.emplace_back(it->deltaQuat().toRotationMatrix());
        jacobians_bg_2_rot.emplace_back(it->getJacobian().block<3, 3>(kOrderRot, kOrderBG));
    }
    bool ric_succ = estimateRIC(img_delta_rots, imu_delta_rots, run_info.ric);
    if (!ric_succ) {
        return false;
    }

    //.求解重力、尺度和速度，即与位移有关的一切未知参数.
    //.重力和初始化时的ba打包在一起了，无法单独求解ba。
    //.秦博假设初始化时的ba可以忽略不计，g.norm==9.81。todo 一加6T、iPhone12的ba有多少？秦博这个假设合理吗？.
    double scale;
    std::vector<Eigen::Vector3d> velocities;
    std::vector<Eigen::Vector3d> img_delta_poses;
    for (int i = 0; i < frames_img_pos.size() - 1; ++i) {
        img_delta_poses.emplace_back(frames_img_pos[i + 1] - frames_img_pos[i]);
    }
    std::vector<Eigen::Vector3d> imu_delta_poses, imu_delta_velocities;
    std::vector<double> imu_delta_times;
    for (const Frame &frame: run_info.frame_window) {
        imu_delta_poses.emplace_back(frame.imu_integral_->deltaPos());
        imu_delta_velocities.emplace_back(frame.imu_integral_->deltaVel());
        imu_delta_times.emplace_back(frame.imu_integral_->deltaTime());
    }
    if (!estimateTICGravityScaleVelocity(frames_img_rot,
                                         img_delta_poses,
                                         imu_delta_poses,
                                         imu_delta_velocities,
                                         imu_delta_times,
                                         run_info.ric,
                                         run_info.tic,
                                         run_info.gravity,
                                         scale,
                                         velocities)) {
        return false;
    }
    for (Eigen::Vector3d &pos: kf_img_pos) {
        pos *= scale;
    }
    for (Eigen::Vector3d &pos: frames_img_pos) {
        pos *= scale;
    }

    assert(run_info.frame_window.front().is_key_frame_);
    Eigen::Matrix3d rot_diff = rotGravityToZAxis(run_info.gravity, run_info.frame_window.front().imu_rot);
    run_info.gravity = rot_diff * run_info.gravity;

    std::unordered_map<int, int> kf_idx_2_frame_idx;
    int kf_idx_iter = 0;
    for (int i = 0; i < run_info.frame_window.size(); ++i) {
        if (run_info.frame_window[i].is_key_frame_) {
            kf_idx_2_frame_idx[kf_idx_iter] = i;
            kf_idx_iter++;
        }
    }
    for (int i = 0; i < run_info.frame_window.size(); ++i) {
        run_info.frame_window[i].imu_pos = rot_diff * (run_info.ric.transpose() * frames_img_pos[i] - run_info.tic);
        run_info.frame_window[i].imu_rot = frames_img_rot[i] * rot_diff;
    }
    for (int i = 0; i < run_info.kf_state_window.size(); ++i) {
        run_info.kf_state_window[i].pos = rot_diff * (run_info.ric.transpose() * kf_img_pos[i] - run_info.tic);
        run_info.kf_state_window[i].rot = kf_img_rot[i] * rot_diff;
        run_info.kf_state_window[i].vel = rot_diff * velocities[kf_idx_2_frame_idx[i]];
    }

    // triangulate on cam pose , no tic
    //.计算特征点深度，initialStructure里面算出来的特征点三维坐标没有ba，也没有对齐惯导
    for (Feature &feature: run_info.feature_window) {
        if (!(feature.points.size() >= 2 && feature.start_kf_window_idx < run_info.kf_state_window.size() - 2))
            continue;

        Eigen::MatrixXd svd_A(2 * feature.points.size(), 4);

        int start_kf_idx = feature.start_kf_window_idx;
        Eigen::Vector3d t0 = run_info.kf_state_window[start_kf_idx].pos;
        Eigen::Matrix3d R0 = run_info.kf_state_window[start_kf_idx].rot * run_info.ric;

        // 特征点深度记为d，特征点在start_kf_idx系中坐标记为p0，在在kf_idx系中坐标记为p_i
        // start_kf_idx 帧的归一化像素坐标为 pixel0 = (x0, y0, 1.0)^T，kf_idx帧的归一化像素坐标为 pixel_i = (x_i, y_i, 1.0)
        // start_kf_idx 坐标系到kf_idx坐标系的映射为 (R, t)
        // p0 = d * (x0, y0, 1.0)^T
        // p_i = R.inv * [d * (x0, y0, 1.0)^T - t] = d * R_inv * (x0, y0, 1.0)^T - R_inv * t
        // p_i 与 (x_i, y_i, 1.0)同向
        // 令R_inv * (x0, y0, 1.0)^T = (f1, f2, f3)^T，R_inv * t = (e1, e2, e3)^T
        // 即 d * (f1, f2, f3)^T - (e1, e2, e3)^T 与 (x_i, y_i, 1.0) 同向
        // 整理为
        //      (x_i * f3 -f1) * d = x_i * e3 - e1
        //      (y_i * f3 -f2) * d = y_g * e3 - e2
        // 从而我们可以得到2 * (feature.points.size() - 1) 个方程，然后最小二乘求解
        int pts_size = feature.points.size();
        Eigen::VectorXd A = Eigen::VectorXd::Zero(2 * pts_size - 2);
        Eigen::VectorXd b = Eigen::VectorXd::Zero(2 * pts_size - 2);
        Eigen::Vector3d pixel0 = Eigen::Vector3d(feature.points[0].x, feature.points[0].y, 1.0);
        for (int i = 1; i < feature.points.size(); ++i) {
            int kf_idx = start_kf_idx + i;
            Eigen::Vector3d t1 = run_info.kf_state_window[kf_idx].pos;
            Eigen::Matrix3d R1 = run_info.kf_state_window[kf_idx].rot * run_info.ric;
            Eigen::Vector3d t = R0.transpose() * (t1 - t0);
            Eigen::Matrix3d R = R0.transpose() * R1;

            Eigen::Vector3d v1 = R.transpose() * pixel0;
            Eigen::Vector3d v2 = R.transpose() * t;
            double f1 = v1.x(), f2 = v1.y(), f3 = v1.z();
            double e1 = v2.x(), e2 = v2.y(), e3 = v2.z();

            A(2 * i - 2) = feature.points[i].x * f3 - f1;
            A(2 * i - 1) = feature.points[i].y * f3 - f2;
            b(2 * i - 2) = feature.points[i].x * e3 - e1;
            b(2 * i - 1) = feature.points[i].y * e3 - e2;
        }
        // 这实际上是个最小二乘，只不过带求解变量是一维的
        double depth = (double) (A.transpose() * b) / (double) (A.transpose() * A);
        feature.inv_depth = 1.0 / depth;
        // 校验是否有离群值
        double var = 0;
        for (int i = 0; i < b.size(); ++i) {
            double depth_i = b(i) / A(i);
            var += (depth_i - depth) * (depth_i - depth);
        }
        var /= b.size();
        double sqrt_var = sqrt(var);
        if (sqrt_var / depth > 0.1) {
            feature.inv_depth = -1;
        }
        // todo 如果只有很少的离群值，可以剔除离群值后再算一遍
    }
    return true;
}
