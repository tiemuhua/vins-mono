//
// Created by gjt on 6/24/23.
//

#include "initiate.h"
#include <vector>
#include "impl/visual_inertial_aligner.h"
#include "impl/visual_initiator.h"
#include "log.h"
using namespace vins;
using namespace std;

static bool isAccVariantBigEnough(const std::vector<Frame> &all_image_frame_) {
    //check imu observability
    Eigen::Vector3d sum_acc;
    // todo tiemuhuaguo 原始代码很奇怪，all_image_frame隔一个用一个，而且all_image_frame.size() - 1是什么意思？
    for (const Frame &frame: all_image_frame_) {
        double dt = frame.pre_integral_->deltaTime();
        Eigen::Vector3d tmp_acc = frame.pre_integral_->deltaVel() / dt;
        sum_acc += tmp_acc;
    }
    Eigen::Vector3d avg_acc = sum_acc / (double )all_image_frame_.size();
    double var = 0;
    for (const Frame &frame:all_image_frame_) {
        double dt = frame.pre_integral_->deltaTime();
        Eigen::Vector3d tmp_acc = frame.pre_integral_->deltaVel() / dt;
        var += (tmp_acc - avg_acc).transpose() * (tmp_acc - avg_acc);
    }

    var = sqrt(var / (double )all_image_frame_.size());
    LOG_I("IMU acc variant:%f", var);
    return var > 0.25;
}

bool Initiate::initiate(const double gravity_norm, RunInfo &run_info) {
    if (!isAccVariantBigEnough(run_info.all_frames)) {
        return false;
    }

    bool visual_succ = initiateByVisual(run_info.state_window.size(),
                                        run_info.feature_window,
                                        run_info.all_frames);
    if (!visual_succ) {
        return false;
    }

    Eigen::Vector3d delta_bg;
    Eigen::Matrix3d rot_diff;
    std::vector<Eigen::Vector3d> velocities;
    double scale;
    bool align_succ = alignVisualAndInertial(gravity_norm,
                                             run_info.tic,
                                             run_info.ric,
                                             run_info.all_frames,
                                             run_info.gravity,
                                             delta_bg,
                                             rot_diff,
                                             velocities,
                                             scale);
    if (!align_succ) {
        return false;
    }

    auto &state_window = run_info.state_window;
    auto &all_frames = run_info.all_frames;
    auto &TIC = run_info.tic;
    auto RIC = run_info.ric;
    for (State & state : state_window) {
        state.bg = state.bg + delta_bg;
    }
    int frame_size = (int )all_frames.size();
    for (int frame_idx = 0, key_frame_id = 0; frame_idx < frame_size; frame_idx++) {
        if (!all_frames[frame_idx].is_key_frame_) {
            continue;
        }
        state_window.at(key_frame_id).rot = rot_diff * all_frames.at(key_frame_id).R;
        state_window.at(key_frame_id).pos = rot_diff * all_frames.at(key_frame_id).T;
        state_window.at(key_frame_id).vel = rot_diff * velocities.at(key_frame_id);
        key_frame_id++;
    }
    for (int i = 0; i < (int) state_window.size(); ++i) {
        state_window.at(i).pos =
                (scale * state_window.at(i).pos - state_window.at(i).rot * TIC) -
                (scale * state_window.at(0).pos - state_window.at(0).rot * TIC);
    }

    for (int i = 0; i < (int) run_info.pre_int_window.size(); ++i) {
        run_info.pre_int_window.at(i).rePredict(Eigen::Vector3d::Zero(), state_window.at(i).bg);
    }

    //triangulate on cam pose , no tic
    //计算特征点深度，initialStructure里面算出来的特征点三维坐标没有ba，也没有对齐惯导
    for (Feature &feature: run_info.feature_window) {
        if (!(feature.points.size() >= 2 && feature.start_kf_idx < run_info.state_window.size() - 2))
            continue;

        Eigen::MatrixXd svd_A(2 * feature.points.size(), 4);

        int imu_i = feature.start_kf_idx;
        Eigen::Vector3d t0 = state_window.at(imu_i).pos;
        Eigen::Matrix3d R0 = state_window.at(imu_i).rot * RIC;

        for (int i = 0; i < feature.points.size(); ++i) {
            int imu_j = feature.start_kf_idx + i;
            Eigen::Vector3d t1 = state_window.at(imu_j).pos;
            Eigen::Matrix3d R1 = state_window.at(imu_j).rot * RIC;
            Eigen::Vector3d t = R0.transpose() * (t1 - t0);
            Eigen::Matrix3d R = R0.transpose() * R1;
            Eigen::Matrix<double, 3, 4> P;
            P.leftCols<3>() = R.transpose();
            P.rightCols<1>() = -R.transpose() * t;
            const cv::Point2f &unified_point = feature.points[i];
            Eigen::Vector3d f = Eigen::Vector3d(unified_point.x, unified_point.y, 1.0).normalized();
            svd_A.row(2 * i) = f[0] * P.row(2) - f[2] * P.row(0);
            svd_A.row(2 * i + 1) = f[1] * P.row(2) - f[2] * P.row(1);
        }
        Eigen::Vector4d svd_V = Eigen::JacobiSVD<Eigen::MatrixXd>(svd_A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        double svd_method = svd_V[2] / svd_V[3];

        if (svd_method < 0.1) {
            feature.inv_depth = -1;
        } else {
            feature.inv_depth = svd_method * scale; // todo tiemuhuaguo 这里是乘还是除？？
        }
    }
    return true;
}
