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

bool Initiate::initiate(RunInfo &run_info) {
    if (!isAccVariantBigEnough(run_info.frame_window)) {
        return false;
    }

    bool visual_succ = initiateByVisual((int )run_info.kf_state_window.size(),
                                        run_info.feature_window,
                                        run_info.frame_window);
    if (!visual_succ) {
        return false;
    }

    //.求解bg.
    Eigen::Vector3d bg = solveGyroBias(run_info.frame_window);
    if (bg.norm() > 1e4) {
        return false;
    }
    for (int i = 1; i < run_info.frame_window.size(); ++i) {
        run_info.frame_window[i].pre_integral_->rePredict(Eigen::Vector3d::Zero(), bg);
    }





    //.求解重力、尺度和速度，即与位移有关的一切未知参数.
    //.重力和初始化时的ba打包在一起了，无法单独求解ba。
    //.秦博假设初始化时的ba可以忽略不计，g.norm==9.81。todo 一加6T、iPhone12的ba有多少？秦博这个假设合理吗？.
    double scale;
    std::vector<Eigen::Vector3d> velocities;
    if (!solveGravityScaleVelocity(run_info.frame_window, run_info.tic, run_info.gravity, scale, velocities)) {
        return false;
    }

    assert(run_info.frame_window.front().is_key_frame_);
    Eigen::Matrix3d rot_diff = rotGravityToZAxis(run_info.gravity, run_info.frame_window.front().R);
    run_info.gravity = rot_diff * run_info.gravity;

    auto &state_window = run_info.kf_state_window;
    auto &all_frames = run_info.frame_window;
    auto &TIC = run_info.tic;
    auto RIC = run_info.ric;
    for (KeyFrameState & state : state_window) {
        state.bg = bg;
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
        run_info.pre_int_window.at(i)->rePredict(Eigen::Vector3d::Zero(), state_window.at(i).bg);
    }

    //triangulate on cam pose , no tic
    //计算特征点深度，initialStructure里面算出来的特征点三维坐标没有ba，也没有对齐惯导
    for (Feature &feature: run_info.feature_window) {
        if (!(feature.points.size() >= 2 && feature.start_kf_window_idx < run_info.kf_state_window.size() - 2))
            continue;

        Eigen::MatrixXd svd_A(2 * feature.points.size(), 4);

        int start_kf_idx = feature.start_kf_window_idx;
        Eigen::Vector3d t0 = state_window.at(start_kf_idx).pos;
        Eigen::Matrix3d R0 = state_window.at(start_kf_idx).rot * RIC;

        for (int i = 0; i < feature.points.size(); ++i) {
            int kf_idx = start_kf_idx + i;
            Eigen::Vector3d t1 = state_window.at(kf_idx).pos;
            Eigen::Matrix3d R1 = state_window.at(kf_idx).rot * RIC;
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
