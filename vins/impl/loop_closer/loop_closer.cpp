#include <sys/time.h>

#include "loop_closer.h"
#include "impl/vins_utils.h"
#include "vins_logic.h"
#include "keyframe.h"
#include "impl/loop_relative_pos.h"
#include "impl/loop_detector.h"

using namespace vins;
using namespace DVision;
using namespace DBoW2;
using namespace Eigen;

class AngleManifoldPi {
public:
    template<typename T>
    bool operator()(const T *first, const T *second, T *result) const {
        *result = utils::normalizeAnglePi(*first + *second);
        return true;
    }

    template<typename T>
    bool Plus(const T *x, const T *delta, T *x_plus_delta) const {
        *x_plus_delta = utils::normalizeAnglePi(*x + *delta);
        return true;
    }

    template<typename T>
    bool Minus(const T *x, const T *delta, T *x_plus_delta) const {
        *x_plus_delta = utils::normalizeAnglePi(*x - *delta);
        return true;
    }

    static ceres::Manifold *Create() {
        return (new ceres::AutoDiffManifold<AngleManifoldPi, 1, 1>);
    }
};

struct Edge4Dof {
    Edge4Dof(Vector3d relative_t, double relative_yaw, double pitch_i, double roll_i, double yaw_weight = 1.0)
            : relative_t_(std::move(relative_t)), relative_yaw(relative_yaw), pitch_i(pitch_i), roll_i(roll_i) {
        yaw_weight_ = yaw_weight;
    }

    template<class T>
    bool operator()(const T *const yaw_i, const T *ti, const T *yaw_j, const T *tj, T *residuals) const {
        typedef Matrix<T, 3, 3> Mat3T;
        typedef Matrix<T, 3, 1> Vec3T;
        Vec3T t_w_ij;
        utils::arrayMinus(tj, ti, t_w_ij.data(), 3);
        Vec3T euler(yaw_i[0], (T) pitch_i, (T) roll_i);
        Mat3T w_R_i = utils::ypr2rot(euler);
        Vec3T t_i_ij = w_R_i.transpose() * t_w_ij;
        Vec3T t((T) relative_t_(0), (T) relative_t_(1), (T) relative_t_(2));
        utils::arrayMinus(t_i_ij.data(), t.data(), residuals, 3);
        residuals[3] = utils::normalizeAngle180(yaw_j[0] - yaw_i[0] - T(relative_yaw)) * yaw_weight_;
        return true;
    }

    static ceres::CostFunction *Create(const Vector3d &relative_t, const double relative_yaw,
                                       const double pitch_i, const double roll_i) {
        return (new ceres::AutoDiffCostFunction<Edge4Dof, 4, 1, 3, 1, 3>(
                new Edge4Dof(relative_t, relative_yaw, pitch_i, roll_i)));
    }

    Vector3d relative_t_;
    double relative_yaw, pitch_i, roll_i;
    double yaw_weight_;
};

LoopCloser::LoopCloser() {
    std::thread(&LoopCloser::optimize4DoF, this).detach();
}

void LoopCloser::addKeyFrame(KeyFrameUniPtr kf_ptr) {
    Synchronized(key_frame_buffer_mutex_) {
        key_frame_buffer_.emplace_back(std::move(kf_ptr));
    }
}

bool LoopCloser::findLoop(KeyFrame &kf, LoopMatchInfo &info) {
    int peer_loop_id = loop_detector_->detectSimilarDescriptor(kf.external_descriptors_, key_frame_list_.size());
    loop_detector_->addDescriptors(kf.external_descriptors_);
    if (peer_loop_id == -1) {
        return false;
    }
    std::vector<uint8_t> status;
    std::vector<cv::Point2f> old_frame_pts2d;
    bool succ = buildLoopRelation(*key_frame_list_[peer_loop_id], peer_loop_id, kf, status, old_frame_pts2d);
    if (!succ) { return false; }
    for (int i = 0; i < kf.points_.size(); ++i) {
        if (status[i]) {
            info.feature_ids.emplace_back(kf.feature_ids_[i]);
            info.peer_pts.emplace_back(old_frame_pts2d[i]);
        }
    }
    return true;
}

[[noreturn]] void LoopCloser::optimize4DoF() {
    while (true) {
        struct timeval tv1{}, tv2{};
        gettimeofday(&tv1, nullptr);
        optimize4DoFImpl();
        gettimeofday(&tv2, nullptr);
        int cost_us = (tv2.tv_sec - tv1.tv_sec) * 1000000 + tv2.tv_usec - tv1.tv_usec;
        if (cost_us < 1 * 1000) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    }
}

void LoopCloser::optimize4DoFImpl() {
    std::vector<KeyFrameUniPtr> tmp_key_frame_buffer;
    Synchronized(key_frame_buffer_mutex_) {
        std::swap(key_frame_buffer_, tmp_key_frame_buffer);
    }
    utils::insert_move_wrapper(key_frame_list_, tmp_key_frame_buffer);

    // todo 这里是ID还是idx？
    if (key_frame_list_.empty()) {
        return;
    }
    loop_interval_upper_bound_ = key_frame_list_.back()->loop_relative_pose_.peer_frame_id;

    if (loop_interval_upper_bound_ == -1) { return; }

    int max_length = loop_interval_upper_bound_ + 1;

    // w^t_i   w^q_i
    Vector3d t_array[max_length];
    Matrix3d r_array[max_length];
    Vector3d euler_array[max_length];

    ceres::Problem problem;
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    options.max_num_iterations = 5;
    ceres::Solver::Summary summary;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
    ceres::Manifold *angle_manifold_pi = AngleManifoldPi::Create();

    for (int frame_idx = loop_interval_lower_bound_; frame_idx <= loop_interval_upper_bound_; ++frame_idx) {
        const KeyFrame& kf = *key_frame_list_[frame_idx];
        kf.getVioPose(t_array[frame_idx], r_array[frame_idx]);
        euler_array[frame_idx] = utils::rot2ypr(r_array[frame_idx]);

        problem.AddParameterBlock(euler_array[frame_idx].data(), 1, angle_manifold_pi);
        problem.AddParameterBlock(t_array[frame_idx].data(), 3);

        problem.SetParameterBlockConstant(euler_array[frame_idx].data());
        problem.SetParameterBlockConstant(t_array[frame_idx].data());

        //add sequential edge
        for (int j = 1; j < 5 && frame_idx - j >= 0; j++) {
            int peer_id = frame_idx - j;
            Vector3d peer_euler = euler_array[peer_id];
            Vector3d relative_t = r_array[peer_id].transpose() * (t_array[frame_idx] - t_array[peer_id]);
            double relative_yaw = euler_array[frame_idx].x() - euler_array[peer_id].x();
            ceres::CostFunction *cost_function =
                    Edge4Dof::Create(relative_t, relative_yaw, peer_euler.y(), peer_euler.z());
            problem.AddResidualBlock(cost_function, nullptr,
                                     euler_array[peer_id].data(), t_array[peer_id].data(),
                                     euler_array[frame_idx].data(), t_array[frame_idx].data());
        }

        //add loop edge
        if (kf.loop_relative_pose_.peer_frame_id != -1) {
            int peer_frame_id = kf.loop_relative_pose_.peer_frame_id;
            assert(peer_frame_id >= loop_interval_lower_bound_);
            Vector3d peer_euler = utils::rot2ypr(r_array[peer_frame_id]);
            Vector3d relative_t = kf.loop_relative_pose_.relative_pos;
            double relative_yaw = kf.loop_relative_pose_.relative_yaw;
            ceres::CostFunction *cost_function =
                    Edge4Dof::Create(relative_t, relative_yaw, peer_euler.y(), peer_euler.z());
            problem.AddResidualBlock(cost_function, loss_function,
                                     euler_array[peer_frame_id].data(), t_array[peer_frame_id].data(),
                                     euler_array[frame_idx].data(), t_array[frame_idx].data());
        }
    }

    ceres::Solve(options, &problem, &summary);

    const KeyFrame &last_loop_kf = *key_frame_list_[loop_interval_upper_bound_];
    KeyFrame::calculatePoseRotDrift(t_array[loop_interval_upper_bound_], euler_array[loop_interval_upper_bound_],
                                    last_loop_kf.vio_T_i_w_, utils::rot2ypr(last_loop_kf.vio_R_i_w_),
                                    t_drift, r_drift);

    for (int frame_idx = loop_interval_lower_bound_; frame_idx <= loop_interval_upper_bound_; ++frame_idx) {
        Vector3d t = t_array[frame_idx];
        Matrix3d r = utils::ypr2rot(euler_array[frame_idx]);
        key_frame_list_[frame_idx]->updateLoopedPose(t, r);
    }

    for (int frame_idx = loop_interval_upper_bound_ + 1; frame_idx < key_frame_list_.size(); ++frame_idx) {
        key_frame_list_[frame_idx]->updatePoseByDrift(t_drift, r_drift);
    }

    handleDriftCalibration(t_drift, r_drift);
}
