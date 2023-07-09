#include "loop_closer.h"
#include "log.h"
#include "impl/feature_retriever.h"
#include "impl/loop_detector.h"
#include "vins/vins_utils.h"
#include "vins/vins_run_info.h"

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

    template <typename T>
    bool Plus(const T *x, const T *delta, T *x_plus_delta) const {
        *x_plus_delta = utils::normalizeAnglePi(*x + *delta);
        return true;
    }

    template <typename T>
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
        Vec3T euler(yaw_i[0], (T)pitch_i, (T)roll_i);
        Mat3T w_R_i = utils::ypr2rot(euler);
        Vec3T t_i_ij = w_R_i.transpose() * t_w_ij;
        Vec3T t((T)relative_t_(0), (T)relative_t_(1), (T)relative_t_(2));
        utils::arrayMinus(t_i_ij.data(), t.data(), residuals, 3);
        // todo tiemuhua 论文里面没有说明为什么loop edge这里要除10
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
    thread_optimize_ = std::thread(&LoopCloser::optimize4DoF, this);
}

LoopCloser::~LoopCloser() {
    thread_optimize_.join();
}

// todo 若使用sift等耗时较长的描述字，可将提取描述子过程放入单独任务队列执行
void LoopCloser::addKeyFrame(const double _time_stamp, const Vector3d &t, const Matrix3d &r, const cv::Mat &_image,
                             const std::vector<cv::Point3f> &_point_3d, const std::vector<cv::Point2f> &_point_2d_uv) {
    std::vector<cv::KeyPoint> key_points;
    for (auto & i : _point_2d_uv) {
        cv::KeyPoint key;
        key.pt = i;
        key_points.push_back(key);
    }
    std::vector<DVision::BRIEF::bitset> descriptors;
    brief_extractor_->m_brief.compute(_image, key_points, descriptors);

    const int fast_th = 20; // corner detector response threshold
    std::vector<cv::KeyPoint> external_key_points_un_normalized;
    cv::FAST(_image, external_key_points_un_normalized, fast_th, true);
    std::vector<DVision::BRIEF::bitset> external_descriptors;
    brief_extractor_->m_brief.compute(_image, external_key_points_un_normalized, external_descriptors);
    std::vector<cv::KeyPoint> external_key_pts2d;
    for (auto & keypoint : external_key_points_un_normalized) {
        Eigen::Vector3d tmp_p;
        RunInfo::Instance().camera_ptr->liftProjective(Eigen::Vector2d(keypoint.pt.x, keypoint.pt.y), tmp_p);
        cv::KeyPoint tmp_norm;
        tmp_norm.pt = cv::Point2f(tmp_p.x() / tmp_p.z(), tmp_p.y() / tmp_p.z());
        external_key_pts2d.push_back(tmp_norm);
    }
    auto cur_kf = std::make_shared<KeyFrame>(_time_stamp, t, r, _point_3d, descriptors, external_key_pts2d, external_descriptors);

    Synchronized(key_frame_buffer_mutex_) {
        key_frame_buffer_.emplace_back(cur_kf);
    }
}

bool LoopCloser::findLoop(const KeyFramePtr& cur_kf, int& peer_loop_id) {
    peer_loop_id = loop_detector_->detectLoop(cur_kf->external_descriptors_, key_frame_list_.size());
    loop_detector_->addDescriptors(cur_kf->external_descriptors_);
    if (peer_loop_id == -1) {
        return false;
    }
    return FeatureRetriever::calculate4DofLoopDrift(key_frame_list_[peer_loop_id], peer_loop_id, cur_kf);
}

[[noreturn]] void LoopCloser::optimize4DoF() {
    while (true) {
        std::chrono::milliseconds dura(2000);
        std::this_thread::sleep_for(dura);
        optimize4DoFImpl();
    }
}

void LoopCloser::optimize4DoFImpl() {
    std::vector<KeyFramePtr> tmp_key_frame_buffer;
    Synchronized(key_frame_buffer_mutex_) {
        std::swap(key_frame_buffer_, tmp_key_frame_buffer);
    }

    for (const KeyFramePtr& kf:tmp_key_frame_buffer) {
        kf->updatePoseByDrift(t_drift, r_drift);
        int peer_loop_id = -1;
        if (findLoop(kf, peer_loop_id)) {
            loop_interval_upper_bound_ = (int )key_frame_list_.size();
            if (loop_interval_lower_bound_ > peer_loop_id || loop_interval_lower_bound_ == -1) {
                loop_interval_lower_bound_ = peer_loop_id;
            }
        }
        key_frame_list_.emplace_back(kf);
    }

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

    for (int frame_id = loop_interval_lower_bound_; frame_id <= loop_interval_upper_bound_; ++frame_id) {
        auto kf = key_frame_list_[frame_id];
        kf->getVioPose(t_array[frame_id], r_array[frame_id]);
        euler_array[frame_id] = utils::rot2ypr(r_array[frame_id]);

        problem.AddParameterBlock(euler_array[frame_id].data(), 1, angle_manifold_pi);
        problem.AddParameterBlock(t_array[frame_id].data(), 3);

        problem.SetParameterBlockConstant(euler_array[frame_id].data());
        problem.SetParameterBlockConstant(t_array[frame_id].data());

        //add sequential edge
        for (int j = 1; j < 5 && frame_id - j >= 0; j++) {
            int peer_id = frame_id - j;
            Vector3d peer_euler = euler_array[peer_id];
            Vector3d relative_t = r_array[peer_id].transpose() * (t_array[frame_id] - t_array[peer_id]);
            double relative_yaw = euler_array[frame_id].x() - euler_array[peer_id].x();
            ceres::CostFunction *cost_function =
                    Edge4Dof::Create(relative_t, relative_yaw, peer_euler.y(), peer_euler.z());
            problem.AddResidualBlock(cost_function, nullptr,
                                     euler_array[peer_id].data(), t_array[peer_id].data(),
                                     euler_array[frame_id].data(), t_array[frame_id].data());
        }

        //add loop edge
        if (kf->loop_relative_pose_.peer_frame_id != -1) {
            int peer_frame_id = kf->loop_relative_pose_.peer_frame_id;
            assert(peer_frame_id >= loop_interval_lower_bound_);
            Vector3d peer_euler = utils::rot2ypr(r_array[peer_frame_id]);
            Vector3d relative_t = kf->loop_relative_pose_.relative_pos;
            double relative_yaw = kf->loop_relative_pose_.relative_yaw;
            ceres::CostFunction *cost_function =
                    Edge4Dof::Create(relative_t, relative_yaw, peer_euler.y(), peer_euler.z());
            problem.AddResidualBlock(cost_function, loss_function,
                                     euler_array[peer_frame_id].data(), t_array[peer_frame_id].data(),
                                     euler_array[frame_id].data(), t_array[frame_id].data());
        }
    }

    ceres::Solve(options, &problem, &summary);

    ConstKeyFramePtr last_loop_kf = key_frame_list_[loop_interval_upper_bound_];
    KeyFrame::calculatePoseRotDrift(t_array[loop_interval_upper_bound_], euler_array[loop_interval_upper_bound_],
                                    last_loop_kf->vio_T_i_w_, utils::rot2ypr(last_loop_kf->vio_R_i_w_),
                                    t_drift, r_drift);

    for (int frame_id = loop_interval_lower_bound_; frame_id <= loop_interval_upper_bound_; ++frame_id) {
        Vector3d t = t_array[frame_id];
        Matrix3d r = utils::ypr2rot(euler_array[frame_id]);
        key_frame_list_[frame_id]->updateLoopedPose(t, r);
    }

    for (int frame_id = loop_interval_upper_bound_ + 1; frame_id < key_frame_list_.size(); ++frame_id) {
        key_frame_list_[frame_id]->updatePoseByDrift(t_drift, r_drift);
    }

    // todo 更新滑动窗口中的位姿，vio中提供了关键帧之间的相对关系，利用drift更新位姿时应当利用相对位姿更新，而不是直接更新绝对位姿
}
