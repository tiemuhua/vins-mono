#include "loop_closer.h"
#include "log.h"
#include "match_frame.h"
#include "vins/vins_utils.h"

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

struct SequentialEdge {
    SequentialEdge(Vector3d t, double relative_yaw, double pitch_i, double roll_i)
            : t_(std::move(t)), relative_yaw(relative_yaw), pitch_i(pitch_i), roll_i(roll_i) {}

    //.T为ceres::Jet<double, 8>，Jet.a为数据，Jet.v为导数.
    template<class T>
    bool operator()(const T *const yaw_i, const T *ti, const T *yaw_j, const T *tj, T *residuals) const {
        typedef Matrix<T, 3, 3> Mat3T;
        typedef Matrix<T, 3, 1> Vec3T;
        Vec3T t_w_ij;
        utils::arrayMinus(tj, ti, t_w_ij.data(), 3);
        Vec3T euler(yaw_i[0], (T)pitch_i, (T)roll_i);
        Mat3T w_R_i = utils::ypr2rot(euler);
        Vec3T t_i_ij = w_R_i.transpose() * t_w_ij;
        Vec3T t((T)t_(0), (T)t_(1), (T)t_(2));
        utils::arrayMinus(t_i_ij.data(), t.data(), residuals, 3);
        residuals[3] = utils::normalizeAngle180(yaw_j[0] - yaw_i[0] - T(relative_yaw));
        return true;
    }

    static ceres::CostFunction *Create(const Vector3d &t,
                                       const double relative_yaw, const double pitch_i, const double roll_i) {
        return (new ceres::AutoDiffCostFunction<SequentialEdge, 4, 1, 3, 1, 3>(
                new SequentialEdge(t, relative_yaw, pitch_i, roll_i)));
    }

    Vector3d t_;
    double relative_yaw, pitch_i, roll_i;
};

struct LoopEdge {
    LoopEdge(Vector3d relative_t, double relative_yaw, double pitch_i, double roll_i)
            : relative_t_(std::move(relative_t)), relative_yaw(relative_yaw), pitch_i(pitch_i), roll_i(roll_i) {
        weight = 1;
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
        // todo tiemuhua 论文里面没有说明这里为什么要除10
        residuals[3] = utils::normalizeAngle180(yaw_j[0] - yaw_i[0] - T(relative_yaw)) / 10.0;
        return true;
    }

    static ceres::CostFunction *Create(const Vector3d &relative_t, const double relative_yaw,
                                       const double pitch_i, const double roll_i) {
        return (new ceres::AutoDiffCostFunction<LoopEdge, 4, 1, 3, 1, 3>(
                new LoopEdge(relative_t, relative_yaw, pitch_i, roll_i)));
    }

    Vector3d relative_t_;
    double relative_yaw, pitch_i, roll_i;
    double weight;
};

LoopCloser::LoopCloser() {
    thread_optimize_ = std::thread(&LoopCloser::optimize4DoF, this);
}

LoopCloser::~LoopCloser() {
    thread_optimize_.join();
}

void LoopCloser::loadVocabulary(const std::string &voc_path) {
    voc = new BriefVocabulary(voc_path);
    db.setVocabulary(*voc, false, 0);
}

void LoopCloser::addKeyFrame(const KeyFramePtr& cur_kf) {
    Synchronized(key_frame_buffer_mutex_) {
        key_frame_buffer_.emplace_back(cur_kf);
    }
}

void LoopCloser::addKeyFrame(const KeyFramePtr& cur_kf, bool flag_detect_loop) {
    int peer_loop_id = -1;
    if (flag_detect_loop) {
        peer_loop_id = _detectLoop(cur_kf, key_frame_list_.size());
    }
    db.add(cur_kf->external_descriptors_);
    if (peer_loop_id == -1) {
        return;
    }
    LoopInfo loop_info;
    bool find_loop = MatchFrame::findLoop(key_frame_list_[peer_loop_id], peer_loop_id, cur_kf, loop_info);
    if (!find_loop) {
        return;
    }
    cur_kf->loop_info_ = loop_info;
    loop_interval_upper_bound_ = key_frame_list_.size() - 1;
    if (loop_interval_lower_bound_ > peer_loop_id || loop_interval_lower_bound_ == -1) {
        loop_interval_lower_bound_ = peer_loop_id;
    }
}

int LoopCloser::_detectLoop(ConstKeyFramePtr& keyframe, int frame_index) const {
    if (frame_index < 50) {
        return -1;
    }
    QueryResults ret;
    db.query(keyframe->external_descriptors_, ret, 4, frame_index - 50);
    cv::Mat loop_result;
    // a good match with its neighbour
    if (ret.size() < 2 || ret[0].Score < 0.05) {
        return -1;
    }
    bool find_loop = false;
    for (unsigned int i = 1; i < ret.size(); i++) {
        if (ret[i].Score > 0.015) {
            find_loop = true;
        }
    }
    if (!find_loop) {
        return -1;
    }
    int min_index = 0x3f3f3f3f;
    assert(ret.size() < min_index);
    for (unsigned int i = 1; i < ret.size(); i++) {
        if (ret[i].Id < min_index && ret[i].Score > 0.015)
            min_index = ret[i].Id;
    }
    return min_index;
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
        tmp_key_frame_buffer = std::move(key_frame_buffer_);
    }

    for (const KeyFramePtr& kf:tmp_key_frame_buffer) {
        kf->updatePoseByDrift(t_drift, r_drift);
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
                    SequentialEdge::Create(relative_t, relative_yaw, peer_euler.y(), peer_euler.z());
            problem.AddResidualBlock(cost_function, nullptr,
                                     euler_array[peer_id].data(), t_array[peer_id].data(),
                                     euler_array[frame_id].data(), t_array[frame_id].data());
        }

        //add loop edge
        if (kf->loop_info_.peer_frame_id != -1) {
            int peer_frame_id = kf->loop_info_.peer_frame_id;
            assert(peer_frame_id >= loop_interval_lower_bound_);
            Vector3d peer_euler = utils::rot2ypr(r_array[peer_frame_id]);
            Vector3d relative_t = kf->loop_info_.relative_pos;
            double relative_yaw = kf->loop_info_.relative_yaw;
            ceres::CostFunction *cost_function =
                    LoopEdge::Create(relative_t, relative_yaw, peer_euler.y(), peer_euler.z());
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
}
