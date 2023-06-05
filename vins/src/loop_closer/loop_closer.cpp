#include "loop_closer.h"
#include "../vins_utils.h"

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

    bool operator()(const double *const yaw_i, const double *ti, const double *yaw_j, const double *tj, double *residuals) const {
        Vector3d t_w_ij;
        utils::arrayMinus(tj, ti, t_w_ij.data(), 3);
        Matrix3d w_R_i = utils::ypr2rot({yaw_i[0], pitch_i, roll_i});
        Vector3d t_i_ij = w_R_i.transpose() * t_w_ij;
        utils::arrayMinus(t_i_ij.data(), t_.data(), residuals, 3);
        residuals[3] = utils::normalizeAngle180(yaw_j[0] - yaw_i[0] - relative_yaw);

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
    LoopEdge(Vector3d t, double relative_yaw, double pitch_i, double roll_i)
            : t_(std::move(t)), relative_yaw(relative_yaw), pitch_i(pitch_i), roll_i(roll_i) {
        weight = 1;
    }

    bool operator()(const double *const yaw_i, const double *ti, const double *yaw_j, const double *tj, double *residuals) const {
        Vector3d t_w_ij;
        utils::arrayMinus(tj, ti, t_w_ij.data(), 3);
        Matrix3d w_R_i = utils::ypr2rot({yaw_i[0], pitch_i, roll_i});
        Vector3d t_i_ij = w_R_i.transpose() * t_w_ij;
        utils::arrayMinus(t_i_ij.data(), t_.data(), residuals, 3);
        utils::arrayMultiply(residuals, residuals, weight, 3);
        residuals[3] = utils::normalizeAngle180((yaw_j[0] - yaw_i[0] - relative_yaw)) * weight / 10.0;
        return true;
    }

    static ceres::CostFunction *Create(const Vector3d &t,
                                       const double relative_yaw, const double pitch_i, const double roll_i) {
        return (new ceres::AutoDiffCostFunction<LoopEdge, 4, 1, 3, 1, 3>(
                new LoopEdge(t, relative_yaw, pitch_i, roll_i)));
    }

    Vector3d t_;
    double relative_yaw, pitch_i, roll_i;
    double weight;

};

LoopCloser::LoopCloser() {
    t_optimization = std::thread(&LoopCloser::optimize4DoF, this);
    earliest_loop_index = -1;
    t_drift = Eigen::Vector3d(0, 0, 0);
    yaw_drift = 0;
    r_drift = Eigen::Matrix3d::Identity();
    w_t_vio = Eigen::Vector3d(0, 0, 0);
    w_r_vio = Eigen::Matrix3d::Identity();
}

LoopCloser::~LoopCloser() {
    t_optimization.join();
}

void LoopCloser::loadVocabulary(const std::string &voc_path) {
    voc = new BriefVocabulary(voc_path);
    db.setVocabulary(*voc, false, 0);
}

void LoopCloser::addKeyFrame(KeyFrame *cur_kf, bool flag_detect_loop) {
    //shift to base frame
    Vector3d vio_P_cur;
    Matrix3d vio_R_cur;
    cur_kf->getVioPose(vio_P_cur, vio_R_cur);
    vio_P_cur = w_r_vio * vio_P_cur + w_t_vio;
    vio_R_cur = w_r_vio * vio_R_cur;
    cur_kf->updateVioPose(vio_P_cur, vio_R_cur);
    int loop_index = -1;
    if (flag_detect_loop) {
        loop_index = detectLoop(cur_kf, keyframelist_.size());
    }
    db.add(cur_kf->brief_descriptors);
    if (loop_index != -1) {
        //printf(" %d detect loop with %d \n", cur_kf->index, loop_index);
        KeyFrame *old_kf = keyframelist_[loop_index];

        if (cur_kf->findConnection(old_kf, loop_index)) {
            if (earliest_loop_index > loop_index || earliest_loop_index == -1) {
                earliest_loop_index = loop_index;
            }

            Vector3d w_P_old, vio_P_cur;
            Matrix3d w_R_old, vio_R_cur;
            old_kf->getVioPose(w_P_old, w_R_old);
            cur_kf->getVioPose(vio_P_cur, vio_R_cur);

            const Vector3d &relative_pos = cur_kf->loop_info_.relative_pos;
            const Matrix3d &relative_rot = cur_kf->loop_info_.relative_rot;
            Vector3d w_P_cur = w_R_old * relative_pos + w_P_old;
            Matrix3d w_R_cur = w_R_old * relative_rot;
            double shift_yaw = utils::rot2ypr(w_R_cur).x() - utils::rot2ypr(vio_R_cur).x();
            Matrix3d shift_r = utils::ypr2rot(Vector3d(shift_yaw, 0, 0));
            Vector3d shift_t = w_P_cur - w_R_cur * vio_R_cur.transpose() * vio_P_cur;

            m_optimize_buf.lock();
            optimize_buf.push(keyframelist_.size());
            m_optimize_buf.unlock();
        }
    }
    m_keyframelist.lock();
    Vector3d P;
    Matrix3d R;
    cur_kf->getVioPose(P, R);
    P = r_drift * P + t_drift;
    R = r_drift * R;
    cur_kf->updatePose(P, R);
    keyframelist_.push_back(cur_kf);
    m_keyframelist.unlock();
}

int LoopCloser::detectLoop(KeyFrame *keyframe, int frame_index) {
    if (frame_index < 50) {
        return -1;
    }
    QueryResults ret;
    db.query(keyframe->brief_descriptors, ret, 4, frame_index - 50);
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

void LoopCloser::optimize4DoF() {
    while (true) {
        std::chrono::milliseconds dura(2000);
        std::this_thread::sleep_for(dura);

        int cur_looped_id = -1;
        int first_looped_index = -1;
        m_optimize_buf.lock();
        while (!optimize_buf.empty()) {
            cur_looped_id = optimize_buf.front();
            first_looped_index = earliest_loop_index;
            optimize_buf.pop();
        }
        m_optimize_buf.unlock();
        if (cur_looped_id == -1) { continue; }
        m_keyframelist.lock();
        KeyFrame *cur_kf = keyframelist_[cur_looped_id];

        int max_length = cur_looped_id + 1;

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

        for (int frame_id = first_looped_index; frame_id <= cur_looped_id; ++frame_id) {
            KeyFrame* kf = keyframelist_[frame_id];
            kf->getVioPose(t_array[frame_id], r_array[frame_id]);
            euler_array[frame_id] = utils::rot2ypr(r_array[frame_id]);

            problem.AddParameterBlock(euler_array[frame_id].data(), 1, angle_manifold_pi);
            problem.AddParameterBlock(t_array[frame_id].data(), 3);

            problem.SetParameterBlockConstant(euler_array[frame_id].data());
            problem.SetParameterBlockConstant(t_array[frame_id].data());

            //add edge
            for (int j = 1; j < 5; j++) {
                int peer_id = frame_id - j;
                if (peer_id < 0) {
                    continue;
                }
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
            if (kf->has_loop) {
                assert(kf->loop_peer_id_ >= first_looped_index);
                int peer_id = kf->loop_peer_id_;
                Vector3d peer_euler = utils::rot2ypr(r_array[peer_id]);
                Vector3d relative_t = kf->getLoopRelativeT();
                double relative_yaw = kf->getLoopRelativeYaw();
                ceres::CostFunction *cost_function =
                        LoopEdge::Create(relative_t, relative_yaw, peer_euler.y(), peer_euler.z());
                problem.AddResidualBlock(cost_function, loss_function,
                                         euler_array[peer_id].data(), t_array[peer_id].data(),
                                         euler_array[frame_id].data(), t_array[frame_id].data());
            }
        }
        m_keyframelist.unlock();

        ceres::Solve(options, &problem, &summary);

        m_keyframelist.lock();
        for (int frame_id = first_looped_index; frame_id <= cur_looped_id; ++frame_id) {
            Vector3d t = t_array[frame_id];
            Matrix3d r = utils::ypr2rot(euler_array[frame_id]);
            keyframelist_[frame_id]->updatePose(t, r);
        }

        Vector3d cur_t, vio_t;
        Matrix3d cur_r, vio_r;
        cur_kf->getPose(cur_t, cur_r);
        cur_kf->getVioPose(vio_t, vio_r);
        m_drift.lock();
        yaw_drift = utils::rot2ypr(cur_r).x() - utils::rot2ypr(vio_r).x();
        r_drift = utils::rot2ypr(Vector3d(yaw_drift, 0, 0));
        t_drift = cur_t - r_drift * vio_t;
        m_drift.unlock();

        for (int frame_id = cur_looped_id + 1; frame_id < keyframelist_.size(); ++frame_id) {
            Vector3d P;
            Matrix3d R;
            keyframelist_[frame_id]->getVioPose(P, R);
            P = r_drift * P + t_drift;
            R = r_drift * R;
            keyframelist_[frame_id]->updatePose(P, R);
        }
        m_keyframelist.unlock();
    }
}

extern int FAST_RELOCALIZATION;

void LoopCloser::updateKeyFrameLoop(int index, Eigen::Matrix<double, 8, 1> &_loop_info) {
    KeyFrame *kf = keyframelist_[index];
    kf->updateLoop(_loop_info);
    if (abs(_loop_info(7)) < 30.0 && Vector3d(_loop_info(0), _loop_info(1), _loop_info(2)).norm() < 20.0) {
        if (FAST_RELOCALIZATION) {
            KeyFrame *old_kf = keyframelist_[kf->loop_peer_id_];
            Vector3d w_P_old, w_P_cur, vio_P_cur;
            Matrix3d w_R_old, w_R_cur, vio_R_cur;
            old_kf->getPose(w_P_old, w_R_old);
            kf->getVioPose(vio_P_cur, vio_R_cur);

            Vector3d relative_t;
            Quaterniond relative_q;
            relative_t = kf->getLoopRelativeT();
            relative_q = (kf->getLoopRelativeQ()).toRotationMatrix();
            w_P_cur = w_R_old * relative_t + w_P_old;
            w_R_cur = w_R_old * relative_q;
            double shift_yaw;
            Matrix3d shift_r;
            Vector3d shift_t;
            shift_yaw = utils::rot2ypr(w_R_cur).x() - utils::rot2ypr(vio_R_cur).x();
            shift_r = utils::ypr2rot(Vector3d(shift_yaw, 0, 0));
            shift_t = w_P_cur - w_R_cur * vio_R_cur.transpose() * vio_P_cur;

            m_drift.lock();
            yaw_drift = shift_yaw;
            r_drift = shift_r;
            t_drift = shift_t;
            m_drift.unlock();
        }
    }
}