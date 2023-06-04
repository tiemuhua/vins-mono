#include "loop_closer.h"
#include "../vins_utils.h"

using namespace vins;
using namespace DVision;
using namespace DBoW2;
using namespace Eigen;

class AngleManifold {
public:

    template<typename T>
    bool operator()(const T *theta_radians, const T *delta_theta_radians,
                    T *theta_radians_plus_delta) const {
        *theta_radians_plus_delta =
                utils::normalizeAngle180(*theta_radians + *delta_theta_radians);

        return true;
    }

    template <typename T>
    bool Plus(const T *x, const T *delta, T *x_plus_delta) const {
        *x_plus_delta = utils::normalizeAngle180(*x + *delta);
        return true;
    }

    template <typename T>
    bool Minus(const T *x, const T *delta, T *x_plus_delta) const {
        *x_plus_delta = utils::normalizeAngle180(*x - *delta);
        return true;
    }

    static ceres::Manifold *Create() {
        return (new ceres::AutoDiffManifold<AngleManifold, 1, 1>);
    }
};

struct FourDOFError {
    FourDOFError(Vector3d t, double relative_yaw, double pitch_i, double roll_i)
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
        return (new ceres::AutoDiffCostFunction<FourDOFError, 4, 1, 3, 1, 3>(
                new FourDOFError(t, relative_yaw, pitch_i, roll_i)));
    }

    Vector3d t_;
    double relative_yaw, pitch_i, roll_i;

};

struct FourDOFWeightError {
    FourDOFWeightError(Vector3d t, double relative_yaw, double pitch_i, double roll_i)
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
        return (new ceres::AutoDiffCostFunction<
                FourDOFWeightError, 4, 1, 3, 1, 3>(
                new FourDOFWeightError(t, relative_yaw, pitch_i, roll_i)));
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
    global_index = 0;
    sequence_cnt = 0;
    sequence_loop.push_back(false);
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
    if (sequence_cnt != cur_kf->sequence) {
        sequence_cnt++;
        sequence_loop.push_back(false);
        w_t_vio = Eigen::Vector3d(0, 0, 0);
        w_r_vio = Eigen::Matrix3d::Identity();
        m_drift.lock();
        t_drift = Eigen::Vector3d(0, 0, 0);
        r_drift = Eigen::Matrix3d::Identity();
        m_drift.unlock();
    }

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

            // shift vio pose of whole sequence to the world frame
            if (old_kf->sequence != cur_kf->sequence && sequence_loop[cur_kf->sequence] == 0) {
                w_r_vio = shift_r;
                w_t_vio = shift_t;
                vio_P_cur = w_r_vio * vio_P_cur + w_t_vio;
                vio_R_cur = w_r_vio * vio_R_cur;
                cur_kf->updateVioPose(vio_P_cur, vio_R_cur);
                for (KeyFrame* key_frame: keyframelist_) {
                    if (key_frame->sequence == cur_kf->sequence) {
                        key_frame->getVioPose(vio_P_cur, vio_R_cur);
                        Vector3d vio_P_cur = w_r_vio * vio_P_cur + w_t_vio;
                        Matrix3d vio_R_cur = w_r_vio * vio_R_cur;
                        key_frame->updateVioPose(vio_P_cur, vio_R_cur);
                    }
                }
                sequence_loop[cur_kf->sequence] = true;
            }
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
    // put image into image_pool; for visualization
    cv::Mat compressed_image;
    //first query; then add this frame into database!
    QueryResults ret;
    db.query(keyframe->brief_descriptors, ret, 4, frame_index - 50);
    bool find_loop = false;
    cv::Mat loop_result;
    // a good match with its neighbour
    if (!ret.empty() && ret[0].Score > 0.05)
        for (unsigned int i = 1; i < ret.size(); i++) {
            if (ret[i].Score > 0.015) {
                find_loop = true;
            }

        }
    if (find_loop && frame_index > 50) {
        int min_index = -1;
        for (unsigned int i = 0; i < ret.size(); i++) {
            if (min_index == -1 || (ret[i].Id < min_index && ret[i].Score > 0.015))
                min_index = ret[i].Id;
        }
        return min_index;
    } else
        return -1;

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
        double sequence_array[max_length];

        ceres::Problem problem;
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.max_num_iterations = 5;
        ceres::Solver::Summary summary;
        ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
        ceres::Manifold *angle_manifold = AngleManifold::Create();

        for (int frame_id = first_looped_index; frame_id <= cur_looped_id; ++frame_id) {
            KeyFrame* kf = keyframelist_[frame_id];
            kf->getVioPose(t_array[frame_id], r_array[frame_id]);
            euler_array[frame_id] = utils::rot2ypr(r_array[frame_id]);
            sequence_array[frame_id] = kf->sequence;

            problem.AddParameterBlock(euler_array[frame_id].data(), 1, angle_manifold);
            problem.AddParameterBlock(t_array[frame_id].data(), 3);

            if (frame_id == first_looped_index || kf->sequence == 0) {
                problem.SetParameterBlockConstant(euler_array[frame_id].data());
                problem.SetParameterBlockConstant(t_array[frame_id].data());
            }

            //add edge
            for (int j = 1; j < 5; j++) {
                int cur_frame_id = frame_id - j;
                if (cur_frame_id < 0 || sequence_array[frame_id] != sequence_array[frame_id - j]) {
                    continue;
                }
                Vector3d euler_connected = euler_array[cur_frame_id];
                Vector3d relative_t = r_array[cur_frame_id].transpose() * (t_array[frame_id] - t_array[cur_frame_id]);
                double relative_yaw = euler_array[frame_id].x() - euler_array[cur_frame_id].x();
                ceres::CostFunction *cost_function =
                        FourDOFError::Create(relative_t, relative_yaw, euler_connected.y(), euler_connected.z());
                problem.AddResidualBlock(cost_function, nullptr,
                                         euler_array[cur_frame_id].data(), t_array[cur_frame_id].data(),
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
                        FourDOFWeightError::Create(relative_t, relative_yaw, peer_euler.y(), peer_euler.z());
                problem.AddResidualBlock(cost_function, loss_function,
                                         euler_array[peer_id].data(), t_array[peer_id].data(),
                                         euler_array[frame_id].data(), t_array[frame_id].data());
            }
        }
        m_keyframelist.unlock();

        ceres::Solve(options, &problem, &summary);

        m_keyframelist.lock();
        for (int frame_id = first_looped_index; frame_id < cur_looped_id; ++frame_id) {
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