#include "loop_closer.h"
#include "../vins_utils.h"

using namespace vins;
using namespace DVision;
using namespace DBoW2;
using namespace Eigen;

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
    cur_kf->index = global_index;
    global_index++;
    int loop_index = -1;
    if (flag_detect_loop) {
        loop_index = detectLoop(cur_kf, cur_kf->index);
    } else {
        _addKeyFrameIntoVoc(cur_kf);
    }
    if (loop_index != -1) {
        //printf(" %d detect loop with %d \n", cur_kf->index, loop_index);
        KeyFrame *old_kf = getKeyFrame(loop_index);

        if (cur_kf->findConnection(old_kf)) {
            if (earliest_loop_index > loop_index || earliest_loop_index == -1)
                earliest_loop_index = loop_index;

            Vector3d w_P_old, vio_P_cur;
            Matrix3d w_R_old, vio_R_cur;
            old_kf->getVioPose(w_P_old, w_R_old);
            cur_kf->getVioPose(vio_P_cur, vio_R_cur);

            Vector3d relative_t = cur_kf->getLoopRelativeT();
            Quaterniond relative_q;
            relative_q = (cur_kf->getLoopRelativeQ()).toRotationMatrix();
            Vector3d w_P_cur = w_R_old * relative_t + w_P_old;
            Matrix3d w_R_cur = w_R_old * relative_q;
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
            optimize_buf.push(cur_kf->index);
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

KeyFrame *LoopCloser::getKeyFrame(int index) {
    for (KeyFrame* key_frame: keyframelist_) {
        if (key_frame->index == index) {
            return key_frame;
        }
    }
    return nullptr;
}

int LoopCloser::detectLoop(KeyFrame *keyframe, int frame_index) {
    // put image into image_pool; for visualization
    cv::Mat compressed_image;
    //first query; then add this frame into database!
    QueryResults ret;
    db.query(keyframe->brief_descriptors, ret, 4, frame_index - 50);

    db.add(keyframe->brief_descriptors);
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

void LoopCloser::_addKeyFrameIntoVoc(KeyFrame *keyframe) {
    // put image into image_pool; for visualization
    db.add(keyframe->brief_descriptors);
}

void LoopCloser::optimize4DoF() {
    while (true) {
        std::chrono::milliseconds dura(2000);
        std::this_thread::sleep_for(dura);

        int cur_index = -1;
        int first_looped_index = -1;
        m_optimize_buf.lock();
        while (!optimize_buf.empty()) {
            cur_index = optimize_buf.front();
            first_looped_index = earliest_loop_index;
            optimize_buf.pop();
        }
        m_optimize_buf.unlock();
        if (cur_index == -1) { continue; }
        m_keyframelist.lock();
        KeyFrame *cur_kf = getKeyFrame(cur_index);

        int max_length = cur_index + 1;

        // w^t_i   w^q_i
        double t_array[max_length][3];
        Quaterniond q_array[max_length];
        double euler_array[max_length][3];
        double sequence_array[max_length];

        ceres::Problem problem;
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.max_num_iterations = 5;
        ceres::Solver::Summary summary;
        ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
        ceres::Manifold *angle_local_parameterization = AngleManifold::Create();

        list<KeyFrame *>::iterator it;

        int i = 0;
        for (it = keyframelist_.begin(); it != keyframelist_.end(); it++) {
            if ((*it)->index < first_looped_index) { continue; }
            (*it)->local_index = i;
            Matrix3d tmp_r;
            Vector3d tmp_t;
            (*it)->getVioPose(tmp_t, tmp_r);
            Quaterniond tmp_q(tmp_r);
            t_array[i][0] = tmp_t(0);
            t_array[i][1] = tmp_t(1);
            t_array[i][2] = tmp_t(2);
            q_array[i] = tmp_q;

            Vector3d euler_angle = utils::rot2ypr(tmp_q.toRotationMatrix());
            euler_array[i][0] = euler_angle.x();
            euler_array[i][1] = euler_angle.y();
            euler_array[i][2] = euler_angle.z();

            sequence_array[i] = (*it)->sequence;

            problem.AddParameterBlock(euler_array[i], 1, angle_local_parameterization);
            problem.AddParameterBlock(t_array[i], 3);

            if ((*it)->index == first_looped_index || (*it)->sequence == 0) {
                problem.SetParameterBlockConstant(euler_array[i]);
                problem.SetParameterBlockConstant(t_array[i]);
            }

            //add edge
            for (int j = 1; j < 5; j++) {
                if (i - j >= 0 && sequence_array[i] == sequence_array[i - j]) {
                    Vector3d euler_connected = utils::rot2ypr(q_array[i - j].toRotationMatrix());
                    Vector3d relative_t(t_array[i][0] - t_array[i - j][0],
                                        t_array[i][1] - t_array[i - j][1],
                                        t_array[i][2] - t_array[i - j][2]);
                    relative_t = q_array[i - j].inverse() * relative_t;
                    double relative_yaw = euler_array[i][0] - euler_array[i - j][0];
                    ceres::CostFunction *cost_function =
                            FourDOFError::Create(relative_t.x(), relative_t.y(), relative_t.z(),
                                                 relative_yaw, euler_connected.y(), euler_connected.z());
                    problem.AddResidualBlock(cost_function, nullptr,
                                             euler_array[i - j], t_array[i - j],
                                             euler_array[i], t_array[i]);
                }
            }

            //add loop edge
            if ((*it)->has_loop) {
                assert((*it)->loop_index >= first_looped_index);
                int connected_index = getKeyFrame((*it)->loop_index)->local_index;
                Vector3d euler_connected = utils::rot2ypr(q_array[connected_index].toRotationMatrix());
                Vector3d relative_t = (*it)->getLoopRelativeT();
                double relative_yaw = (*it)->getLoopRelativeYaw();
                ceres::CostFunction *cost_function =
                        FourDOFWeightError::Create(relative_t.x(), relative_t.y(), relative_t.z(),
                                                   relative_yaw, euler_connected.y(), euler_connected.z());
                problem.AddResidualBlock(cost_function, loss_function,
                                         euler_array[connected_index], t_array[connected_index],
                                         euler_array[i], t_array[i]);

            }

            if ((*it)->index == cur_index)
                break;
            i++;
        }
        m_keyframelist.unlock();

        ceres::Solve(options, &problem, &summary);

        m_keyframelist.lock();
        i = 0;
        for (it = keyframelist_.begin(); it != keyframelist_.end(); it++) {
            if ((*it)->index < first_looped_index)
                continue;
            Quaterniond tmp_q;
            tmp_q = utils::ypr2rot(Vector3d(euler_array[i][0], euler_array[i][1], euler_array[i][2]));
            Vector3d tmp_t = Vector3d(t_array[i][0], t_array[i][1], t_array[i][2]);
            Matrix3d tmp_r = tmp_q.toRotationMatrix();
            (*it)->updatePose(tmp_t, tmp_r);

            if ((*it)->index == cur_index)
                break;
            i++;
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

        it++;
        for (; it != keyframelist_.end(); it++) {
            Vector3d P;
            Matrix3d R;
            (*it)->getVioPose(P, R);
            P = r_drift * P + t_drift;
            R = r_drift * R;
            (*it)->updatePose(P, R);
        }
        m_keyframelist.unlock();
    }
}

void LoopCloser::updateKeyFrameLoop(int index, Eigen::Matrix<double, 8, 1> &_loop_info) {
    KeyFrame *kf = getKeyFrame(index);
    kf->updateLoop(_loop_info);
    if (abs(_loop_info(7)) < 30.0 && Vector3d(_loop_info(0), _loop_info(1), _loop_info(2)).norm() < 20.0) {
        if (FAST_RELOCALIZATION) {
            KeyFrame *old_kf = getKeyFrame(kf->loop_index);
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