#include "initial_sfm.h"

Vector3d triangulatePoint(const Eigen::Matrix<double, 3, 4> &Pose0, const Eigen::Matrix<double, 3, 4> &Pose1,
                          const cv::Point2f &point0, const cv::Point2f &point1) {
    Matrix4d design_matrix = Matrix4d::Zero();
    design_matrix.row(0) = point0.x * Pose0.row(2) - Pose0.row(0);
    design_matrix.row(1) = point0.y * Pose0.row(2) - Pose0.row(1);
    design_matrix.row(2) = point1.x * Pose1.row(2) - Pose1.row(0);
    design_matrix.row(3) = point1.y * Pose1.row(2) - Pose1.row(1);
    Vector4d triangulated_point =
            design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
    return triangulated_point.block<3,1>(0,0) / triangulated_point(3);
}


bool solveFrameByPnP(Matrix3d &R_initial, Vector3d &P_initial, int i,
                     const vector<SFMFeature> &sfm_features) {
    vector<cv::Point2f> pts_2_vector;
    vector<cv::Point3f> pts_3_vector;
    for (const SFMFeature & sfm : sfm_features) {
        if (!sfm.state)
            continue;
        for (int k = 0; k < (int) sfm.observation.size(); k++) {
            if (sfm.observation[k].first == i) {
                cv::Point2f img_pts = sfm.observation[k].second;
                pts_2_vector.emplace_back(img_pts);
                pts_3_vector.emplace_back(sfm.position[0], sfm.position[1], sfm.position[2]);
                break;
            }
        }
    }
    if (int(pts_2_vector.size()) < 15) {
        printf("unstable features tracking, please slowly move you device!\n");
        if (int(pts_2_vector.size()) < 10)
            return false;
    }
    cv::Mat r, rvec, t, D, tmp_r;
    cv::eigen2cv(R_initial, tmp_r);
    cv::Rodrigues(tmp_r, rvec);
    cv::eigen2cv(P_initial, t);
    cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
    bool pnp_succ;
    pnp_succ = cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, true);
    if (!pnp_succ) {
        return false;
    }
    cv::Rodrigues(rvec, r);
    MatrixXd R_pnp;
    cv::cv2eigen(r, R_pnp);
    MatrixXd T_pnp;
    cv::cv2eigen(t, T_pnp);
    R_initial = R_pnp;
    P_initial = T_pnp;
    return true;

}

void triangulateTwoFrames(int frame0, Eigen::Matrix<double, 3, 4> &Pose0,
                          int frame1, Eigen::Matrix<double, 3, 4> &Pose1,
                          vector<SFMFeature> &sfm_features) {
    assert(frame0 != frame1);
    for (SFMFeature & sfm : sfm_features) {
        if (sfm.state)
            continue;
        bool has_0 = false, has_1 = false;
        cv::Point2f point0;
        cv::Point2f point1;
        for (auto & k : sfm.observation) {
            if (k.first == frame0) {
                point0 = k.second;
                has_0 = true;
            }
            if (k.first == frame1) {
                point1 = k.second;
                has_1 = true;
            }
        }
        if (has_0 && has_1) {
            Vector3d point_3d = triangulatePoint(Pose0, Pose1, point0, point1);
            sfm.state = true;
            sfm.position[0] = point_3d(0);
            sfm.position[1] = point_3d(1);
            sfm.position[2] = point_3d(2);
        }
    }
}

// 	 q w_R_cam t w_R_cam
//  c_rotation cam_R_w 
//  c_translation cam_R_w
// relative_q[i][j]  j_q_i
// relative_t[i][j]  j_t_ji  (j < i)
bool GlobalSFM::construct(int frame_num, Quaterniond *q, Vector3d *T, int l,
                          const Matrix3d &relative_R, const Vector3d &relative_T,
                          vector<SFMFeature> &sfm_features, map<int, Vector3d> &sfm_tracked_points) {
    // have relative_r relative_t
    // initial two view
    q[l].w() = 1;
    q[l].x() = 0;
    q[l].y() = 0;
    q[l].z() = 0;
    T[l].setZero();
    q[frame_num - 1] = q[l] * Quaterniond(relative_R);
    T[frame_num - 1] = relative_T;

    //rotate to cam frame
    Matrix3d c_Rotation[frame_num];
    Vector3d c_Translation[frame_num];
    Quaterniond c_Quaterniond[frame_num];
    double c_rotation[frame_num][4];
    double c_translation[frame_num][3];
    Eigen::Matrix<double, 3, 4> Pose[frame_num];

    c_Quaterniond[l] = q[l].inverse();
    c_Rotation[l] = c_Quaterniond[l].toRotationMatrix();
    c_Translation[l] = -1 * (c_Rotation[l] * T[l]);
    Pose[l].block<3, 3>(0, 0) = c_Rotation[l];
    Pose[l].block<3, 1>(0, 3) = c_Translation[l];

    c_Quaterniond[frame_num - 1] = q[frame_num - 1].inverse();
    c_Rotation[frame_num - 1] = c_Quaterniond[frame_num - 1].toRotationMatrix();
    c_Translation[frame_num - 1] = -1 * (c_Rotation[frame_num - 1] * T[frame_num - 1]);
    Pose[frame_num - 1].block<3, 3>(0, 0) = c_Rotation[frame_num - 1];
    Pose[frame_num - 1].block<3, 1>(0, 3) = c_Translation[frame_num - 1];


    //1: triangulate between l ----- frame_num - 1
    //2: solve pnp l + 1; triangulate l + 1 ------- frame_num - 1;
    for (int i = l; i < frame_num - 1; i++) {
        // solve pnp
        if (i > l) {
            Matrix3d R_initial = c_Rotation[i - 1];
            Vector3d P_initial = c_Translation[i - 1];
            if (!solveFrameByPnP(R_initial, P_initial, i, sfm_features))
                return false;
            c_Rotation[i] = R_initial;
            c_Translation[i] = P_initial;
            c_Quaterniond[i] = c_Rotation[i];
            Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
            Pose[i].block<3, 1>(0, 3) = c_Translation[i];
        }

        // triangulate point based on to solve pnp result
        triangulateTwoFrames(i, Pose[i], frame_num - 1, Pose[frame_num - 1], sfm_features);
    }
    //3: triangulate l-----l+1 l+2 ... frame_num -2
    for (int i = l + 1; i < frame_num - 1; i++)
        triangulateTwoFrames(l, Pose[l], i, Pose[i], sfm_features);
    //4: solve pnp l-1; triangulate l-1 ----- l
    //             l-2              l-2 ----- l
    for (int i = l - 1; i >= 0; i--) {
        //solve pnp
        Matrix3d R_initial = c_Rotation[i + 1];
        Vector3d P_initial = c_Translation[i + 1];
        if (!solveFrameByPnP(R_initial, P_initial, i, sfm_features))
            return false;
        c_Rotation[i] = R_initial;
        c_Translation[i] = P_initial;
        c_Quaterniond[i] = c_Rotation[i];
        Pose[i].block<3, 3>(0, 0) = c_Rotation[i];
        Pose[i].block<3, 1>(0, 3) = c_Translation[i];
        //triangulate
        triangulateTwoFrames(i, Pose[i], l, Pose[l], sfm_features);
    }
    //5: triangulate all others points
    for (SFMFeature& sfm: sfm_features) {
        if (sfm.state)
            continue;
        if (sfm.observation.size() >= 2) {
            int frame_0 = sfm.observation[0].first;
            cv::Point2f point0 = sfm.observation[0].second;
            int frame_1 = sfm.observation.back().first;
            cv::Point2f point1 = sfm.observation.back().second;
            Vector3d point_3d = triangulatePoint(Pose[frame_0], Pose[frame_1], point0, point1);
            sfm.state = true;
            sfm.position[0] = point_3d(0);
            sfm.position[1] = point_3d(1);
            sfm.position[2] = point_3d(2);
        }
    }

    //full BA
    ceres::Problem problem;
    ceres::Manifold *local_parameterization = new ceres::QuaternionManifold();
    for (int i = 0; i < frame_num; i++) {
        //double array for ceres
        c_translation[i][0] = c_Translation[i].x();
        c_translation[i][1] = c_Translation[i].y();
        c_translation[i][2] = c_Translation[i].z();
        c_rotation[i][0] = c_Quaterniond[i].w();
        c_rotation[i][1] = c_Quaterniond[i].x();
        c_rotation[i][2] = c_Quaterniond[i].y();
        c_rotation[i][3] = c_Quaterniond[i].z();
        problem.AddParameterBlock(c_rotation[i], 4, local_parameterization);
        problem.AddParameterBlock(c_translation[i], 3);
        if (i == l) {
            problem.SetParameterBlockConstant(c_rotation[i]);
        }
        if (i == l || i == frame_num - 1) {
            problem.SetParameterBlockConstant(c_translation[i]);
        }
    }

    for (SFMFeature & sfm : sfm_features) {
        if (!sfm.state)
            continue;
        for (int j = 0; j < int(sfm.observation.size()); j++) {
            ceres::CostFunction *cost_function = ReProjectionError3D::Create(
                    sfm.observation[j].second.x,
                    sfm.observation[j].second.y);

            problem.AddResidualBlock(cost_function, nullptr, c_rotation[l], c_translation[l],
                                     sfm.position);
        }
    }
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_solver_time_in_seconds = 0.2;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if (summary.termination_type != ceres::CONVERGENCE && summary.final_cost > 5e-03) {
        return false;
    }
    for (int i = 0; i < frame_num; i++) {
        q[i].w() = c_rotation[i][0];
        q[i].x() = c_rotation[i][1];
        q[i].y() = c_rotation[i][2];
        q[i].z() = c_rotation[i][3];
        q[i] = q[i].inverse();
    }
    for (int i = 0; i < frame_num; i++) {
        T[i] = -1 * (q[i] * Vector3d(c_translation[i][0], c_translation[i][1], c_translation[i][2]));
    }
    for (SFMFeature & sfm : sfm_features) {
        if (sfm.state)
            sfm_tracked_points[sfm.id] = Vector3d(sfm.position[0], sfm.position[1],sfm.position[2]);
    }
    return true;

}

