//
// Created by gjt on 5/14/23.
//

#include "visual_initiator.h"

#include <vector>

#include <opencv2/core/eigen.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "log.h"
#include "vins/impl/vins_utils.h"
#include "vins/impl/feature_helper.h"
#include "motion_estimator.h"

namespace vins {
    using namespace std;

    struct SFMFeature {
        bool has_been_triangulated = false;
        Eigen::Vector3d position = Eigen::Vector3d::Zero();
        Feature feature;
    };

    struct ReProjectionError3D {
        explicit ReProjectionError3D(const cv::Point2f &observed_point)
                : observed_point_(observed_point) {}

        template<typename T>
        bool operator()(const T *const camera_R, const T *const camera_T, const T *point, T *residuals) const {
            T p[3];
            ceres::QuaternionRotatePoint(camera_R, point, p);
            utils::arrayPlus(p, camera_T, p, 3);
            T xp = p[0] / p[2];
            T yp = p[1] / p[2];
            residuals[0] = xp - T(observed_point_.x);
            residuals[1] = yp - T(observed_point_.y);
            return true;
        }

        static ceres::CostFunction *Create(const cv::Point2f &observed_point) {
            return (new ceres::AutoDiffCostFunction<ReProjectionError3D, 2, 4, 3, 3>(
                    new ReProjectionError3D(observed_point)));
        }

        cv::Point2f observed_point_;
    };

    static double getAverageParallax(const vector<pair<cv::Point2f , cv::Point2f>>& correspondences) {
        double sum_parallax = 0;
        for (auto &correspond: correspondences) {
            double parallax = norm(correspond.first - correspond.second);
            sum_parallax += parallax;
        }
        return 1.0 * sum_parallax / int(correspondences.size());
    }

    inline bool isFrameHasFeature(int frame_idx, const Feature& feature) {
        return feature.start_kf_window_idx <= frame_idx && frame_idx < feature.start_kf_window_idx + feature.points.size();
    }

    using Mat34 = Eigen::Matrix<double, 3, 4>;
    static Eigen::Vector3d triangulatePoint(const Mat34 &Pose0, const Mat34 &Pose1,
                                            const cv::Point2f &point0, const cv::Point2f &point1) {
        Eigen::Matrix4d design_matrix = Eigen::Matrix4d::Zero();
        design_matrix.row(0) = point0.x * Pose0.row(2) - Pose0.row(0);
        design_matrix.row(1) = point0.y * Pose0.row(2) - Pose0.row(1);
        design_matrix.row(2) = point1.x * Pose1.row(2) - Pose1.row(0);
        design_matrix.row(3) = point1.y * Pose1.row(2) - Pose1.row(1);
        Eigen::Vector4d triangulated_point =
                design_matrix.jacobiSvd(Eigen::ComputeFullV).matrixV().rightCols<1>();
        return triangulated_point.block<3,1>(0,0) / triangulated_point(3);
    }

    static void triangulatePtsByFramePos(int frame0, const Mat34 &Pose0,
                                         int frame1, const Mat34 &Pose1,
                                         vector<SFMFeature> &sfm_features) {
        assert(frame0 != frame1);
        for (SFMFeature & sfm : sfm_features) {
            if (sfm.has_been_triangulated)
                continue;
            if (!isFrameHasFeature(frame0, sfm.feature) || !isFrameHasFeature(frame1, sfm.feature)) {
                continue;
            }
            cv::Point2f point0 = sfm.feature.points.at(frame0 - sfm.feature.start_kf_window_idx);
            cv::Point2f point1 = sfm.feature.points.at(frame1 - sfm.feature.start_kf_window_idx);
            sfm.position = triangulatePoint(Pose0, Pose1, point0, point1);
            sfm.has_been_triangulated = true;
        }
    }

    static void collectFeaturesInFrame(const vector<SFMFeature> &sfm_features, const int frame_idx,
                                       vector<cv::Point2f> &pts_2d, vector<cv::Point3f> &pts_3d) {
        for (const SFMFeature & sfm : sfm_features) {
            if (sfm.has_been_triangulated && isFrameHasFeature(frame_idx, sfm.feature)) {
                cv::Point2f img_pts = sfm.feature.points[frame_idx - sfm.feature.start_kf_window_idx];
                pts_2d.emplace_back(img_pts);
                pts_3d.emplace_back(sfm.position[0], sfm.position[1], sfm.position[2]);
            }
        }
    }

    static bool solveFrameByPnP(const vector<cv::Point2f> &pts_2d, const vector<cv::Point3f> &pts_3d,
                                const bool use_extrinsic_guess, Eigen::Matrix3d &R, Eigen::Vector3d &T) {
        cv::Mat r, rvec, t, D, tmp_r;
        cv::eigen2cv(R, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(T, t);
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        if (!cv::solvePnP(pts_3d, pts_2d, K, D, rvec, t, use_extrinsic_guess)) {
            LOG_E("solve pnp fail");
            return false;
        }
        cv::Rodrigues(rvec, r);
        cv::cv2eigen(r, R);
        cv::cv2eigen(t, T);
        return true;
    }

    bool initiateByVisual(const int cur_window_size,
                          const std::vector<Feature>& feature_window,
                          vector<Frame> &all_frames) {
        // 计算sfm_features
        vector<SFMFeature> sfm_features;
        for (const Feature &feature: feature_window) {
            SFMFeature sfm_feature;
            sfm_feature.feature = feature;
            sfm_features.push_back(sfm_feature);
        }

        // 找到和末关键帧视差足够大的关键帧，并计算末关键帧相对该帧的位姿
        Eigen::Matrix3d relative_R;
        Eigen::Vector3d relative_unit_T;
        int big_parallax_frame_id = -1;
        for (int i = 0; i < cur_window_size; ++i) {
            vector<pair<cv::Point2f , cv::Point2f>> correspondences =
                    FeatureHelper::getCorrespondences(i, cur_window_size, feature_window);
            constexpr double avg_parallax_threshold = 30.0/460;
            if (correspondences.size() < 20 || getAverageParallax(correspondences) < avg_parallax_threshold) {
                continue;
            }
            MotionEstimator::solveRelativeRT(correspondences, relative_R, relative_unit_T);
            big_parallax_frame_id = i;
            break;
        }
        if (big_parallax_frame_id == -1) {
            LOG_E("Not enough feature_window or parallax; Move device around");
            return false;
        }

        // .记big_parallax_frame_id为l，秦通的代码里用的l这个符号.
        // 1: triangulate between l <-> frame_num - 1
        std::vector<Mat34> kf_poses(cur_window_size, Mat34::Zero());
        kf_poses[big_parallax_frame_id].block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
        kf_poses[cur_window_size - 1].block<3, 3>(0, 0) = relative_R;
        kf_poses[cur_window_size - 1].block<3, 1>(0, 3) = relative_unit_T;
        triangulatePtsByFramePos(big_parallax_frame_id, kf_poses[big_parallax_frame_id],
                                 cur_window_size - 1, kf_poses[cur_window_size - 1], sfm_features);

        // 2: solve pnp [l+1, frame_num-2]
        // triangulate [l+1, frame_num-2] <-> frame_num-1;
        for (int frame_idx = big_parallax_frame_id + 1; frame_idx < cur_window_size - 1; frame_idx++) {
            // solve pnp
            Eigen::Matrix3d R_initial = kf_poses[frame_idx - 1].block<3, 3>(0, 0);
            Eigen::Vector3d P_initial = kf_poses[frame_idx - 1].block<3, 1>(0, 3);
            vector<cv::Point2f> pts_2d;
            vector<cv::Point3f> pts_3d;
            collectFeaturesInFrame(sfm_features, frame_idx, pts_2d, pts_3d);
            if (!solveFrameByPnP(pts_2d, pts_3d, true, R_initial, P_initial))
                return false;
            kf_poses[frame_idx].block<3, 3>(0, 0) = R_initial;
            kf_poses[frame_idx].block<3, 1>(0, 3) = P_initial;

            // triangulate point based on to solve pnp result
            triangulatePtsByFramePos(frame_idx, kf_poses[frame_idx],
                                     cur_window_size - 1, kf_poses[cur_window_size - 1], sfm_features);
        }

        //3: triangulate l <-> [l+1, frame_num -2]
        for (int frame_idx = big_parallax_frame_id + 1; frame_idx < cur_window_size - 1; frame_idx++) {
            triangulatePtsByFramePos(big_parallax_frame_id, kf_poses[big_parallax_frame_id],
                                     frame_idx, kf_poses[frame_idx], sfm_features);
        }

        // 4: solve pnp for frame [0, l-1]
        //    triangulate [0, l-1] <-> l
        for (int frame_idx = big_parallax_frame_id - 1; frame_idx >= 0; frame_idx--) {
            //solve pnp
            Eigen::Matrix3d R_init = kf_poses[frame_idx + 1].block<3, 3>(0, 0);
            Eigen::Vector3d T_init = kf_poses[frame_idx + 1].block<3, 1>(0, 3);
            vector<cv::Point2f> pts_2d;
            vector<cv::Point3f> pts_3d;
            collectFeaturesInFrame(sfm_features, frame_idx, pts_2d, pts_3d);
            if (!solveFrameByPnP(pts_2d, pts_3d, true, R_init, T_init)) {
                return false;
            }
            kf_poses[frame_idx].block<3, 3>(0, 0) = R_init;
            kf_poses[frame_idx].block<3, 1>(0, 3) = T_init;
            //triangulate
            triangulatePtsByFramePos(frame_idx, kf_poses[frame_idx],
                                     big_parallax_frame_id, kf_poses[big_parallax_frame_id], sfm_features);
        }

        //5: triangulate all others points
        for (SFMFeature& sfm: sfm_features) {
            if (sfm.has_been_triangulated || sfm.feature.points.size() < 2) {
                continue;
            }
            int frame_0 = sfm.feature.start_kf_window_idx;
            cv::Point2f point0 = sfm.feature.points.front();
            int frame_1 = sfm.feature.start_kf_window_idx + (int )sfm.feature.points.size();
            cv::Point2f point1 = sfm.feature.points.back();
            sfm.position = triangulatePoint(kf_poses[frame_0], kf_poses[frame_1], point0, point1);
            sfm.has_been_triangulated = true;
        }

        /**************************************************************
         * 前面通过"帧位姿->点深度->下个帧的位姿"递推地求出了初始值，下面进行BA *
         * ************************************************************/
        ceres::Problem problem;
        ceres::Manifold *quat_manifold = new ceres::QuaternionManifold();
        std::vector<std::array<double, 4>> c_key_frames_rot(cur_window_size);
        std::vector<std::array<double, 3>> c_key_frames_pos(cur_window_size);
        for (int i = 0; i < cur_window_size; i++) {
            utils::quat2array(Eigen::Quaterniond(kf_poses[i].block<3, 3>(0, 0)), c_key_frames_rot[i].data());
            utils::vec3d2array(kf_poses[i].block<3, 1>(0, 3), c_key_frames_pos[i].data());
            problem.AddParameterBlock(c_key_frames_rot[i].data(), 4, quat_manifold);
            problem.AddParameterBlock(c_key_frames_pos[i].data(), 3);
            if (i == big_parallax_frame_id) {
                problem.SetParameterBlockConstant(c_key_frames_rot[i].data());
            }
            if (i == big_parallax_frame_id || i == cur_window_size - 1) {
                problem.SetParameterBlockConstant(c_key_frames_pos[i].data());
            }
        }

        for (SFMFeature & sfm : sfm_features) {
            if (!sfm.has_been_triangulated)
                continue;
            for (int frame_bias = 0; frame_bias < sfm.feature.points.size(); ++frame_bias) {
                ceres::CostFunction *cost_function =
                        ReProjectionError3D::Create(sfm.feature.points[frame_bias]);
                problem.AddResidualBlock(cost_function, nullptr,
                                         c_key_frames_rot[big_parallax_frame_id].data(),
                                         c_key_frames_pos[big_parallax_frame_id].data(),
                                         sfm.position.data());
            }
        }
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.max_solver_time_in_seconds = 0.2;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        if (summary.termination_type != ceres::CONVERGENCE && summary.final_cost > 5e-03) {
            LOG_E("full ba not convergence!, termination_type:%d, final_cost:%f",
                  summary.termination_type, summary.final_cost);
            return false;
        }

        /******************************************************************
         * BA结果储存于key_frames_rot、key_frames_pos、feature_id_2_position *
         * ****************************************************************/
        vector<Eigen::Matrix3d> key_frames_rot(cur_window_size + 1); // todo tiemuhuaguo Q和T是在哪个坐标系里的位姿？？？
        vector<Eigen::Vector3d> key_frames_pos(cur_window_size + 1);
        map<int, Eigen::Vector3d> feature_id_2_position;
        for (int i = 0; i < cur_window_size; i++) {
            key_frames_rot[i] = utils::array2quat(c_key_frames_rot[i].data()).toRotationMatrix();
        }
        for (int i = 0; i < cur_window_size; i++) {
            key_frames_pos[i] = utils::array2vec3d(c_key_frames_pos[i].data());
        }
        for (SFMFeature & sfm : sfm_features) {
            if (sfm.has_been_triangulated)
                feature_id_2_position[sfm.feature.feature_id] = sfm.position;
        }

        /******************************************************************
         *             利用关键帧位姿和特征点深度PNP求解非关键帧位姿              *
         * ****************************************************************/
        int key_frame_idx = 0;
        for (Frame &frame:all_frames) {
            // provide initial guess
            if (frame.is_key_frame_) {
                frame.R = key_frames_rot[key_frame_idx];
                frame.T = key_frames_pos[key_frame_idx];
                key_frame_idx++;
                continue;
            }

            vector<cv::Point3f> pts_3d;
            vector<cv::Point2f> pts_2d;
            for (int i = 0; i < frame.points.size(); ++i) {
                if (feature_id_2_position.count(frame.feature_ids[i])) {
                    Eigen::Vector3d world_pts = feature_id_2_position[frame.feature_ids[i]];
                    pts_3d.emplace_back(world_pts(0), world_pts(1), world_pts(2));
                    pts_2d.emplace_back(frame.points[i]);
                }
            }
            if (pts_3d.size() < 6) {
                LOG_E("pts_3_vector size:%lu, Not enough points for solve pnp !", pts_3d.size());
                return false;
            }

            Eigen::Matrix3d R_initial = key_frames_rot[key_frame_idx];
            Eigen::Vector3d P_initial = key_frames_pos[key_frame_idx];
            if (!solveFrameByPnP(pts_2d, pts_3d, false, R_initial, P_initial)) {
                LOG_E("solve pnp fail!");
                return false;
            }
            frame.R = R_initial;
            frame.T = P_initial;
        }
        return true;
    }
}
