//
// Created by gjt on 5/14/23.
//

#include "visual_initiator.h"

#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "log.h"
#include "vins_utils.h"
#include "parameters.h"

#include "feature_helper.h"
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

    inline bool isFrameHasFeature(int frame_id, const Feature& feature) {
        return feature.start_frame <= frame_id && frame_id < feature.start_frame + feature.points.size();
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

    static void triangulateTwoFrames(int frame0, const Mat34 &Pose0,
                                     int frame1, const Mat34 &Pose1,
                                     vector<SFMFeature> &sfm_features) {
        assert(frame0 != frame1);
        for (SFMFeature & sfm : sfm_features) {
            if (sfm.has_been_triangulated)
                continue;
            if (!isFrameHasFeature(frame0, sfm.feature) || !isFrameHasFeature(frame1, sfm.feature)) {
                continue;
            }
            cv::Point2f point0 = sfm.feature.points.at(frame0 - sfm.feature.start_frame);
            cv::Point2f point1 = sfm.feature.points.at(frame1 - sfm.feature.start_frame);
            sfm.position = triangulatePoint(Pose0, Pose1, point0, point1);
            sfm.has_been_triangulated = true;
        }
    }

    static void calcFeaturePtsInFrame(const vector<SFMFeature> &sfm_features, const int frame_id,
                                      vector<cv::Point2f> &pts_2_vector, vector<cv::Point3f> &pts_3_vector) {
        for (const SFMFeature & sfm : sfm_features) {
            if (sfm.has_been_triangulated && isFrameHasFeature(frame_id, sfm.feature)) {
                cv::Point2f img_pts = sfm.feature.points[frame_id - sfm.feature.start_frame];
                pts_2_vector.emplace_back(img_pts);
                pts_3_vector.emplace_back(sfm.position[0], sfm.position[1], sfm.position[2]);
            }
        }
    }

    static bool solveFrameByPnP(const vector<cv::Point2f> &pts_2_vector, const vector<cv::Point3f> &pts_3_vector,
                                const bool use_extrinsic_guess, Eigen::Matrix3d &R, Eigen::Vector3d &T) {
        cv::Mat r, rvec, t, D, tmp_r;
        cv::eigen2cv(R, tmp_r);
        cv::Rodrigues(tmp_r, rvec);
        cv::eigen2cv(T, t);
        cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        if (!cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, use_extrinsic_guess)) {
            LOG_E("solve pnp fail");
            return false;
        }
        cv::Rodrigues(rvec, r);
        cv::cv2eigen(r, R);
        cv::cv2eigen(t, T);
        return true;
    }

    bool initiateByVisual(int key_frame_num,
                          const std::vector<Feature>& features,
                          vector<Frame> &all_frames) {
        // 计算sfm_features
        vector<SFMFeature> sfm_features;
        for (const Feature &feature: features) {
            SFMFeature sfm_feature;
            sfm_feature.feature = feature;
            sfm_features.push_back(sfm_feature);
        }

        // 找到和末关键帧视差足够大的关键帧，并计算末关键帧相对该帧的位姿
        Eigen::Matrix3d relative_R;
        Eigen::Vector3d relative_T;
        int big_parallax_frame_id = -1;
        for (int i = 0; i < Param::Instance().window_size; ++i) {
            vector<pair<cv::Point2f , cv::Point2f>> correspondences =
                    FeatureHelper::getCorrespondences(i, Param::Instance().window_size, features);
            constexpr double avg_parallax_threshold = 30.0/460;
            if (correspondences.size() < 20 || getAverageParallax(correspondences) < avg_parallax_threshold) {
                continue;
            }
            MotionEstimator::solveRelativeRT(correspondences, relative_R, relative_T);
            big_parallax_frame_id = i;
            break;
        }
        if (big_parallax_frame_id == -1) {
            LOG_E("Not enough features or parallax; Move device around");
            return false;
        }

        // .记big_parallax_frame_id为l，秦通的代码里用的l这个符号.
        // 1: triangulate between l <-> frame_num - 1
        std::vector<Mat34> Pose(key_frame_num, Mat34::Zero());
        Pose[big_parallax_frame_id].block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
        Pose[key_frame_num - 1].block<3, 3>(0, 0) = relative_R;
        Pose[key_frame_num - 1].block<3, 1>(0, 3) = relative_T;
        triangulateTwoFrames(big_parallax_frame_id, Pose[big_parallax_frame_id],
                             key_frame_num - 1, Pose[key_frame_num - 1], sfm_features);

        // 2: solve pnp [l+1, frame_num-2]
        // triangulate [l+1, frame_num-2] <-> frame_num-1;
        for (int frame_id = big_parallax_frame_id + 1; frame_id < key_frame_num - 1; frame_id++) {
            // solve pnp
            Eigen::Matrix3d R_initial = Pose[frame_id - 1].block<3, 3>(0, 0);
            Eigen::Vector3d P_initial = Pose[frame_id - 1].block<3, 1>(0, 3);
            vector<cv::Point2f> pts_2_vector;
            vector<cv::Point3f> pts_3_vector;
            calcFeaturePtsInFrame(sfm_features, frame_id, pts_2_vector, pts_3_vector);
            if (!solveFrameByPnP(pts_2_vector, pts_3_vector, true, R_initial, P_initial))
                return false;
            Pose[frame_id].block<3, 3>(0, 0) = R_initial;
            Pose[frame_id].block<3, 1>(0, 3) = P_initial;

            // triangulate point based on to solve pnp result
            triangulateTwoFrames(frame_id, Pose[frame_id], key_frame_num - 1, Pose[key_frame_num - 1], sfm_features);
        }

        //3: triangulate l <-> [l+1, frame_num -2]
        for (int frame_id = big_parallax_frame_id + 1; frame_id < key_frame_num - 1; frame_id++) {
            triangulateTwoFrames(big_parallax_frame_id, Pose[big_parallax_frame_id], frame_id, Pose[frame_id], sfm_features);
        }

        // 4: solve pnp for frame [0, l-1]
        //    triangulate [0, l-1] <-> l
        for (int frame_id = big_parallax_frame_id - 1; frame_id >= 0; frame_id--) {
            //solve pnp
            Eigen::Matrix3d R_initial = Pose[frame_id + 1].block<3, 3>(0, 0);
            Eigen::Vector3d P_initial = Pose[frame_id + 1].block<3, 1>(0, 3);
            vector<cv::Point2f> pts_2_vector;
            vector<cv::Point3f> pts_3_vector;
            calcFeaturePtsInFrame(sfm_features, frame_id, pts_2_vector, pts_3_vector);
            if (!solveFrameByPnP(pts_2_vector, pts_3_vector, true, R_initial, P_initial)) {
                return false;
            }
            Pose[frame_id].block<3, 3>(0, 0) = R_initial;
            Pose[frame_id].block<3, 1>(0, 3) = P_initial;
            //triangulate
            triangulateTwoFrames(frame_id, Pose[frame_id], big_parallax_frame_id, Pose[big_parallax_frame_id], sfm_features);
        }

        //5: triangulate all others points
        for (SFMFeature& sfm: sfm_features) {
            if (sfm.has_been_triangulated || sfm.feature.points.size() < 2) {
                continue;
            }
            int frame_0 = sfm.feature.start_frame;
            cv::Point2f point0 = sfm.feature.points.front();
            int frame_1 = sfm.feature.start_frame + sfm.feature.points.size();
            cv::Point2f point1 = sfm.feature.points.back();
            sfm.position = triangulatePoint(Pose[frame_0], Pose[frame_1], point0, point1);
            sfm.has_been_triangulated = true;
        }

        //6: full BA
        ceres::Problem problem;
        ceres::Manifold *local_parameterization = new ceres::QuaternionManifold();
        double c_rotation[key_frame_num][4];
        double c_translation[key_frame_num][3];
        for (int i = 0; i < key_frame_num; i++) {
            utils::quat2array(Eigen::Quaterniond(Pose[i].block<3, 3>(0, 0)), c_rotation[i]);
            utils::vec3d2array(Pose[i].block<3, 1>(0, 3), c_translation[i]);
            problem.AddParameterBlock(c_rotation[i], 4, local_parameterization);
            problem.AddParameterBlock(c_translation[i], 3);
            if (i == big_parallax_frame_id) {
                problem.SetParameterBlockConstant(c_rotation[i]);
            }
            if (i == big_parallax_frame_id || i == key_frame_num - 1) {
                problem.SetParameterBlockConstant(c_translation[i]);
            }
        }

        for (SFMFeature & sfm : sfm_features) {
            if (!sfm.has_been_triangulated)
                continue;
            for (int frame_bias = 0; frame_bias < sfm.feature.points.size(); ++frame_bias) {
                ceres::CostFunction *cost_function =
                        ReProjectionError3D::Create(sfm.feature.points[frame_bias]);
                problem.AddResidualBlock(cost_function, nullptr,
                                         c_rotation[big_parallax_frame_id], c_translation[big_parallax_frame_id], sfm.position.data());
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

        //后续只需要Q、key_frames_pos、feature_id_2_position
        vector<Eigen::Matrix3d> key_frames_rot(Param::Instance().window_size + 1); // todo tiemuhuaguo Q和T是在哪个坐标系里的位姿？？？
        vector<Eigen::Vector3d> key_frames_pos(Param::Instance().window_size + 1);
        map<int, Eigen::Vector3d> feature_id_2_position;
        for (int i = 0; i < key_frame_num; i++) {
            key_frames_rot[i] = utils::array2quat(c_rotation[i]).toRotationMatrix();
        }
        for (int i = 0; i < key_frame_num; i++) {
            key_frames_pos[i] = utils::array2vec3d(c_translation[i]);
        }
        for (SFMFeature & sfm : sfm_features) {
            if (sfm.has_been_triangulated)
                feature_id_2_position[sfm.feature.feature_id] = sfm.position;
        }

        //利用关键帧位姿和特征点深度PNP求解非关键帧位姿
        int key_frame_idx = 0;
        for (Frame &frame:all_frames) {
            // provide initial guess
            if (frame.is_key_frame_) {
                frame.R = key_frames_rot[key_frame_idx];
                frame.T = key_frames_pos[key_frame_idx];
                key_frame_idx++;
                continue;
            }

            vector<cv::Point3f> pts_3_vector;
            vector<cv::Point2f> pts_2_vector;
            for (int i = 0; i < frame.points.size(); ++i) {
                if (feature_id_2_position.count(frame.feature_ids[i])) {
                    Eigen::Vector3d world_pts = feature_id_2_position[frame.feature_ids[i]];
                    pts_3_vector.emplace_back(world_pts(0), world_pts(1), world_pts(2));
                    pts_2_vector.emplace_back(frame.points[i]);
                }
            }
            if (pts_3_vector.size() < 6) {
                LOG_E("pts_3_vector size:%lu, Not enough points for solve pnp !", pts_3_vector.size());
                return false;
            }

            Eigen::Matrix3d R_initial = key_frames_rot[key_frame_idx];
            Eigen::Vector3d P_initial = key_frames_pos[key_frame_idx];
            if (!solveFrameByPnP(pts_2_vector, pts_3_vector, false, R_initial, P_initial)) {
                LOG_E("solve pnp fail!");
                return false;
            }
            frame.R = R_initial;
            frame.T = P_initial;
        }
        return true;
    }
}
