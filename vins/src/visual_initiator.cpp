//
// Created by gjt on 5/14/23.
//

#include "visual_initiator.h"

namespace vins {
    bool VisualInitiator::initialStructure() {
        //check imu observability
        Vector3d sum_acc;
        // todo tiemuhuaguo 原始代码很奇怪，all_image_frame隔一个用一个，而且all_image_frame.size() - 1是什么意思？
        for (const ImageFrame &frame: all_image_frame_) {
            double dt = frame.pre_integrate_.sum_dt;
            Vector3d tmp_acc = frame.pre_integrate_.DeltaVel() / dt;
            sum_acc += tmp_acc;
        }
        Vector3d avg_acc = sum_acc / (double )all_image_frame_.size();
        double var = 0;
        for (const ImageFrame &frame:all_image_frame_) {
            double dt = frame.pre_integrate_.sum_dt;
            Vector3d tmp_acc = frame.pre_integrate_.DeltaVel() / dt;
            var += (tmp_acc - avg_acc).transpose() * (tmp_acc - avg_acc);
        }
        var = sqrt(var / (double )all_image_frame_.size());
        if (var < 0.25) {
            LOG_E("IMU excitation not enough!");
            return false;
        }

        // global sfm
        map<int, Vector3d> sfm_tracked_points;
        vector<SFMFeature> sfm_features;
        for (const FeaturesOfId &features_of_id: feature_manager_.features_) {
            SFMFeature sfm_feature;
            sfm_feature.state = false;
            sfm_feature.id = features_of_id.feature_id_;
            for (int i = 0; i < features_of_id.feature_points_.size(); ++i) {
                sfm_feature.observation.emplace_back(
                        make_pair(features_of_id.start_frame_ + i, features_of_id.feature_points_[i].unified_point));
            }
            sfm_features.push_back(sfm_feature);
        }
        Matrix3d relative_R;
        Vector3d relative_T;
        int l;
        if (!relativePose(relative_R, relative_T, l)) {
            LOG_I("Not enough features or parallax; Move device around");
            return false;
        }
        Quaterniond Q[frame_count_ + 1];
        Vector3d T[frame_count_ + 1];
        if (!GlobalSFM::construct(frame_count_ + 1, Q, T, l, relative_R, relative_T, sfm_features, sfm_tracked_points)) {
            LOG_D("global SFM failed!");
            return false;
        }

        //solve pnp for all frame
        int i = 0;
        for (ImageFrame &frame:all_image_frame_) {
            // provide initial guess
            cv::Mat r, rvec, t, D, tmp_r;
            if (frame.t == time_stamp_window[i]) {
                frame.is_key_frame_ = true;
                frame.R = Q[i].toRotationMatrix() * RIC.transpose();
                frame.T = T[i];
                i++;
                continue;
            }
            if ((frame.t) > time_stamp_window[i]) {
                i++;
            }
            Matrix3d R_initial = (Q[i].inverse()).toRotationMatrix();
            Vector3d P_initial = -R_initial * T[i];
            cv::eigen2cv(R_initial, tmp_r);
            cv::Rodrigues(tmp_r, rvec);
            cv::eigen2cv(P_initial, t);

            frame.is_key_frame_ = false;
            vector<cv::Point3f> pts_3_vector;
            vector<cv::Point2f> pts_2_vector;
            for (const FeaturePoint &point:frame.points) {
                int feature_id = point.feature_id;
                const auto it = sfm_tracked_points.find(feature_id);
                if (it != sfm_tracked_points.end()) {
                    Vector3d world_pts = it->second;
                    pts_3_vector.emplace_back(world_pts(0), world_pts(1), world_pts(2));
                    pts_2_vector.emplace_back(point.unified_point);
                }
            }
            cv::Mat K = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
            if (pts_3_vector.size() < 6) {
                LOG_D("pts_3_vector size:%lu, Not enough points for solve pnp !", pts_3_vector.size());
                return false;
            }
            if (!cv::solvePnP(pts_3_vector, pts_2_vector, K, D, rvec, t, false)) {
                LOG_D("solve pnp fail!");
                return false;
            }

            cv::Rodrigues(rvec, r);
            MatrixXd tmp_R_pnp;
            cv::cv2eigen(r, tmp_R_pnp);
            MatrixXd R_pnp = tmp_R_pnp.transpose();
            MatrixXd T_pnp;
            cv::cv2eigen(t, T_pnp);
            frame.T = R_pnp * (-T_pnp);
            frame.R = R_pnp * RIC.transpose();
        }
        return true;
    }
}
