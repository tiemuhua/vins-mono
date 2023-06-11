#include "rotation_extrinsic_estimator.h"
#include "log.h"
#include "vins_utils.h"
#include "vins_define_internal.h"

namespace vins {
    using namespace Eigen;
    using namespace std;

    bool RotationExtrinsicEstimator::calibrateRotationExtrinsic(const PointCorrespondences & correspondences,
                                                                ConstQuatRef delta_q_imu,
                                                                Matrix3d &calib_ric_result) {
        rot_visual_que_.emplace_back(solveRelativeR(correspondences));
        rot_imu_que_.emplace_back(delta_q_imu.toRotationMatrix());
        rot_imu_in_world_frame_que_.emplace_back(ric_.inverse() * delta_q_imu * ric_);

        int frame_count = (int )rot_visual_que_.size();

        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(frame_count * 4, 4);
        for (int i = 0; i < frame_count; i++) {
            Quaterniond rot_visual(rot_visual_que_[i]);
            Quaterniond rot_imu_in_world_frame(rot_imu_in_world_frame_que_[i]);
            Quaterniond rot_imu(rot_imu_que_[i]);

            Matrix4d L, R;

            double w = rot_visual.w();
            Vector3d q = rot_visual.vec();
            L.block<3, 3>(0, 0) = w * Matrix3d::Identity() + utils::skewSymmetric(q);
            L.block<3, 1>(0, 3) = q;
            L.block<1, 3>(3, 0) = -q.transpose();
            L(3, 3) = w;

            w = rot_imu.w();
            q = rot_imu.vec();
            R.block<3, 3>(0, 0) = w * Matrix3d::Identity() - utils::skewSymmetric(q);
            R.block<3, 1>(0, 3) = q;
            R.block<1, 3>(3, 0) = -q.transpose();
            R(3, 3) = w;

            double angular_distance = 180 / pi * rot_visual.angularDistance(rot_imu_in_world_frame);
            LOG_D("%d %f", i, angular_distance);
            double huber = angular_distance > 5.0 ? 5.0 / angular_distance : 1.0;
            A.block<4, 4>((i - 1) * 4, 0) = huber * (L - R);
        }

        JacobiSVD<MatrixXd> svd(A, ComputeFullU | ComputeFullV);
        Matrix<double, 4, 1> x = svd.matrixV().col(3);
        Quaterniond estimated_R(x);
        ric_ = estimated_R.toRotationMatrix().inverse();
        Vector3d ric_cov = svd.singularValues().tail<3>();
        if (frame_count >= window_size_ && ric_cov(1) > 0.25) {
            calib_ric_result = ric_;
            return true;
        } else
            return false;
    }

    Matrix3d RotationExtrinsicEstimator::solveRelativeR(const vector<pair<cv::Point2f, cv::Point2f>> &correspondences) {
        if (correspondences.size() < 9) {
            return Matrix3d::Identity();
        }
        vector<cv::Point2f> ll, rr;
        for (const PointCorrespondence & correspondence : correspondences) {
            ll.emplace_back(correspondence.first);
            rr.emplace_back(correspondence.second);
        }
        cv::Mat E = cv::findFundamentalMat(ll, rr);
        cv::Mat_<double> R1, R2, t1, t2;
        decomposeE(E, R1, R2, t1, t2);

        if (determinant(R1) + 1.0 < 1e-09) {
            E = -E;
            decomposeE(E, R1, R2, t1, t2);
        }
        double ratio1 = std::max(testTriangulation(ll, rr, R1, t1), testTriangulation(ll, rr, R1, t2));
        double ratio2 = std::max(testTriangulation(ll, rr, R2, t1), testTriangulation(ll, rr, R2, t2));
        cv::Mat_<double> ans_R_cv = ratio1 > ratio2 ? R1 : R2;

        Matrix3d ans_R_eigen;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                ans_R_eigen(j, i) = ans_R_cv(i, j);
        return ans_R_eigen;
    }

    double RotationExtrinsicEstimator::testTriangulation(const std::vector<cv::Point2f> &l, const std::vector<cv::Point2f> &r,
                                                         cv::Mat_<double> R, cv::Mat_<double> t) {
        static const cv::Matx34f P = cv::Matx34f(1, 0, 0, 0,
                                                 0, 1, 0, 0,
                                                 0, 0, 1, 0);
        const cv::Matx34f P1 = cv::Matx34f(R(0, 0), R(0, 1), R(0, 2), t(0),
                                           R(1, 0), R(1, 1), R(1, 2), t(1),
                                           R(2, 0), R(2, 1), R(2, 2), t(2));
        cv::Mat point_cloud;
        cv::triangulatePoints(P, P1, l, r, point_cloud);
        int front_count = 0;
        for (int i = 0; i < point_cloud.cols; i++) {
            double normal_factor = point_cloud.col(i).at<float>(3);

            cv::Mat_<double> p_3d_l = cv::Mat(P) * (point_cloud.col(i) / normal_factor);
            cv::Mat_<double> p_3d_r = cv::Mat(P1) * (point_cloud.col(i) / normal_factor);
            if (p_3d_l(2) > 0 && p_3d_r(2) > 0)
                front_count++;
        }
        LOG_D("testTriangulation: %f", 1.0 * front_count / point_cloud.cols);
        return 1.0 * front_count / point_cloud.cols;
    }

    void RotationExtrinsicEstimator::decomposeE(const cv::Mat& E,
                                                cv::Mat_<double> &R1, cv::Mat_<double> &R2,
                                                cv::Mat_<double> &t1, cv::Mat_<double> &t2) {
        cv::SVD svd(E, cv::SVD::MODIFY_A);
        cv::Matx33d W(0, -1, 0,
                      1, 0, 0,
                      0, 0, 1);
        cv::Matx33d Wt(0, 1, 0,
                       -1, 0, 0,
                       0, 0, 1);
        R1 = svd.u * cv::Mat(W) * svd.vt;
        R2 = svd.u * cv::Mat(Wt) * svd.vt;
        t1 = svd.u.col(2);
        t2 = -svd.u.col(2);
    }
}