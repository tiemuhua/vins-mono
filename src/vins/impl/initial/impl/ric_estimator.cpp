#include "ric_estimator.h"
#include "log.h"
#include "vins/impl/vins_utils.h"

namespace vins {
    using namespace Eigen;
    using namespace std;

    bool estimateRIC(const std::vector<Eigen::Matrix3d> &img_rots,
                     const std::vector<Eigen::Matrix3d> &imu_rots,
                     Matrix3d &calib_ric_result) {
        int frame_count = (int )img_rots.size();

        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(frame_count * 4, 4);
        for (int i = 0; i < frame_count; i++) {
            Quaterniond rot_img(img_rots[i]);
            Quaterniond rot_imu(imu_rots[i]);

            Matrix4d L;
            double img_w = rot_img.w();
            Vector3d img_q = rot_img.vec();
            L.block<3, 3>(0, 0) = img_w * Matrix3d::Identity() + utils::skewSymmetric(img_q);
            L.block<3, 1>(0, 3) = img_q;
            L.block<1, 3>(3, 0) = -img_q.transpose();
            L(3, 3) = img_w;

            Matrix4d R;
            double imu_w = rot_imu.w();
            Vector3d imu_q = rot_imu.vec();
            R.block<3, 3>(0, 0) = imu_w * Matrix3d::Identity() - utils::skewSymmetric(imu_q);
            R.block<3, 1>(0, 3) = imu_q;
            R.block<1, 3>(3, 0) = -imu_q.transpose();
            R(3, 3) = imu_w;

            A.block<4, 4>((i - 1) * 4, 0) = L - R;
        }

        JacobiSVD<MatrixXd> svd(A, ComputeFullU | ComputeFullV);
        Matrix<double, 4, 1> x = svd.matrixV().col(3);
        Quaterniond estimated_R(x);
        if (svd.singularValues()(2) < 0.25) {
            return false;
        }
        calib_ric_result = estimated_R.toRotationMatrix().inverse();
        return true;
    }
}