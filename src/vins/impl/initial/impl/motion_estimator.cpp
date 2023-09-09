#include "motion_estimator.h"
#include <opencv2/calib3d.hpp>
#include <Eigen/Eigen>
#include <opencv2/core/eigen.hpp>

// todo tiemuhuaguo 坐标系搞混了，应该是在l坐标系中r相对于l的旋转和位移
namespace vins{
    bool MotionEstimator::solveRelativeRT(const Correspondences &correspondences,
                                          Eigen::Matrix3d &rotation,
                                          Eigen::Vector3d &unit_translation) {
        std::vector<cv::Point2f> ll, rr;
        for (const auto & correspondence : correspondences) {
            ll.emplace_back(correspondence.first);
            rr.emplace_back(correspondence.second);
        }
        cv::Mat mask;
        cv::Mat E = cv::findFundamentalMat(ll, rr, cv::FM_RANSAC, 0.3 / 460, 0.99, mask);
        cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        cv::Mat cv_rot, cv_trans;
        int inliner_cnt = cv::recoverPose(E, ll, rr, cameraMatrix, cv_rot, cv_trans, mask);
        if (inliner_cnt < 13) {
            return false;
        }
        cv::cv2eigen(cv_rot, rotation);
        cv::cv2eigen(cv_trans, unit_translation);
        assert(abs(unit_translation.norm() - 1) < 1e-5);
        return true;
    }
}
