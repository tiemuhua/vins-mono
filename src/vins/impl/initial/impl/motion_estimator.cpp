#include "motion_estimator.h"
#include <opencv2/calib3d.hpp>

// todo tiemuhuaguo 坐标系搞混了，应该是在l坐标系中r相对于l的旋转和位移
namespace vins{
    bool MotionEstimator::solveRelativeRT(const Correspondences &correspondences,
                                          Eigen::Matrix3d &Rotation, Eigen::Vector3d &Translation) {
        if (correspondences.size() < 15) {
            return false;
        }
        std::vector<cv::Point2f> ll, rr;
        for (const auto & correspondence : correspondences) {
            ll.emplace_back(correspondence.first);
            rr.emplace_back(correspondence.second);
        }
        cv::Mat mask;
        cv::Mat E = cv::findFundamentalMat(ll, rr, cv::FM_RANSAC, 0.3 / 460, 0.99, mask);
        cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << 1, 0, 0, 0, 1, 0, 0, 0, 1);
        cv::Mat rot, trans;
        int inliner_cnt = cv::recoverPose(E, ll, rr, cameraMatrix, rot, trans, mask);

        for (int i = 0; i < 3; i++) {
            Translation(i) = trans.at<double>(i, 0);
            for (int j = 0; j < 3; j++)
                Rotation(i, j) = rot.at<double>(i, j);
        }
        if (inliner_cnt > 12)
            return true;
        else
            return false;

    }
}



