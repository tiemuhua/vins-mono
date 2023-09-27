#include "project_cost.h"
#include <glog/logging.h>
#include "vins/impl/vins_utils.h"
#include "vins/param.h"
#include "vins/vins_logic.h"

using namespace vins;

ProjectCost::ProjectCost(const cv::Point2f &_pts_i, const cv::Point2f &_pts_j) {
    pts_i = Eigen::Vector3d(_pts_i.x, _pts_i.y, 1.0);
    pts_j = Eigen::Vector3d(_pts_j.x, _pts_j.y, 1.0);
    Eigen::Vector3d a = pts_j.normalized();
    Eigen::Vector3d tmp(0, 0, 1);
    if ((a - tmp).norm() < 0.001) {
        tmp << 1, 0, 0;
    }
    Eigen::Vector3d b1 = (tmp - a * (a.transpose() * tmp)).normalized();
    Eigen::Vector3d b2 = a.cross(b1);
    tangent_base.block<1, 3>(0, 0) = b1.transpose();
    tangent_base.block<1, 3>(1, 0) = b2.transpose();
};

bool ProjectCost::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
    Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Eigen::Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
    Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    Eigen::Vector3d tic(parameters[2][0], parameters[2][1], parameters[2][2]);
    Eigen::Quaterniond qic(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);

    double inv_dep_i = parameters[3][0];

    Eigen::Vector3d pts_camera_i = pts_i / inv_dep_i;
    Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic;
    Eigen::Vector3d pts_w = Qi * pts_imu_i + Pi;
    Eigen::Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj);
    Eigen::Vector3d pts_camera_j = qic.inverse() * (pts_imu_j - tic);
    Eigen::Map<Eigen::Vector2d> residual(residuals);

    residual = tangent_base * (pts_camera_j.normalized() - pts_j.normalized());
    Eigen::Matrix2d sqrt_info = vins::getParam()->camera.focal / 1.5 * Eigen::Matrix2d::Identity();
    residual = sqrt_info * residual;

    assert(jacobians && jacobians[0] && jacobians[1] && jacobians[2] && jacobians[3]);
    if (!(jacobians && jacobians[0] && jacobians[1] && jacobians[2] && jacobians[3])) {
        LOG(ERROR) << "!jacobian";
        return false;
    }

    Eigen::Matrix3d Ri = Qi.toRotationMatrix();
    Eigen::Matrix3d Rj = Qj.toRotationMatrix();
    Eigen::Matrix3d ric = qic.toRotationMatrix();
    Eigen::Matrix<double, 2, 3> reduce(2, 3);
    double norm = pts_camera_j.norm();
    double norm3 = pow(norm, 3);
    Eigen::Matrix3d norm_jacobian =
            Eigen::Matrix3d::Identity() / norm - pts_camera_j * pts_camera_j.transpose() / norm3;
    reduce = tangent_base * norm_jacobian;
    reduce = sqrt_info * reduce;

    Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
    Eigen::Matrix<double, 3, 6> jaco_i;
    jaco_i.leftCols<3>() = ric.transpose() * Rj.transpose();
    jaco_i.rightCols<3>() = ric.transpose() * Rj.transpose() * Ri * utils::skewSymmetric(pts_imu_i);
    jacobian_pose_i.leftCols<6>() = reduce * jaco_i;
    jacobian_pose_i.rightCols<1>().setZero();

    Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[1]);
    Eigen::Matrix<double, 3, 6> jaco_j;
    jaco_j.leftCols<3>() = ric.transpose() * -Rj.transpose();
    jaco_j.rightCols<3>() = ric.transpose() * utils::skewSymmetric(pts_imu_j);
    jacobian_pose_j.leftCols<6>() = reduce * jaco_j;
    jacobian_pose_j.rightCols<1>().setZero();

    Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_ex_pose(jacobians[2]);
    Eigen::Matrix<double, 3, 6> jaco_ex;
    jaco_ex.leftCols<3>() = ric.transpose() * (Rj.transpose() * Ri - Eigen::Matrix3d::Identity());
    Eigen::Matrix3d tmp_r = ric.transpose() * Rj.transpose() * Ri * ric;
    jaco_ex.rightCols<3>() =
            -tmp_r * utils::skewSymmetric(pts_camera_i) +
            utils::skewSymmetric(tmp_r * pts_camera_i) +
            utils::skewSymmetric(ric.transpose() * (Rj.transpose() * (Ri * tic + Pi - Pj) - tic));
    jacobian_ex_pose.leftCols<6>() = reduce * jaco_ex;
    jacobian_ex_pose.rightCols<1>().setZero();

    Eigen::Map<Eigen::Vector2d> jacobian_feature(jacobians[3]);
    jacobian_feature = reduce * ric.transpose() * Rj.transpose() * Ri * ric * pts_i * -1.0 / (inv_dep_i * inv_dep_i);

    return true;
}
