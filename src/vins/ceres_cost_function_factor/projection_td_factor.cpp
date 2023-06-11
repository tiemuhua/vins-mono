#include "projection_td_factor.h"
#include "log.h"
#include "vins/parameters.h"
#include "vins/vins_utils.h"

using namespace vins;

Eigen::Matrix2d ProjectionTdFactor::sqrt_info;
double ProjectionTdFactor::sum_t;

ProjectionTdFactor::ProjectionTdFactor(const FeaturePoint2D& p1, const FeaturePoint2D& p2)
                                       : td_i(p1.time_stamp), td_j(p2.time_stamp) {
    pts_i = Eigen::Vector3d(p1.point.x, p1.point.y, 1.0);
    pts_j = Eigen::Vector3d(p2.point.x, p2.point.y, 1.0);
    velocity_i = Eigen::Vector3d(p1.velocity.x, p1.velocity.y, 0.0);
    velocity_j = Eigen::Vector3d(p2.velocity.x, p2.velocity.y, 0.0);
    row_i = p1.point.y - Param::Instance().camera.row / 2;
    row_j = p2.point.y - Param::Instance().camera.row / 2;

#ifdef UNIT_SPHERE_ERROR
    Eigen::Vector3d b1, b2;
    Eigen::Vector3d a = pts_j.normalized();
    Eigen::Vector3d tmp(0, 0, 1);
    if(a == tmp)
        tmp << 1, 0, 0;
    b1 = (tmp - a * (a.transpose() * tmp)).normalized();
    b2 = a.cross(b1);
    tangent_base.block<1, 3>(0, 0) = b1.transpose();
    tangent_base.block<1, 3>(1, 0) = b2.transpose();
#endif
};

bool ProjectionTdFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
    Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Eigen::Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
    Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    Eigen::Vector3d tic(parameters[2][0], parameters[2][1], parameters[2][2]);
    Eigen::Quaterniond qic(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);

    double inv_dep_i = parameters[3][0];

    double td = parameters[4][0];

    Eigen::Vector3d pts_i_td = pts_i - (td - td_i + Param::Instance().getTimeShatPerRol() * row_i) * velocity_i;
    Eigen::Vector3d pts_j_td = pts_j - (td - td_j + Param::Instance().getTimeShatPerRol() * row_j) * velocity_j;
    Eigen::Vector3d pts_camera_i = pts_i_td / inv_dep_i;
    Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic;
    Eigen::Vector3d pts_w = Qi * pts_imu_i + Pi;
    Eigen::Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj);
    Eigen::Vector3d pts_camera_j = qic.inverse() * (pts_imu_j - tic);
    Eigen::Map <Eigen::Vector2d> residual(residuals);

#ifdef UNIT_SPHERE_ERROR
    residual =  tangent_base * (pts_camera_j.normalized() - pts_j_td.normalized());
#else
    double dep_j = pts_camera_j.z();
    residual = (pts_camera_j / dep_j).head<2>() - pts_j_td.head<2>();
#endif

    residual = sqrt_info * residual;
    assert(jacobians && jacobians[0]&& jacobians[1]&& jacobians[2]&& jacobians[3] && jacobians[4]);
    if (!(jacobians && jacobians[0]&& jacobians[1]&& jacobians[2]&& jacobians[3] && jacobians[4])) {
        LOG_E("!jacobian");
        return false;
    }

    Eigen::Matrix3d Ri = Qi.toRotationMatrix();
    Eigen::Matrix3d Rj = Qj.toRotationMatrix();
    Eigen::Matrix3d ric = qic.toRotationMatrix();
    Eigen::Matrix<double, 2, 3> reduce(2, 3);
#ifdef UNIT_SPHERE_ERROR
    double norm = pts_camera_j.norm();
    double norm3 = pow(norm, 3);
    Eigen::Matrix3d norm_jacobian = Eigen::Matrix3d::Identity() / norm - pts_camera_j * pts_camera_j.transpose() / norm3;
    reduce = tangent_base * norm_jacobian;
#else
    reduce << 1. / dep_j, 0, -pts_camera_j(0) / (dep_j * dep_j),0, 1. / dep_j, -pts_camera_j(1) / (dep_j * dep_j);
#endif
    reduce = sqrt_info * reduce;

    typedef Eigen::Matrix<double, 2, 7, Eigen::RowMajor> Mat27Row;
    typedef Eigen::Matrix<double, 3, 6> mat36;
    typedef Eigen::Matrix<double, 2, 3> mat23;
    Eigen::Map <Mat27Row> jacobian_pose_i(jacobians[0]);
    mat36 jaco_i;
    jaco_i.leftCols<3>() = ric.transpose() * Rj.transpose();
    jaco_i.rightCols<3>() = ric.transpose() * Rj.transpose() * Ri * - utils::skewSymmetric(pts_imu_i);
    jacobian_pose_i.leftCols<6>() = reduce * jaco_i;
    jacobian_pose_i.rightCols<1>().setZero();

    Eigen::Map <Mat27Row> jacobian_pose_j(jacobians[1]);
    mat36 jaco_j;
    jaco_j.leftCols<3>() = ric.transpose() * -Rj.transpose();
    jaco_j.rightCols<3>() = ric.transpose() * utils::skewSymmetric(pts_imu_j);
    jacobian_pose_j.leftCols<6>() = reduce * jaco_j;
    jacobian_pose_j.rightCols<1>().setZero();

    Eigen::Map <Mat27Row> jacobian_ex_pose(jacobians[2]);
    mat36 jaco_ex;
    jaco_ex.leftCols<3>() = ric.transpose() * (Rj.transpose() * Ri - Eigen::Matrix3d::Identity());
    Eigen::Matrix3d rot_diff_in_camera_frame = ric.transpose() * Rj.transpose() * Ri * ric;
    jaco_ex.rightCols<3>() =
            -rot_diff_in_camera_frame * utils::skewSymmetric(pts_camera_i)
            + utils::skewSymmetric(rot_diff_in_camera_frame * pts_camera_i)
            + utils::skewSymmetric(ric.transpose() * (Rj.transpose() * (Ri * tic + Pi - Pj) - tic));
    jacobian_ex_pose.leftCols<6>() = reduce * jaco_ex;
    jacobian_ex_pose.rightCols<1>().setZero();

    mat23 r = reduce * ric.transpose() * Rj.transpose() * Ri * ric;
    Eigen::Map <Eigen::Vector2d> jacobian_feature(jacobians[3]);
    jacobian_feature = r * pts_i_td * -1.0 / (inv_dep_i * inv_dep_i);
    Eigen::Map <Eigen::Vector2d> jacobian_td(jacobians[4]);
    jacobian_td = r * velocity_i / inv_dep_i * -1.0 + sqrt_info * velocity_j.head(2);

    return true;
}

void ProjectionTdFactor::check(double **parameters) {
    double *res = new double[2];
    double **jaco = new double *[5];
    jaco[0] = new double[2 * 7];
    jaco[1] = new double[2 * 7];
    jaco[2] = new double[2 * 7];
    jaco[3] = new double[2 * 1];
    jaco[4] = new double[2 * 1];
    Evaluate(parameters, res, jaco);

    Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

    Eigen::Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
    Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

    Eigen::Vector3d tic(parameters[2][0], parameters[2][1], parameters[2][2]);
    Eigen::Quaterniond qic(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);
    double inv_dep_i = parameters[3][0];
    double td = parameters[4][0];

    Eigen::Vector3d pts_i_td, pts_j_td;
    pts_i_td = pts_i - (td - td_i + Param::Instance().getTimeShatPerRol() * row_i) * velocity_i;
    pts_j_td = pts_j - (td - td_j + Param::Instance().getTimeShatPerRol() * row_j) * velocity_j;
    Eigen::Vector3d pts_camera_i = pts_i_td / inv_dep_i;
    Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic;
    Eigen::Vector3d pts_w = Qi * pts_imu_i + Pi;
    Eigen::Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj);
    Eigen::Vector3d pts_camera_j = qic.inverse() * (pts_imu_j - tic);
    Eigen::Vector2d residual;

#ifdef UNIT_SPHERE_ERROR
    residual =  tangent_base * (pts_camera_j.normalized() - pts_j_td.normalized());
#else
    double dep_j = pts_camera_j.z();
    residual = (pts_camera_j / dep_j).head<2>() - pts_j_td.head<2>();
#endif
    residual = sqrt_info * residual;

    puts("num");
    std::cout << residual.transpose() << std::endl;

    const double eps = 1e-6;
    Eigen::Matrix<double, 2, 20> num_jacobian;
    for (int k = 0; k < 20; k++) {
        Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

        Eigen::Vector3d Pj(parameters[1][0], parameters[1][1], parameters[1][2]);
        Eigen::Quaterniond Qj(parameters[1][6], parameters[1][3], parameters[1][4], parameters[1][5]);

        Eigen::Vector3d tic(parameters[2][0], parameters[2][1], parameters[2][2]);
        Eigen::Quaterniond qic(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);
        double inv_dep_i = parameters[3][0];
        double td = parameters[4][0];


        int a = k / 3, b = k % 3;
        Eigen::Vector3d delta = Eigen::Vector3d(b == 0, b == 1, b == 2) * eps;

        if (a == 0)
            Pi += delta;
        else if (a == 1)
            Qi = Qi * utils::deltaQ(delta);
        else if (a == 2)
            Pj += delta;
        else if (a == 3)
            Qj = Qj * utils::deltaQ(delta);
        else if (a == 4)
            tic += delta;
        else if (a == 5)
            qic = qic * utils::deltaQ(delta);
        else if (a == 6 && b == 0)
            inv_dep_i += delta.x();
        else if (a == 6 && b == 1)
            td += delta.y();

        Eigen::Vector3d pts_i_td, pts_j_td;
        pts_i_td = pts_i - (td - td_i + Param::Instance().getTimeShatPerRol() * row_i) * velocity_i;
        pts_j_td = pts_j - (td - td_j + Param::Instance().getTimeShatPerRol() * row_j) * velocity_j;
        Eigen::Vector3d pts_camera_i = pts_i_td / inv_dep_i;
        Eigen::Vector3d pts_imu_i = qic * pts_camera_i + tic;
        Eigen::Vector3d pts_w = Qi * pts_imu_i + Pi;
        Eigen::Vector3d pts_imu_j = Qj.inverse() * (pts_w - Pj);
        Eigen::Vector3d pts_camera_j = qic.inverse() * (pts_imu_j - tic);
        Eigen::Vector2d tmp_residual;

#ifdef UNIT_SPHERE_ERROR
        tmp_residual =  tangent_base * (pts_camera_j.normalized() - pts_j_td.normalized());
#else
        double dep_j = pts_camera_j.z();
        tmp_residual = (pts_camera_j / dep_j).head<2>() - pts_j_td.head<2>();
#endif
        tmp_residual = sqrt_info * tmp_residual;

        num_jacobian.col(k) = (tmp_residual - residual) / eps;
    }
    std::cout << num_jacobian << std::endl;
}
