#include "imu_factor.h"
#include "vins/vins_utils.h"

namespace vins{
    bool IMUCost::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {

        Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
        Eigen::Quaterniond Qi(parameters[0][6], parameters[0][3], parameters[0][4], parameters[0][5]);

        Eigen::Vector3d Vi(parameters[1][0], parameters[1][1], parameters[1][2]);
        Eigen::Vector3d Bai(parameters[1][3], parameters[1][4], parameters[1][5]);
        Eigen::Vector3d Bgi(parameters[1][6], parameters[1][7], parameters[1][8]);

        Eigen::Vector3d Pj(parameters[2][0], parameters[2][1], parameters[2][2]);
        Eigen::Quaterniond Qj(parameters[2][6], parameters[2][3], parameters[2][4], parameters[2][5]);

        Eigen::Vector3d Vj(parameters[3][0], parameters[3][1], parameters[3][2]);
        Eigen::Vector3d Baj(parameters[3][3], parameters[3][4], parameters[3][5]);
        Eigen::Vector3d Bgj(parameters[3][6], parameters[3][7], parameters[3][8]);

        Eigen::Map<ImuIntegrator::State> residual(residuals);
        residual = pre_integration.evaluate(Pi, Qi, Vi, Bai, Bgi,
                                             Pj, Qj, Vj, Baj, Bgj);

        Eigen::Matrix<double, 15, 15> sqrt_info =
                Eigen::LLT<ImuIntegrator::Covariance>(pre_integration.getCovariance().inverse()).matrixL().transpose();
        residual = sqrt_info * residual;

        assert(jacobians);
        assert(jacobians[0]);
        assert(jacobians[1]);
        assert(jacobians[2]);
        assert(jacobians[3]);

        double sum_dt = pre_integration.deltaTime();
        Eigen::Matrix3d dp_dba = pre_integration.getJacobian().template block<3, 3>(O_P, O_BA);
        Eigen::Matrix3d dp_dbg = pre_integration.getJacobian().template block<3, 3>(O_P, O_BG);
        Eigen::Matrix3d dq_dbg = pre_integration.getJacobian().template block<3, 3>(O_R, O_BG);
        Eigen::Matrix3d dv_dba = pre_integration.getJacobian().template block<3, 3>(O_V, O_BA);
        Eigen::Matrix3d dv_dbg = pre_integration.getJacobian().template block<3, 3>(O_V, O_BG);

        if (pre_integration.getJacobian().maxCoeff() > 1e8 || pre_integration.getJacobian().minCoeff() < -1e8) {
            LOG_W("numerical unstable in pre-integration");
        }

        Eigen::Quaterniond corrected_delta_q =
                pre_integration.deltaQuat() * utils::deltaQ(dq_dbg * (Bgi - pre_integration.getBg()));

        // jacobian_pose_i
        const Eigen::Vector3d G = {0,0,-9.81};// todo tiemuhuaguo G应该用标定后的值
        Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
        jacobian_pose_i.setZero();
        jacobian_pose_i.block<3, 3>(O_P, O_P) = -Qi.inverse().toRotationMatrix();
        jacobian_pose_i.block<3, 3>(O_P, O_R) = utils::skewSymmetric(
                Qi.inverse() * (0.5 * G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt));
        jacobian_pose_i.block<3, 3>(O_R, O_R) =
                -(utils::Qleft(Qj.inverse() * Qi) * utils::Qright(corrected_delta_q)).bottomRightCorner<3, 3>();
        jacobian_pose_i.block<3, 3>(O_V, O_R) = utils::skewSymmetric(Qi.inverse() * (G * sum_dt + Vj - Vi));
        jacobian_pose_i = sqrt_info * jacobian_pose_i;
        if (jacobian_pose_i.maxCoeff() > 1e8 || jacobian_pose_i.minCoeff() < -1e8) {
            LOG_W("numerical unstable in pre-integration");
        }

        // jacobian_speed_bias_i
        Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_speed_bias_i(jacobians[1]);
        jacobian_speed_bias_i.setZero();
        jacobian_speed_bias_i.block<3, 3>(O_P, O_V - O_V) = -Qi.inverse().toRotationMatrix() * sum_dt;
        jacobian_speed_bias_i.block<3, 3>(O_P, O_BA - O_V) = -dp_dba;
        jacobian_speed_bias_i.block<3, 3>(O_P, O_BG - O_V) = -dp_dbg;
        jacobian_speed_bias_i.block<3, 3>(O_R, O_BG - O_V) =
                -utils::Qleft(Qj.inverse() * Qi * pre_integration.deltaQuat()).bottomRightCorner<3, 3>() * dq_dbg;
        jacobian_speed_bias_i.block<3, 3>(O_V, O_V - O_V) = -Qi.inverse().toRotationMatrix();
        jacobian_speed_bias_i.block<3, 3>(O_V, O_BA - O_V) = -dv_dba;
        jacobian_speed_bias_i.block<3, 3>(O_V, O_BG - O_V) = -dv_dbg;
        jacobian_speed_bias_i.block<3, 3>(O_BA, O_BA - O_V) = -Eigen::Matrix3d::Identity();
        jacobian_speed_bias_i.block<3, 3>(O_BG, O_BG - O_V) = -Eigen::Matrix3d::Identity();
        jacobian_speed_bias_i = sqrt_info * jacobian_speed_bias_i;

        // jacobian_pose_j
        Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[2]);
        jacobian_pose_j.setZero();
        jacobian_pose_j.block<3, 3>(O_P, O_P) = Qi.inverse().toRotationMatrix();
        jacobian_pose_j.block<3, 3>(O_R, O_R) = utils::Qleft(
                corrected_delta_q.inverse() * Qi.inverse() * Qj).bottomRightCorner<3, 3>();
        jacobian_pose_j = sqrt_info * jacobian_pose_j;

        // jacobian_speed_bias_j
        Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_speed_bias_j(jacobians[3]);
        jacobian_speed_bias_j.setZero();
        jacobian_speed_bias_j.block<3, 3>(O_V, O_V - O_V) = Qi.inverse().toRotationMatrix();
        jacobian_speed_bias_j.block<3, 3>(O_BA, O_BA - O_V) = Eigen::Matrix3d::Identity();
        jacobian_speed_bias_j.block<3, 3>(O_BG, O_BG - O_V) = Eigen::Matrix3d::Identity();
        jacobian_speed_bias_j = sqrt_info * jacobian_speed_bias_j;

        return true;
    }

}