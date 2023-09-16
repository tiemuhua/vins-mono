#include "imu_cost.h"
#include "vins/impl/vins_utils.h"

namespace vins {

    template<typename T>
    // Eigen::Map<Eigen::Matrix<double, 15, row, Eigen::RowMajor>>
    bool checkJacobianNumericalStable(const T &jacobian, const char *jacobian_name) {
        if (jacobian.maxCoeff() > 1e8 || jacobian.minCoeff() < -1e8) {
            LOG(ERROR) << "numerical unstable in pre-integral, name:" << jacobian_name
                       << "\tjacobian_ value:" << utils::eigen2string(jacobian);
            return false;
        }
        return true;
    }

    /**
     * @param parameters 顺序与StateOrder相同，pos_i、quat_i、vel_i、ba_i、bg_i、pos_j、quat_j、vel_j、ba_j、bg_j
     * */
    bool IMUCost::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {

        const Eigen::Vector3d pos_i = utils::array2vec3d(parameters[0]);
        const Eigen::Quaterniond quat_i = utils::array2quat(parameters[1]);
        const Eigen::Vector3d vel_i = utils::array2vec3d(parameters[2]);
        const Eigen::Vector3d ba_i = utils::array2vec3d(parameters[3]);
        const Eigen::Vector3d bg_i = utils::array2vec3d(parameters[4]);

        const Eigen::Vector3d pos_j = utils::array2vec3d(parameters[5]);
        const Eigen::Quaterniond quat_j = utils::array2quat(parameters[6]);
        const Eigen::Vector3d vel_j = utils::array2vec3d(parameters[7]);
        const Eigen::Vector3d ba_j = utils::array2vec3d(parameters[8]);
        const Eigen::Vector3d bg_j = utils::array2vec3d(parameters[9]);

        Eigen::Map<ImuIntegrator::State> residual(residuals);
        residual = pre_integral_.evaluate(pos_i, quat_i, vel_i, ba_i, bg_i,
                                          pos_j, quat_j, vel_j, ba_j, bg_j);

        ImuIntegrator::SqrtInfo sqrt_info =
                Eigen::LLT<ImuIntegrator::Covariance>(pre_integral_.getCovariance().inverse()).matrixL().transpose();
        residual = sqrt_info * residual;

        assert(jacobians);
        assert(jacobians[0]);
        assert(jacobians[1]);
        assert(jacobians[2]);
        assert(jacobians[3]);
        assert(jacobians[4]);
        assert(jacobians[5]);
        assert(jacobians[6]);
        assert(jacobians[7]);
        assert(jacobians[8]);
        assert(jacobians[9]);

        double sum_dt = pre_integral_.deltaTime();
        Eigen::Matrix3d dp_dba = pre_integral_.getJacobian().template block<3, 3>(kOrderPos, kOrderBA);
        Eigen::Matrix3d dp_dbg = pre_integral_.getJacobian().template block<3, 3>(kOrderPos, kOrderBG);
        Eigen::Matrix3d dq_dbg = pre_integral_.getJacobian().template block<3, 3>(kOrderRot, kOrderBG);
        Eigen::Matrix3d dv_dba = pre_integral_.getJacobian().template block<3, 3>(kOrderVel, kOrderBA);
        Eigen::Matrix3d dv_dbg = pre_integral_.getJacobian().template block<3, 3>(kOrderVel, kOrderBG);

        checkJacobianNumericalStable(pre_integral_.getJacobian(), "origin jacobian");

        typedef Eigen::Matrix<double, 15, 3, Eigen::RowMajor> Mat15_3;
        typedef Eigen::Matrix<double, 15, 4, Eigen::RowMajor> Mat15_4;
        Eigen::Map<Mat15_3> jacobian_pos_i(jacobians[0]);
        Eigen::Map<Mat15_4> jacobian_quat_i(jacobians[1]);
        Eigen::Map<Mat15_3> jacobian_vel_i(jacobians[2]);
        Eigen::Map<Mat15_3> jacobian_ba_i(jacobians[3]);
        Eigen::Map<Mat15_3> jacobian_bg_i(jacobians[4]);

        Eigen::Map<Mat15_3> jacobian_pos_j(jacobians[5]);
        Eigen::Map<Mat15_4> jacobian_quat_j(jacobians[6]);
        Eigen::Map<Mat15_3> jacobian_vel_j(jacobians[7]);
        Eigen::Map<Mat15_3> jacobian_ba_j(jacobians[8]);
        Eigen::Map<Mat15_3> jacobian_bg_j(jacobians[9]);

        jacobian_pos_i = Mat15_3::Zero();
        jacobian_quat_i = Mat15_4::Zero();
        jacobian_vel_i = Mat15_3::Zero();
        jacobian_ba_i = Mat15_3::Zero();
        jacobian_bg_i = Mat15_3::Zero();

        jacobian_pos_j = Mat15_3::Zero();
        jacobian_quat_j = Mat15_4::Zero();
        jacobian_vel_j = Mat15_3::Zero();
        jacobian_ba_j = Mat15_3::Zero();
        jacobian_bg_j = Mat15_3::Zero();

        const Eigen::Vector3d G = {0, 0, -9.81};// todo tiemuhuaguo G应该用标定后的值
        const Eigen::Vector3d delta_pos = 0.5 * G * sum_dt * sum_dt + pos_j - pos_i - vel_i * sum_dt;
        const Eigen::Vector3d delta_vel = G * sum_dt + vel_j - vel_i;
        const Eigen::Quaterniond delta_quat_inv = quat_j.inverse() * quat_i;
        const Eigen::Quaterniond bg_correction =
                pre_integral_.deltaQuat() * utils::deltaQ(dq_dbg * (bg_i - pre_integral_.getBg()));

        jacobian_pos_i.block<3, 3>(kOrderPos, 0) = -quat_i.inverse().toRotationMatrix();

        jacobian_quat_i.block<3, 3>(kOrderPos, 0) = utils::skewSymmetric(quat_i.inverse() * delta_pos);
        jacobian_quat_i.block<3, 3>(kOrderRot, 0) =
                -(utils::Qleft(delta_quat_inv) * utils::Qright(bg_correction)).bottomRightCorner<3, 3>();
        jacobian_quat_i.block<3, 3>(kOrderVel, 0) = utils::skewSymmetric(quat_i.inverse() * delta_vel);

        jacobian_vel_i.block<3, 3>(kOrderPos, 0) = -quat_i.inverse().toRotationMatrix() * sum_dt;
        jacobian_vel_i.block<3, 3>(kOrderVel, 0) = -quat_i.inverse().toRotationMatrix();

        jacobian_ba_i.block<3, 3>(kOrderPos, 0) = -dp_dba;
        jacobian_ba_i.block<3, 3>(kOrderVel, 0) = -dv_dba;
        jacobian_ba_i.block<3, 3>(kOrderBA, 0) = -Eigen::Matrix3d::Identity();

        jacobian_bg_i.block<3, 3>(kOrderPos, 0) = -dp_dbg;
        jacobian_bg_i.block<3, 3>(kOrderRot, 0) =
                -utils::Qleft(delta_quat_inv * pre_integral_.deltaQuat()).bottomRightCorner<3, 3>() * dq_dbg;
        jacobian_bg_i.block<3, 3>(kOrderVel, 0) = -dv_dbg;
        jacobian_bg_i.block<3, 3>(kOrderBG, 0) = -Eigen::Matrix3d::Identity();

        jacobian_pos_j.block<3, 3>(kOrderPos, 0) = quat_i.inverse().toRotationMatrix();

        jacobian_quat_j.block<3, 3>(kOrderRot, 0) = utils::Qleft(
                bg_correction.inverse() * quat_i.inverse() * quat_j).bottomRightCorner<3, 3>();

        jacobian_vel_j.block<3, 3>(kOrderVel, 0) = quat_i.inverse().toRotationMatrix();

        jacobian_ba_j.block<3, 3>(kOrderBA, 0) = Eigen::Matrix3d::Identity();

        jacobian_bg_j.block<3, 3>(kOrderBG, 0) = Eigen::Matrix3d::Identity();

        jacobian_pos_i = sqrt_info * jacobian_pos_i;
        jacobian_quat_i = sqrt_info * jacobian_quat_i;
        jacobian_vel_i = sqrt_info * jacobian_vel_i;
        jacobian_ba_i = sqrt_info * jacobian_ba_i;
        jacobian_bg_i = sqrt_info * jacobian_bg_i;

        jacobian_pos_j = sqrt_info * jacobian_pos_j;
        jacobian_quat_j = sqrt_info * jacobian_quat_j;
        jacobian_vel_j = sqrt_info * jacobian_vel_j;
        jacobian_ba_j = sqrt_info * jacobian_ba_j;
        jacobian_bg_j = sqrt_info * jacobian_bg_j;

        checkJacobianNumericalStable(jacobian_pos_i, "jacobian_pos_i");
        checkJacobianNumericalStable(jacobian_quat_i, "jacobian_quat_i");
        checkJacobianNumericalStable(jacobian_vel_i, "jacobian_vel_i");
        checkJacobianNumericalStable(jacobian_ba_i, "jacobian_ba_i");
        checkJacobianNumericalStable(jacobian_bg_i, "jacobian_bg_i");
        checkJacobianNumericalStable(jacobian_pos_j, "jacobian_pos_j");
        checkJacobianNumericalStable(jacobian_quat_j, "jacobian_quat_j");
        checkJacobianNumericalStable(jacobian_vel_j, "jacobian_vel_j");
        checkJacobianNumericalStable(jacobian_ba_j, "jacobian_ba_j");
        checkJacobianNumericalStable(jacobian_bg_j, "jacobian_bg_j");

        return true;
    }

}