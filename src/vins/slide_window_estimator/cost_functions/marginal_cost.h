#pragma once

#include <cstdlib>
#include <ceres/ceres.h>
#include <utility>

namespace vins {

    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> JacobianType;

    /**
     * MarginalFactor中的核心数据储存在MarginalInfo中。
     * MarginalInfo由多个多个ResidualBlockInfo组成。
     * ResidualBlockInfo包括当前帧的预积分信息、视觉信息和之前帧的边缘化信息，分别对应不同的cost_function_
     * */

    struct MarginalMetaFactor {
        MarginalMetaFactor(ceres::CostFunction *_cost_function,
                           ceres::LossFunction *_loss_function,
                           std::vector<double *> _parameter_blocks,
                           std::vector<int> _drop_set):
                cost_function_(_cost_function),
                loss_function_(_loss_function),
                parameter_blocks_(std::move(_parameter_blocks)),
                drop_set_(std::move(_drop_set)) {}

        void Evaluate();

        ceres::CostFunction *cost_function_;
        ceres::LossFunction *loss_function_;
        std::vector<double *> parameter_blocks_;
        std::vector<int> drop_set_;

        std::vector<JacobianType> jacobians_;
        Eigen::VectorXd residuals_;
    };

    class MarginalInfo {
    public:
        ~MarginalInfo();
        void addMetaFactor(const MarginalMetaFactor &residual_block_info);
        void marginalize();
        std::vector<double *> getReserveParamBlocksWithCertainOrder() const;

        std::vector<MarginalMetaFactor> factors_;
        int discard_param_dim_ = 0, reserve_param_dim_ = 0;

        std::vector<int> reserve_block_sizes_;      //原始数据维度，旋转为4维
        std::vector<int> reserve_block_ids_;        //切空间维度，旋转为3维
        std::vector<double *> reserve_block_data_frozen_;
        std::vector<double *> reserve_block_addr_origin_;

        Eigen::MatrixXd reserve_param_jacobians_;
        Eigen::VectorXd reserve_param_residuals_;
        static constexpr double EPS = 1e-8;
    };

    class MarginalCost : public ceres::CostFunction {
    public:
        explicit MarginalCost(MarginalInfo* _marginal_info);
        bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;

        // marginal_info_生命周期由SlideWindowEstimator负责维护
        MarginalInfo* marginal_info_;
    };
}