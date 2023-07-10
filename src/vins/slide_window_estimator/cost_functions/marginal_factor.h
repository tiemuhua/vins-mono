#pragma once

#include <cstdlib>
#include <ceres/ceres.h>
#include <unordered_map>
#include <utility>

namespace vins {

    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> JacobianType;

    /**
     * MarginalFactor中的核心数据储存在MarginalInfo中。
     * MarginalInfo由多个多个ResidualBlockInfo组成。
     * ResidualBlockInfo包括当前帧的预积分信息、视觉信息和之前帧的边缘化信息，分别对应不同的cost_function_
     * */

    struct ResidualBlockInfo {
        ResidualBlockInfo(ceres::CostFunction *_cost_function,
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

    typedef std::unordered_map<double*, int> DoublePtr2Int;
    typedef std::unordered_map<double*, double *> DoublePtr2DoublePtr;

    class MarginalInfo {
    public:
        ~MarginalInfo();
        void addResidualBlockInfo(const ResidualBlockInfo &residual_block_info);
        void preMarginalize();
        void marginalize();
        std::vector<double *> getParameterBlocks(const DoublePtr2DoublePtr &addr_shift);

        std::vector<ResidualBlockInfo> factors_;
        int old_param_dim_ = 0, new_param_dim_ = 0;
        DoublePtr2Int parameter_block_size_; //global size
        DoublePtr2Int parameter_block_idx_; //local size
        DoublePtr2DoublePtr parameter_block_data_;

        std::vector<int> keep_block_size_; //global size
        std::vector<int> keep_block_idx_;  //local size
        std::vector<double *> keep_block_data_;

        Eigen::MatrixXd linearized_jacobians_;
        Eigen::VectorXd linearized_residuals_;
        static constexpr double EPS = 1e-8;
    };

    class MarginalFactor : public ceres::CostFunction {
    public:
        explicit MarginalFactor(std::shared_ptr<MarginalInfo>  _marginal_info);
        bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;

        std::shared_ptr<MarginalInfo> marginal_info_;
    };
}