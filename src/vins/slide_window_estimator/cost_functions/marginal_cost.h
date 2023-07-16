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
        /**
         * @param _discard_set 需要丢弃的参数
         * @param _cost_function 与需要丢弃的参数相关的代价函数
         * @param _parameter_blocks
         *  在_cost_function中的所有参数。注意，和需要丢弃的参数无cost边相连的参数块并不会卷入到边缘化过程中。
         * @param _loss_function 鲁棒函数
         * */
        MarginalMetaFactor(ceres::CostFunction *_cost_function,
                           ceres::LossFunction *_loss_function,
                           std::vector<double *> _parameter_blocks,
                           std::vector<int> _discard_set):
                cost_function_(_cost_function),
                loss_function_(_loss_function),
                parameter_blocks_(std::move(_parameter_blocks)),
                discard_set_(std::move(_discard_set)) {}

        void Evaluate();

        ceres::CostFunction *cost_function_;
        ceres::LossFunction *loss_function_;
        std::vector<double *> parameter_blocks_;
        std::vector<int> discard_set_;

        std::vector<JacobianType> jacobians_;
        Eigen::VectorXd residuals_;
    };

    class MarginalInfo {
    public:
        ~MarginalInfo();
        void addMetaFactor(const MarginalMetaFactor &marginal_meta_factor);

        /**
         * 将待优化变量按是否需要丢弃分为:
         * 1.丢弃量:需要从滑动窗口中丢弃的位置、姿态、速度、特征点深度
         *   只存在于下文的原始量当中，不存在于冻结量和更新量当中
         * 2.保留量:与丢弃量有约束边相连，但是尚不从滑动窗口中丢弃的量
         *   存在于原始量、冻结量和更新量当中
         * 3.无关量:既不需要从滑动窗口中丢弃，也不和丢弃量相连，这些量不卷入边缘化过程
         *   在原始量、冻结量更新量当中均不存在
         *
         * 将待优化变量在时间上分为：
         * 1.原始量x_original:
         *   addMetaFactor()和marginalize()时MarginalMetaFactor.parameter_blocks_对应的数据，此时优化过程尚未开始.
         *   原始量本质上是slide_window_estimator.cpp中的c_pos、c_vel等数组
         *   marginalize()会计算残差R和保留原始量的雅可比J，保存在reserve_param_residuals_和reserve_param_jacobians_中
         * 2.冻结量x_frozen:
         *   marginalize()会将保留原始量复制一份副本，储存在reserve_block_frozen_中。
         *   @param reserve_block_origin 持有保留原始量，且顺序上和reserve_block_frozen_一一对应，其内容同样是c_pos、c_vel等数组。
         * 3.更新量x_updated:
         *   保留原始量在窗口中滑动后被塞入MarginalCost，并在后续优化中不断更新，更新后的量称之为更新量。
         *   更新量本质上同样是slide_window_estimator.cpp中的c_pos、c_vel等数组，
         *   但由于滑动窗口的滑动，同一变量在更新量和原始量中的地址并不相同，sp_slide_addr_map描述了从原始量到更新量的地址映射。
         *
         * 优化过程：
         *   后续优化会通过x_updated与x_frozen的差来估计残差，即更新后的残差为 r = R + J * (x_updated - x_frozen)
         *   由于reserve_block_origin和sp_slide_addr_map构建起了冻结量和更新量之间的地址对应关系，这样的减法是可行的。
         * */
        void marginalize(std::vector<double *>& reserve_block_origin);

        std::vector<MarginalMetaFactor> factors_;
        int discard_dim_ = 0, reserve_dim_ = 0;

        std::vector<int> reserve_block_sizes_;      //原始数据维度，旋转为4维
        std::vector<int> reserve_block_ids_;        //切空间维度，旋转为3维
        std::vector<double *> reserve_block_frozen_;

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