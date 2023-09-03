#include <thread>
#include <unordered_map>
#include "log.h"
#include "vins/impl/vins_utils.h"
#include "marginal_cost.h"

using namespace vins;

inline int tangentSpaceDimensionSize(int size) {
    return size == 4 ? 3 : size;
}

void MarginalMetaFactor::Evaluate() {
    residuals_.resize(cost_function_->num_residuals());

    std::vector<int> block_sizes = cost_function_->parameter_block_sizes();
    std::vector<double*> raw_jacobians(block_sizes.size(), nullptr);
    jacobians_.resize(block_sizes.size());

    for (int i = 0; i < static_cast<int>(block_sizes.size()); i++) {
        jacobians_[i].resize(cost_function_->num_residuals(), block_sizes[i]);
        raw_jacobians[i] = jacobians_[i].data();
    }
    cost_function_->Evaluate(parameter_blocks_.data(), residuals_.data(), raw_jacobians.data());

    if (!loss_function_) {
        LOG_I("marginal factor not set loss function");
        return;
    }
    double rho[3];
    double residuals_norm2 = residuals_.squaredNorm();
    loss_function_->Evaluate(residuals_norm2, rho);
    double sqrt_rho1 = sqrt(rho[1]);

    double residual_scaling, alpha_sq_norm;
    if (abs(residuals_norm2) < MarginalInfo::EPS || (rho[2] <= 0.0)) {
        residual_scaling = sqrt_rho1;
        alpha_sq_norm = 0.0;
    } else {
        const double D = 1.0 + 2.0 * residuals_norm2 * rho[2] / rho[1];
        const double alpha = 1.0 - sqrt(D);
        residual_scaling = sqrt_rho1 / (1 - alpha);
        alpha_sq_norm = alpha / residuals_norm2;
    }

    const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(residuals_.size(), residuals_.size());
    const Eigen::MatrixXd k = sqrt_rho1 * (identity - alpha_sq_norm * residuals_ * residuals_.transpose());
    for (int i = 0; i < static_cast<int>(parameter_blocks_.size()); i++) {
        jacobians_[i] = k * jacobians_[i];
    }

    residuals_ *= residual_scaling;
}

MarginalInfo::~MarginalInfo() {
    for (auto & addr : reserve_block_frozen_)
        delete[] addr;

    for (auto & factor : factors_) {
        delete factor.cost_function_;
    }
}

void MarginalInfo::addMetaFactor(const MarginalMetaFactor &marginal_meta_factor) {
    factors_.emplace_back(marginal_meta_factor);
}

namespace {
struct ThreadsStruct {
    std::vector<MarginalMetaFactor> sub_factors;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
};
}

/**
 * 将待优化变量按是否需要丢弃分为:
 * 1.丢弃量:需要从滑动窗口中丢弃的位置、姿态、速度、特征点深度
 *   只存在于下文的原始量当中，不存在于冻结量和更新量当中
 * 2.保留量:与丢弃量有约束边相连，但是尚不从滑动窗口中丢弃的量
 *   存在于原始量、冻结量和更新量当中
 * 3.无关量:既不需要从滑动窗口中丢弃，也不和丢弃量相连，这些量不卷入边缘化过程
 *   在原始量、冻结量、更新量当中均不存在
 *
 * 将待优化变量在时间上分为：
 * 1.原始量x_original:
 *   上轮优化过程结束后，边缘化过程中，即addMetaFactor()和marginalize()时对应的数据.
 *   保存于@param MarginalMetaFactor.parameter_blocks_，指向slide_window_estimator.cpp中c_pos、c_vel等数组。
 *   SlideWindowEstimator负责new/delete。
 *   保留原始量即为原始量中需要保留的参数，保存于@param s_reserve_block_origin
 * 2.更新量x_updated:
 *   保留原始量在窗口中滑动后被塞入MarginalCost，并在下一轮优化中不断更新，更新后的量称之为更新量。
 *   更新量在MarginalCost::Evaluate()中通过@param parameters 传入，本质上同样是c_pos、c_vel等数组。
 *   由于窗口的滑动，同一变量在更新量和原始量中的地址并不相同。
 *   SlideWindowEstimator::slide()中的slide_addr_map描述了从保留原始量到更新量的地址映射。
 * 3.冻结量x_frozen:
 *   由于c_pos、c_vel中的数据在下轮优化中不断更新，marginalize()生成一份保留原始量的快照，称之为冻结量，在下轮优化中保持不变。
 *   冻结量保存于@param reserve_block_frozen_，MarginalInfo负责new/delete。
 *   由于冻结量需要在MarginalCost::Evaluate()中和原始量做差，因此冻结量和保留原始量的顺序必须一一对应
 *
 * 优化过程：
 *   marginalize()计算残差R和保留原始量的雅可比J，储存于@param reserve_param_residuals_、@param reserve_param_jacobians_
 *   后续优化会通过x_updated与x_frozen的差来估计残差，即更新后的残差为 r = R + J * (x_updated - x_frozen)
 *   由于reserve_block_origin和sp_slide_addr_map构建起了冻结量和更新量之间的地址对应关系，这样的减法是可行的。
 * */
void MarginalInfo::marginalize(std::vector<double *>& reserve_block_origin) {
    /**
     * STEP 1:
     * 参数块的维度记为dim，长度记为size，例如四元数的维度为3，长度为4
     * 将所有原始量参数化后视作一个整体向量，要丢弃的参数块在前，要保留的参数块在后。
     * @return 各参数块在该向量中的长度和位置，即@param origin_addr_to_size 和 @param origin_addr_to_idx
     * NOTE:位置是维度的前缀和，而非长度的前缀和
     * @param discard_param_dim_ 所有要丢弃的参数块的维度的和
     * @param reserve_param_dim_ 所有要保留的参数块的维度的和
     * */
    std::unordered_map<double*, int> origin_addr_to_size;
    std::unordered_map<double*, int> origin_addr_to_idx;
    std::set<double*> param_should_discard;
    for (const auto &factor:factors_) {
        const std::vector<double *> &parameter_blocks = factor.parameter_blocks_;
        for (const int discard_block_id:factor.discard_set_) {
            int cur_discard_param_dim_raw = factor.cost_function_->parameter_block_sizes()[discard_block_id];
            discard_dim_ += tangentSpaceDimensionSize(cur_discard_param_dim_raw);
            param_should_discard.insert(parameter_blocks[discard_block_id]);
        }
    }
    int discard_idx = 0, reserve_idx = discard_dim_;
    for (const auto &factor:factors_) {
        const std::vector<double *> &param_blocks = factor.parameter_blocks_;
        const std::vector<int> &param_sizes = factor.cost_function_->parameter_block_sizes();
        for (int i = 0; i < static_cast<int>(param_blocks.size()); i++) {
            double *block_addr = param_blocks[i];
            int size = param_sizes[i];
            origin_addr_to_size[block_addr] = size;
            if (param_should_discard.count(block_addr) == 0) {
                origin_addr_to_idx[block_addr] = reserve_idx;
                reserve_idx += tangentSpaceDimensionSize(size);
            } else {
                origin_addr_to_idx[block_addr] = discard_idx;
                discard_idx += tangentSpaceDimensionSize(size);
            }
        }
    }
    reserve_dim_ = reserve_idx - discard_dim_;

    /**
     * STEP 2:
     * 生成冻结量和保留原始量及其对应的idx和size
     * @return reserve_block_frozen_、reserve_block_origin、
     * */
    reserve_block_sizes_.clear();
    reserve_block_ids_.clear();
    reserve_block_frozen_.clear();
    reserve_block_origin.clear();
    for (const auto& it: origin_addr_to_idx) {
        double * addr = it.first;
        if (param_should_discard.count(addr)) {
            continue;
        }
        int size = origin_addr_to_size[addr];
        int idx = it.second;
        auto *frozen_data = new double [size];
        memcpy(frozen_data, it.first, sizeof(double) * size);
        reserve_block_frozen_.emplace_back(frozen_data);
        reserve_block_origin.emplace_back(addr);
        reserve_block_sizes_.emplace_back(size);
        reserve_block_ids_.emplace_back(idx);
    }

    /**
     * STEP 3:
     * 求解边缘化时刻各个子factor的雅可比与残差
     * */
    for (MarginalMetaFactor &it: factors_) {
        it.Evaluate();
    }

    /**
     * STEP 4:
     * 多线程计算边缘化之前的最小二乘方程Ax=b
     * */
    int total_dim = reserve_dim_ + discard_dim_;
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(total_dim, total_dim);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(total_dim);

    std::vector<std::thread> threads;
    constexpr int NUM_THREADS = 4;
    ThreadsStruct thread_structs[NUM_THREADS];
    for (int j = 0; j < factors_.size(); ++j) {
        thread_structs[j % NUM_THREADS].sub_factors.emplace_back(factors_[j]);
    }
    for (ThreadsStruct & thread_struct : thread_structs) {
        thread_struct.A = Eigen::MatrixXd::Zero(total_dim, total_dim);
        thread_struct.b = Eigen::VectorXd::Zero(total_dim);
        threads.emplace_back([&thread_struct, &origin_addr_to_size, &origin_addr_to_idx]() {
            for (const MarginalMetaFactor& it: thread_struct.sub_factors) {
                for (int i = 0; i < static_cast<int>(it.parameter_blocks_.size()); i++) {
                    int idx_i = origin_addr_to_idx[it.parameter_blocks_[i]];
                    int size_i = tangentSpaceDimensionSize(origin_addr_to_size[it.parameter_blocks_[i]]);
                    Eigen::MatrixXd jacobian_i = it.jacobians_[i].leftCols(size_i);
                    for (int j = i + 1; j < static_cast<int>(it.parameter_blocks_.size()); j++) {
                        int idx_j = origin_addr_to_idx[it.parameter_blocks_[j]];
                        int size_j = tangentSpaceDimensionSize(origin_addr_to_size[it.parameter_blocks_[j]]);
                        Eigen::MatrixXd jacobian_j = it.jacobians_[j].leftCols(size_j);
                        thread_struct.A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                        thread_struct.A.block(idx_j, idx_i, size_j, size_i) =
                                thread_struct.A.block(idx_i, idx_j, size_i, size_j).transpose();
                    }
                    thread_struct.A.block(idx_i, idx_i, size_i, idx_i) += jacobian_i.transpose() * jacobian_i;
                    thread_struct.b.segment(idx_i, size_i) += jacobian_i.transpose() * it.residuals_;
                }
            }
        });
    }
    for (int i = NUM_THREADS - 1; i >= 0; i--) {
        threads[i].join();
        A += thread_structs[i].A;
        b += thread_structs[i].b;
    }

    /**
     * STEP 5:
     * 边缘化，计算残差和雅可比
     * @param reserve_param_residuals_ 边缘化时刻要保留的参数对应的b-Ax
     * @param reserve_param_jacobians_ 边缘化时刻要保留的参数相对于residual的雅可比
     * */
    const Eigen::VectorXd bmm = b.segment(0, discard_dim_);
    const Eigen::VectorXd brr = b.segment(discard_dim_, reserve_dim_);
    const Eigen::MatrixXd Amm = A.block(0, 0, discard_dim_, discard_dim_);
    const Eigen::MatrixXd Amr = A.block(0, discard_dim_, discard_dim_, reserve_dim_);
    const Eigen::MatrixXd Arm = A.block(discard_dim_, 0, reserve_dim_, discard_dim_);
    const Eigen::MatrixXd Arr = A.block(discard_dim_, discard_dim_, reserve_dim_, reserve_dim_);

    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(0.5 * (Amm + Amm.transpose()));
    const auto eigen_values = saes.eigenvalues().array();
    const Eigen::VectorXd eigen_val_inv_vec = (eigen_values > EPS).select(eigen_values.inverse(), 0);
    const Eigen::MatrixXd eigen_val_inv_mat = eigen_val_inv_vec.asDiagonal();
    const Eigen::MatrixXd Amm_inv = saes.eigenvectors() * eigen_val_inv_mat * saes.eigenvectors().transpose();

    A = Arr - Arm * Amm_inv * Amr;
    b = brr - Arm * Amm_inv * bmm;

    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(A);
    const auto eigen_values2 = saes2.eigenvalues().array();
    const Eigen::VectorXd S = (eigen_values2 > EPS).select(eigen_values2, 0);
    const Eigen::VectorXd S_inv = (eigen_values2 > EPS).select(eigen_values2.inverse(), 0);

    const Eigen::VectorXd S_sqrt = S.cwiseSqrt();
    const Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();

    reserve_block_jacobians_ = S_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
    reserve_block_residuals_ = S_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose() * b;
}

// marginal_info_生命周期由SlideWindowEstimator负责维护
MarginalCost::MarginalCost(MarginalInfo* _marginal_info)
 : marginal_info_(_marginal_info) {
    for (auto reserve_block_size: marginal_info_->reserve_block_sizes_) {
        mutable_parameter_block_sizes()->push_back(reserve_block_size);
    }
    set_num_residuals(marginal_info_->reserve_dim_);
}

bool MarginalCost::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
    int reserve_dim = marginal_info_->reserve_dim_;
    int discard_dim = marginal_info_->discard_dim_;

    Eigen::VectorXd dx(reserve_dim);
    for (int i = 0; i < static_cast<int>(marginal_info_->reserve_block_sizes_.size()); i++) {
        int size = marginal_info_->reserve_block_sizes_[i];
        int idx = marginal_info_->reserve_block_ids_[i] - discard_dim;
        Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(parameters[i], size);
        Eigen::VectorXd x0 = Eigen::Map<const Eigen::VectorXd>(marginal_info_->reserve_block_frozen_[i], size);
        if (size != 4)
            dx.segment(idx, size) = x - x0;
        else {
            Eigen::Quaterniond q1 = Eigen::Quaterniond(x0(3), x0(0), x0(1), x0(2)).inverse();
            Eigen::Quaterniond q2 = Eigen::Quaterniond(x(3), x(0), x(1), x(2));
            if ((q1 * q2).w() < 0) {
                dx.segment<3>(idx) = 2.0 * -utils::positify(q1 * q2).vec();
            } else {
                dx.segment<3>(idx) = 2.0 * utils::positify(q1 * q2).vec();
            }
        }
    }
    Eigen::Map<Eigen::VectorXd>(residuals, reserve_dim) =
            marginal_info_->reserve_block_residuals_ + marginal_info_->reserve_block_jacobians_ * dx;

    assert(jacobians);
    for (int i = 0; i < static_cast<int>(marginal_info_->reserve_block_sizes_.size()); i++) {
        assert(jacobians[i]);
        int size = marginal_info_->reserve_block_sizes_[i];
        int local_size = tangentSpaceDimensionSize(size);
        int idx = marginal_info_->reserve_block_ids_[i] - discard_dim;
        Eigen::Map<JacobianType> jacobian(jacobians[i], reserve_dim, size);
        jacobian.setZero();
        jacobian.leftCols(local_size) =
                marginal_info_->reserve_block_jacobians_.middleCols(idx, local_size);
    }
    return true;
}
