#include <thread>
#include <utility>
#include "log.h"
#include "vins_utils.h"
#include "marginal_factor.h"

using namespace vins;

static int localSize(int size) {
    return size == 7 ? 6 : size;
}

void ResidualBlockInfo::Evaluate() {
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

    for (auto & it : parameter_block_data_)
        delete[] it.second;

    for (auto & factor : factors_) {
        delete factor.cost_function_;
    }
}

void MarginalInfo::addResidualBlockInfo(const ResidualBlockInfo &residual_block_info) {
    factors_.emplace_back(residual_block_info);

    const std::vector<double *> &parameter_blocks = residual_block_info.parameter_blocks_;
    std::vector<int> parameter_block_sizes = residual_block_info.cost_function_->parameter_block_sizes();

    for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++) {
        double *addr = parameter_blocks[i];
        int size = parameter_block_sizes[i];
        parameter_block_size_[addr] = size;
    }

    for (int i : residual_block_info.drop_set_) {
        double *addr = parameter_blocks[i];
        parameter_block_idx_[addr] = 0;
    }
}

void MarginalInfo::preMarginalize() {
    for (ResidualBlockInfo &it: factors_) {
        it.Evaluate();

        std::vector<int> block_sizes = it.cost_function_->parameter_block_sizes();
        for (int i = 0; i < static_cast<int>(block_sizes.size()); i++) {
            double *addr = it.parameter_blocks_[i];
            int size = block_sizes[i];
            if (parameter_block_data_.find(addr) == parameter_block_data_.end()) {
                auto *data = new double[size];
                memcpy(data, it.parameter_blocks_[i], sizeof(double) * size);
                parameter_block_data_[addr] = data;
            }
        }
    }
}

struct ThreadsStruct {
    std::vector<ResidualBlockInfo> sub_factors;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    std::unordered_map<double*, int> parameter_block_size; //global size
    std::unordered_map<double*, int> parameter_block_idx; //local size
};

void ThreadsConstructA(ThreadsStruct *p) {
    for (const ResidualBlockInfo& it: p->sub_factors) {
        for (int i = 0; i < static_cast<int>(it.parameter_blocks_.size()); i++) {
            int idx_i = p->parameter_block_idx[it.parameter_blocks_[i]];
            int size_i = localSize(p->parameter_block_size[it.parameter_blocks_[i]]);
            Eigen::MatrixXd jacobian_i = it.jacobians_[i].leftCols(size_i);
            for (int j = i; j < static_cast<int>(it.parameter_blocks_.size()); j++) {
                int idx_j = p->parameter_block_idx[it.parameter_blocks_[j]];
                int size_j = localSize(p->parameter_block_size[it.parameter_blocks_[j]]);
                Eigen::MatrixXd jacobian_j = it.jacobians_[j].leftCols(size_j);
                p->A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                if(i != j) {
                    p->A.block(idx_j, idx_i, size_j, size_i) =
                            p->A.block(idx_i, idx_j, size_i, size_j).transpose();
                }
            }
            p->b.segment(idx_i, size_i) += jacobian_i.transpose() * it.residuals_;
        }
    }
}

void MarginalInfo::marginalize() {
    int total_dim = 0;
    // addResidualBlockInfo中已经将要丢掉的状态对应的parameter_block_idx_设置成0
    // 这时候遍历parameter_block_idx_得到的都是要丢掉的状态
    for (auto &it: parameter_block_idx_) {
        it.second = total_dim;
        total_dim += localSize(parameter_block_size_[it.first]);
    }
    discard_param_dim_ = total_dim;
    for (const std::pair<double * const, int> &it: parameter_block_size_) {
        if (parameter_block_idx_.find(it.first) == parameter_block_idx_.end()) {
            parameter_block_idx_[it.first] = total_dim;
            total_dim += localSize(it.second);
        }
    }
    reserve_param_dim_ = total_dim - discard_param_dim_;

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
        thread_struct.parameter_block_size = parameter_block_size_;
        thread_struct.parameter_block_idx = parameter_block_idx_;
        threads.emplace_back(&ThreadsConstructA, &thread_struct);
    }
    for (int i = NUM_THREADS - 1; i >= 0; i--) {
        threads[i].join();
        A += thread_structs[i].A;
        b += thread_structs[i].b;
    }

    const Eigen::VectorXd bmm = b.segment(0, discard_param_dim_);
    const Eigen::VectorXd brr = b.segment(discard_param_dim_, reserve_param_dim_);
    const Eigen::MatrixXd Amm = A.block(0, 0, discard_param_dim_, discard_param_dim_);
    const Eigen::MatrixXd Amr = A.block(0, discard_param_dim_, discard_param_dim_, reserve_param_dim_);
    const Eigen::MatrixXd Arm = A.block(discard_param_dim_, 0, reserve_param_dim_, discard_param_dim_);
    const Eigen::MatrixXd Arr = A.block(discard_param_dim_, discard_param_dim_, reserve_param_dim_, reserve_param_dim_);

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

    linearized_jacobians_ = S_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
    linearized_residuals_ = S_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose() * b;

    reserve_block_sizes_.clear();
    reserve_block_ids_.clear();
    reserve_block_datas_frozen_.clear();

    for (const auto& it: parameter_block_idx_) {
        if (it.second < discard_param_dim_) {
            // 丢弃的参数it.second都应该是0
            continue;
        }
        reserve_block_sizes_.emplace_back(parameter_block_size_[it.first]);
        reserve_block_ids_.emplace_back(parameter_block_idx_[it.first]);
        reserve_block_datas_frozen_.emplace_back(parameter_block_data_[it.first]);
    }
}

std::vector<double *> MarginalInfo::getParameterBlocks(const DoublePtr2DoublePtr &addr_shift) const {
    std::vector<double *> reserve_block_addrs;
    for (const auto& it: parameter_block_idx_) {
        if (it.second < discard_param_dim_) {
            // 丢弃的参数it.second都应该是0
            continue;
        }
        reserve_block_addrs.emplace_back(addr_shift.at(it.first));
    }
    return reserve_block_addrs;
}

MarginalFactor::MarginalFactor(std::shared_ptr<MarginalInfo>  _marginal_info)
 : marginal_info_(std::move(_marginal_info)) {
    for (auto it: marginal_info_->reserve_block_sizes_) {
        mutable_parameter_block_sizes()->push_back(it);
    }
    set_num_residuals(marginal_info_->reserve_param_dim_);
}

bool MarginalFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
    int reserve_param_dim = marginal_info_->reserve_param_dim_;
    int discard_param_dim = marginal_info_->discard_param_dim_;
    Eigen::VectorXd dx(reserve_param_dim);
    for (int i = 0; i < static_cast<int>(marginal_info_->reserve_block_sizes_.size()); i++) {
        int size = marginal_info_->reserve_block_sizes_[i];
        int idx = marginal_info_->reserve_block_ids_[i] - discard_param_dim;
        Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(parameters[i], size);
        Eigen::VectorXd x0 = Eigen::Map<const Eigen::VectorXd>(marginal_info_->reserve_block_datas_frozen_[i], size);
        if (size != 7)
            dx.segment(idx, size) = x - x0;
        else {
            Eigen::Quaterniond q1 = Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse();
            Eigen::Quaterniond q2 = Eigen::Quaterniond(x(6), x(3), x(4), x(5));
            dx.segment<3>(idx + 0) = x.head<3>() - x0.head<3>();
            dx.segment<3>(idx + 3) = 2.0 * utils::positify(q1 * q2).vec();
            if ((q1 * q2).w() < 0) {
                dx.segment<3>(idx + 3) = 2.0 * -utils::positify(q1 * q2).vec();
            }
        }
    }
    Eigen::Map<Eigen::VectorXd>(residuals, reserve_param_dim) =
            marginal_info_->linearized_residuals_ + marginal_info_->linearized_jacobians_ * dx;
    assert(jacobians);
    for (int i = 0; i < static_cast<int>(marginal_info_->reserve_block_sizes_.size()); i++) {
        assert(jacobians[i]);
        int size = marginal_info_->reserve_block_sizes_[i];
        int local_size = localSize(size);
        int idx = marginal_info_->reserve_block_ids_[i] - discard_param_dim;
        Eigen::Map<JacobianType> jacobian(jacobians[i], reserve_param_dim, size);
        jacobian.setZero();
        jacobian.leftCols(local_size) =
                marginal_info_->linearized_jacobians_.middleCols(idx, local_size);
    }
    return true;
}
