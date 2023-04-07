#include <thread>
#include "marginalization_factor.h"
#include "log.h"

void ResidualBlockInfo::Evaluate() {
    residuals.resize(cost_function->num_residuals());

    std::vector<int> block_sizes = cost_function->parameter_block_sizes();
    raw_jacobians = new double *[block_sizes.size()];
    jacobians.resize(block_sizes.size());

    for (int i = 0; i < static_cast<int>(block_sizes.size()); i++) {
        jacobians[i].resize(cost_function->num_residuals(), block_sizes[i]);
        raw_jacobians[i] = jacobians[i].data();
    }
    cost_function->Evaluate(parameter_blocks.data(), residuals.data(), raw_jacobians);

    if (loss_function) {
        double residual_scaling_, alpha_sq_norm_;

        double sq_norm, rho[3];

        sq_norm = residuals.squaredNorm();
        loss_function->Evaluate(sq_norm, rho);

        double sqrt_rho1_ = sqrt(rho[1]);

        if ((sq_norm == 0.0) || (rho[2] <= 0.0)) {
            residual_scaling_ = sqrt_rho1_;
            alpha_sq_norm_ = 0.0;
        } else {
            const double D = 1.0 + 2.0 * sq_norm * rho[2] / rho[1];
            const double alpha = 1.0 - sqrt(D);
            residual_scaling_ = sqrt_rho1_ / (1 - alpha);
            alpha_sq_norm_ = alpha / sq_norm;
        }

        for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++) {
            jacobians[i] =
                    sqrt_rho1_ * (jacobians[i] - alpha_sq_norm_ * residuals * (residuals.transpose() * jacobians[i]));
        }

        residuals *= residual_scaling_;
    }
}

MarginalizationInfo::~MarginalizationInfo() {

    for (auto & it : parameter_block_data_)
        delete[] it.second;

    for (auto & factor : factors_) {
        delete[] factor.raw_jacobians;
        delete factor.cost_function;
    }
}

void MarginalizationInfo::addResidualBlockInfo(const ResidualBlockInfo &residual_block_info) {
    factors_.emplace_back(residual_block_info);

    const std::vector<double *> &parameter_blocks = residual_block_info.parameter_blocks;
    std::vector<int> parameter_block_sizes = residual_block_info.cost_function->parameter_block_sizes();

    for (int i = 0; i < static_cast<int>(residual_block_info.parameter_blocks.size()); i++) {
        double *addr = parameter_blocks[i];
        int size = parameter_block_sizes[i];
        parameter_block_size_[reinterpret_cast<long>(addr)] = size;
    }

    for (int i : residual_block_info.drop_set) {
        double *addr = parameter_blocks[i];
        parameter_block_idx_[reinterpret_cast<long>(addr)] = 0;
    }
}

void MarginalizationInfo::preMarginalize() {
    for (ResidualBlockInfo &it: factors_) {
        it.Evaluate();

        std::vector<int> block_sizes = it.cost_function->parameter_block_sizes();
        for (int i = 0; i < static_cast<int>(block_sizes.size()); i++) {
            long addr = reinterpret_cast<long>(it.parameter_blocks[i]);
            int size = block_sizes[i];
            if (parameter_block_data_.find(addr) == parameter_block_data_.end()) {
                auto *data = new double[size];
                memcpy(data, it.parameter_blocks[i], sizeof(double) * size);
                parameter_block_data_[addr] = data;
            }
        }
    }
}

int MarginalizationInfo::localSize(int size) {
    return size == 7 ? 6 : size;
}

void *ThreadsConstructA(ThreadsStruct *p) {
    for (const ResidualBlockInfo& it: p->sub_factors) {
        for (int i = 0; i < static_cast<int>(it.parameter_blocks.size()); i++) {
            int idx_i = p->parameter_block_idx[reinterpret_cast<long>(it.parameter_blocks[i])];
            int size_i = p->parameter_block_size[reinterpret_cast<long>(it.parameter_blocks[i])];
            if (size_i == 7)
                size_i = 6;
            Eigen::MatrixXd jacobian_i = it.jacobians[i].leftCols(size_i);
            for (int j = i; j < static_cast<int>(it.parameter_blocks.size()); j++) {
                int idx_j = p->parameter_block_idx[reinterpret_cast<long>(it.parameter_blocks[j])];
                int size_j = p->parameter_block_size[reinterpret_cast<long>(it.parameter_blocks[j])];
                if (size_j == 7)
                    size_j = 6;
                Eigen::MatrixXd jacobian_j = it.jacobians[j].leftCols(size_j);
                p->A.block(idx_i, idx_j, size_i, size_j) += jacobian_i.transpose() * jacobian_j;
                if(i != j) {
                    p->A.block(idx_j, idx_i, size_j, size_i) =
                            p->A.block(idx_i, idx_j, size_i, size_j).transpose();
                }
            }
            p->b.segment(idx_i, size_i) += jacobian_i.transpose() * it.residuals;
        }
    }
    return p;
}

void MarginalizationInfo::marginalize() {
    int pos = 0;
    for (auto &it: parameter_block_idx_) {
        it.second = pos;
        pos += localSize(parameter_block_size_[it.first]);
    }

    m = pos;

    for (const auto &it: parameter_block_size_) {
        if (parameter_block_idx_.find(it.first) == parameter_block_idx_.end()) {
            parameter_block_idx_[it.first] = pos;
            pos += localSize(it.second);
        }
    }

    n = pos - m;

    TicToc t_summing;
    Eigen::MatrixXd A(pos, pos);
    Eigen::VectorXd b(pos);
    A.setZero();
    b.setZero();

    TicToc t_thread_summing;
    std::vector<std::thread> threads;
    ThreadsStruct thread_structs[NUM_THREADS];
    for (int j = 0; j < factors_.size(); ++j) {
        thread_structs[j % NUM_THREADS].sub_factors.emplace_back(factors_[j]);
    }
    for (auto & thread_struct : thread_structs) {
        TicToc zero_matrix;
        thread_struct.A = Eigen::MatrixXd::Zero(pos, pos);
        thread_struct.b = Eigen::VectorXd::Zero(pos);
        thread_struct.parameter_block_size = parameter_block_size_;
        thread_struct.parameter_block_idx = parameter_block_idx_;
        threads.emplace_back(ThreadsConstructA, &thread_struct);
    }
    for (int i = NUM_THREADS - 1; i >= 0; i--) {
        threads[i].join();
        A += thread_structs[i].A;
        b += thread_structs[i].b;
    }


    const Eigen::MatrixXd &Amm = A.block(0, 0, m, m);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(0.5 * (Amm + Amm.transpose()));

    Eigen::MatrixXd Amm_inv = saes.eigenvectors() * Eigen::VectorXd(
            (saes.eigenvalues().array() > EPS).select(saes.eigenvalues().array().inverse(), 0)).asDiagonal() *
                              saes.eigenvectors().transpose();

    Eigen::VectorXd bmm = b.segment(0, m);
    Eigen::MatrixXd Amr = A.block(0, m, m, n);
    Eigen::MatrixXd Arm = A.block(m, 0, n, m);
    Eigen::MatrixXd Arr = A.block(m, m, n, n);
    Eigen::VectorXd brr = b.segment(m, n);
    A = Arr - Arm * Amm_inv * Amr;
    b = brr - Arm * Amm_inv * bmm;

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(A);
    Eigen::VectorXd S = Eigen::VectorXd((saes2.eigenvalues().array() > EPS).select(saes2.eigenvalues().array(), 0));
    Eigen::VectorXd S_inv = Eigen::VectorXd(
            (saes2.eigenvalues().array() > EPS).select(saes2.eigenvalues().array().inverse(), 0));

    Eigen::VectorXd S_sqrt = S.cwiseSqrt();
    Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();

    linearized_jacobians_ = S_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
    linearized_residuals_ = S_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose() * b;
}

std::vector<double *> MarginalizationInfo::getParameterBlocks(std::unordered_map<long, double *> &addr_shift) {
    std::vector<double *> keep_block_addr;
    keep_block_size_.clear();
    keep_block_idx_.clear();
    keep_block_data_.clear();

    for (const auto &it: parameter_block_idx_) {
        if (it.second >= m) {
            keep_block_size_.push_back(parameter_block_size_[it.first]);
            keep_block_idx_.push_back(parameter_block_idx_[it.first]);
            keep_block_data_.push_back(parameter_block_data_[it.first]);
            keep_block_addr.push_back(addr_shift[it.first]);
        }
    }

    return keep_block_addr;
}

MarginalizationFactor::MarginalizationFactor(MarginalizationInfo *_marginalization_info) : marginalization_info_(
        _marginalization_info) {
    for (auto it: marginalization_info_->keep_block_size_) {
        mutable_parameter_block_sizes()->push_back(it);
    }
    set_num_residuals(marginalization_info_->n);
};

bool MarginalizationFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const {
    int n = marginalization_info_->n;
    int m = marginalization_info_->m;
    Eigen::VectorXd dx(n);
    for (int i = 0; i < static_cast<int>(marginalization_info_->keep_block_size_.size()); i++) {
        int size = marginalization_info_->keep_block_size_[i];
        int idx = marginalization_info_->keep_block_idx_[i] - m;
        Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(parameters[i], size);
        Eigen::VectorXd x0 = Eigen::Map<const Eigen::VectorXd>(marginalization_info_->keep_block_data_[i], size);
        if (size != 7)
            dx.segment(idx, size) = x - x0;
        else {
            Eigen::Quaterniond q1 = Eigen::Quaterniond(x0(6), x0(3), x0(4), x0(5)).inverse();
            Eigen::Quaterniond q2 = Eigen::Quaterniond(x(6), x(3), x(4), x(5));
            dx.segment<3>(idx + 0) = x.head<3>() - x0.head<3>();
            dx.segment<3>(idx + 3) = 2.0 * Utility::positify(q1 * q2).vec();
            if ((q1 * q2).w() < 0) {
                dx.segment<3>(idx + 3) = 2.0 * -Utility::positify(q1 * q2).vec();
            }
        }
    }
    Eigen::Map<Eigen::VectorXd>(residuals, n) =
            marginalization_info_->linearized_residuals_ + marginalization_info_->linearized_jacobians_ * dx;
    if (jacobians) {
        for (int i = 0; i < static_cast<int>(marginalization_info_->keep_block_size_.size()); i++) {
            if (jacobians[i]) {
                int size = marginalization_info_->keep_block_size_[i], local_size = marginalization_info_->localSize(size);
                int idx = marginalization_info_->keep_block_idx_[i] - m;
                typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> JacobianType;
                Eigen::Map<JacobianType> jacobian(jacobians[i], n, size);
                jacobian.setZero();
                jacobian.leftCols(local_size) = marginalization_info_->linearized_jacobians_.middleCols(idx, local_size);
            }
        }
    }
    return true;
}
