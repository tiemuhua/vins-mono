#pragma once

#include <cstdlib>
#include <pthread.h>
#include <ceres/ceres.h>
#include <unordered_map>

#include "../utility/utility.h"
#include "../utility/tic_toc.h"

const int NUM_THREADS = 4;

struct ResidualBlockInfo
{
    ResidualBlockInfo(ceres::CostFunction *_cost_function, ceres::LossFunction *_loss_function, std::vector<double *> _parameter_blocks, std::vector<int> _drop_set)
        : cost_function(_cost_function), loss_function(_loss_function), parameter_blocks(_parameter_blocks), drop_set(_drop_set) {}

    void Evaluate();

    ceres::CostFunction *cost_function;
    ceres::LossFunction *loss_function;
    std::vector<double *> parameter_blocks;
    std::vector<int> drop_set;

    double **raw_jacobians{};
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;
    Eigen::VectorXd residuals;

    int localSize(int size)
    {
        return size == 7 ? 6 : size;
    }
};

struct ThreadsStruct
{
    std::vector<ResidualBlockInfo> sub_factors;
    Eigen::MatrixXd A;
    Eigen::VectorXd b;
    std::unordered_map<long, int> parameter_block_size; //global size
    std::unordered_map<long, int> parameter_block_idx; //local size
};

class MarginalizationInfo
{
  public:
    ~MarginalizationInfo();
    static int localSize(int size) ;
    void addResidualBlockInfo(const ResidualBlockInfo &residual_block_info);
    void preMarginalize();
    void marginalize();
    std::vector<double *> getParameterBlocks(std::unordered_map<long, double *> &addr_shift);

    std::vector<ResidualBlockInfo> factors_;
    int m, n;
    std::unordered_map<long, int> parameter_block_size_; //global size
    std::unordered_map<long, int> parameter_block_idx_; //local size
    std::unordered_map<long, double *> parameter_block_data_;

    std::vector<int> keep_block_size_; //global size
    std::vector<int> keep_block_idx_;  //local size
    std::vector<double *> keep_block_data_;

    Eigen::MatrixXd linearized_jacobians_;
    Eigen::VectorXd linearized_residuals_;
    static constexpr double EPS = 1e-8;
};

class MarginalizationFactor : public ceres::CostFunction
{
  public:
    explicit MarginalizationFactor(MarginalizationInfo* _marginalization_info);
    bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const override;

    MarginalizationInfo* marginalization_info_;
};
