#pragma once

#include "model_interface.hpp"

namespace GPMPC
{

namespace Model
{

class CartesianAugmentedModel: public ModelInterface
{
    public:
        CartesianAugmentedModel(int input_window): ModelInterface(5 + 4 * (input_window - 1), 2) {}
        void set_dynamic_matrices(Eigen::MatrixXd &A, Eigen::MatrixXd &B, Eigen::MatrixXd &C, double dt) override;
};

} // end namespace Model

} // end namespace GPMPC