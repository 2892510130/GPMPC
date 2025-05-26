#pragma once

#include "model_interface.hpp"

namespace GPMPC
{

namespace Model
{

class KinematicCartesianModel: public ModelInterface
{
    public:
        KinematicCartesianModel(): ModelInterface(5, 2) {}
        void set_dynamic_matrices(Eigen::MatrixXd &A, Eigen::MatrixXd &B, Eigen::MatrixXd &C, double dt) override;
};

} // end namespace Model

} // end namespace GPMPC