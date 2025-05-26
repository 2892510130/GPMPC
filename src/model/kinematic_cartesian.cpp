#include "model/kinematic_cartesian.hpp"

namespace GPMPC
{

namespace Model
{

    void KinematicCartesianModel::set_dynamic_matrices(Eigen::MatrixXd &A, Eigen::MatrixXd &B, Eigen::MatrixXd &C, double dt)
    {
        init_matrices(A, B, C);

        Eigen::MatrixXd x_lin, u_lin;
        set_linearlization_points(x_lin, u_lin);

        A(0, 2) = -1.0 * x_lin(3) * sin(x_lin(2));
        A(0, 3) = cos(x_lin(2));
        A(1, 2) = x_lin(3) * cos(x_lin(2));
        A(1, 3) = sin(x_lin(2));
        A(2, 4) = 1.0;

        B(3, 0) = 1.0;
        B(4, 1) = 1.0;

        C(0, 0) = -1.0 * A(0, 2) * x_lin(2);
        C(1, 0) = -1.0 * A(1, 2) * x_lin(2);

    }

} // end namespace Model

} // end namespace GPMPC