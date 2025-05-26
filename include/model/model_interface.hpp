#pragma once

#include <Eigen/Core>

namespace GPMPC
{

namespace Model
{

// How the discretization is done
enum DiscretizeMethod
{
    FirstOrder,
    Bylinear,
    RK4
};

// How the linearlization is done
enum LinearlizationMethod
{
    Reference,
    Guess
};

// The base class for all models
class ModelInterface
{
    public:
        ModelInterface(int state_d, int input_d) : m_state_dimension(state_d), m_input_dimension(input_d) {}
        virtual ~ModelInterface() = default;

        // function that set points where linearlization is taken
        void set_reference(const Eigen::MatrixXd &x_r, const Eigen::MatrixXd &u_r);
        void set_guess(const Eigen::MatrixXd &x_g, const Eigen::MatrixXd &u_g);
        void set_linearlization_points(Eigen::MatrixXd &x_lin, Eigen::MatrixXd &u_lin);

        void init_matrices(Eigen::MatrixXd &A, Eigen::MatrixXd &B, Eigen::MatrixXd &C);

        // set model matrices, x_{k+1} = A_k @ x_k + B_k @ u_k + C_k
        virtual void set_dynamic_matrices(Eigen::MatrixXd &A, Eigen::MatrixXd &B, Eigen::MatrixXd &C, double dt);

    public:
        const int m_state_dimension;
        const int m_input_dimension;

    protected:
        LinearlizationMethod m_linear_method = LinearlizationMethod::Reference;
        DiscretizeMethod m_discrete_method = DiscretizeMethod::FirstOrder;

        Eigen::MatrixXd m_state_reference;
        Eigen::MatrixXd m_input_reference;
        Eigen::MatrixXd m_state_guess;
        Eigen::MatrixXd m_input_guess;
};

} // end namespace Model

} // end namespace GPMPC