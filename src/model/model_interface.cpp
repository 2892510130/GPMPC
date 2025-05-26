#include "model/model_interface.hpp"

namespace GPMPC
{

namespace Model
{

    void ModelInterface::set_reference(const Eigen::MatrixXd &x_r, const Eigen::MatrixXd &u_r)
    {
        m_state_reference = x_r;
        m_input_reference = u_r;
    }

    void ModelInterface::set_guess(const Eigen::MatrixXd &x_g, const Eigen::MatrixXd &u_g)
    {
        m_state_guess = x_g;
        m_input_guess = u_g;
    }

    void ModelInterface::set_linearlization_points(Eigen::MatrixXd &x_lin, Eigen::MatrixXd &u_lin)
    {
        switch (m_linear_method)
        {
            case LinearlizationMethod::Reference:
            {
                x_lin = m_state_reference;
                u_lin = m_input_reference;
                break;
            }
            
            case LinearlizationMethod::Guess:
            {
                x_lin = m_state_guess;
                u_lin = m_input_guess;
                break;
            }

            default:
                break;
        }
    }

    void ModelInterface::init_matrices(Eigen::MatrixXd &A, Eigen::MatrixXd &B, Eigen::MatrixXd &C)
    {
        A = Eigen::MatrixXd::Zero(m_state_dimension, m_state_dimension);
        B = Eigen::MatrixXd::Zero(m_state_dimension, m_input_dimension);
        C = Eigen::MatrixXd::Zero(m_state_dimension, 1);
    }

} // end namespace Model

} // end namespace GPMPC