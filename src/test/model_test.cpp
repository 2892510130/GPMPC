#include <iostream>

#include "model/model_list.hpp"

int main()
{
    using namespace GPMPC::Model;
    ModelLists model_lists;
    // auto model = KinematicCartesianModel();
    auto model = model_lists.create_model("Kinematic Cartesian");
    // auto model = model_lists.create_model("Cartesian Augmented");
    Eigen::MatrixXd a, b, c, xr, ur;
    xr = Eigen::MatrixXd::Zero(model->m_state_dimension, 1);
    ur = Eigen::MatrixXd::Zero(model->m_input_dimension, 1);
    double dt = 0.05;
    model->set_reference(xr, ur);
    model->set_dynamic_matrices(a, b, c, dt);
    std::cout << "A:\n" << a << '\n';
    std::cout << "B:\n" << b << '\n';
    std::cout << "C:\n" << c << '\n';
}