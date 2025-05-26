#include <iostream>
#include <Eigen/Core>

int main() {
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(4, 4);
    A << 0.01, 0.02, 0.03, 0.04,
         0.01, 0.02, 0.03, 0.04,
         0.01, 0.02, 0.03, 0.04,
         0.01, 0.02, 0.03, 0.04;
    
    Eigen::Array<bool, 1, Eigen::Dynamic> mask(4);
    mask.setConstant(false);
    mask(0) = mask(2) = true;

    Eigen::RowVectorXd dmu_dx = Eigen::RowVectorXd::Zero(4);
    dmu_dx << 0.05, 0.06, 0.07, 0.08;
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> B = (A.array() > 0.02);
    
    Eigen::MatrixXd C = B.select(A.array(), 0);
    
    // A.block(0, 0, 1, 4) = mask.select(dmu_dx.array(), 0);
    A.block(0, 0, 1, 4) = (mask).select(dmu_dx.array(), 0);

    Eigen::RowVectorXi indices = Eigen::RowVectorXi::Zero(2);
    indices << 1, 3;
    auto dm_ind = dmu_dx(indices);
    std::cout << "mask:\n" << mask << '\n';
    std::cout << "indices:\n" << indices << '\n';
    std::cout << "dmu_dx indices:\n" << dm_ind << '\n';
    std::cout << "Matrix A:\n" << A << '\n';
    std::cout << "Boolean mask B (true where A > 0.02):\n" << B.cast<int>() << '\n';
    std::cout << "Matrix C (A values where > 0.02, 0 otherwise):\n" << C << '\n';
    
    return 0;
}