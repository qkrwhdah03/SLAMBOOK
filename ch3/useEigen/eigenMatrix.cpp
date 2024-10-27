#include <iostream>

using namespace std;

#include <ctime>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace Eigen;

#define MATRIX_SIZE 50


int main(int argc, char **argv) {

  // Basic of using Eigen Library
  // You may see the official documents of the following link
  // https://eigen.tuxfamily.org

  // 1. Eigen::Matrix - Representing vectors and matrixes
  Matrix<float, 2, 3> matrix_23;

  // 2. Eigen Have a lot of typedef
  Vector3d v_3d;
  Matrix<float, 3, 1> vd_3d;

  // 3. Matrix3d = Eigen::Matrix<double, 3, 3>
  Matrix3d matrix_33 = Matrix3d::Zero();

  // Dynamic size Matrix3d
  Matrix<double, Dynamic, Dynamic> matrix_dynamic; 

  MatrixXd matrix_x;

  // 4. Eigen Matrix
  // Initialize
  matrix_23 << 1, 2, 3, 4, 5, 6; 

  cout << "matrix 2x3 from 1 to 6 : \n" << matrix_23 << '\n';

  cout << "print matrix 2x3: " << '\n';
  for(int i=0; i<2; ++i){
      for(int j=0; j<3; ++j){
          cout << matrix_23(i,j) << ' ';
      }
      cout << '\n';
  }

  // 5. Egien needs to match the type of matrix
  // #x) Matrix<double, 2, 1> result_wrong_type = matrix_23 * v+3d;  We Need casting!  
  v_3d << 3, 2, 1;
  vd_3d << 4, 5, 6;

  Matrix<double, 2, 1> result = matrix_23.cast<double>() * v_3d;

  cout << "[1,2,3,4,5,6]*[3,2,1] = " << result.transpose() << '\n';

   Matrix<float, 2, 1> result2 = matrix_23 * vd_3d;

  cout << "[1,2,3,4,5,6]*[4,5,6] = " << result2.transpose() << '\n';

  // 5. Matrix Operation
  matrix_33 = Matrix3d::Random();
  cout << "Random Matrix :  \n" << matrix_33 << '\n';
  cout << "Transpose : \n" << matrix_33.transpose() << '\n'; 
  cout << "Sum : " << matrix_33.sum() << '\n';
  cout << "Trace : " << matrix_33.trace() << '\n';
  cout << "Times : \n" << 10 * matrix_33 << '\n';
  cout << "Inverse : \n" << matrix_33.inverse() << '\n';
  cout << "Determinant : " << matrix_33.determinant() << '\n';
 

  // 6. Eigen Values
  // Computes eigenvalues and eigenvectors of selfadjoint matrices.
  SelfAdjointEigenSolver<Matrix3d> eigen_solver(matrix_33.transpose() * matrix_33);
  cout << "Eigen Values : \n" << eigen_solver.eigenvalues() << '\n';
  cout << "Eigen Vectors : \n" << eigen_solver.eigenvectors() << '\n';

  // 7. Solving Linear Equation Ax = b
  Matrix<double, MATRIX_SIZE, MATRIX_SIZE> matrix_NN = MatrixXd::Random(MATRIX_SIZE, MATRIX_SIZE);
  matrix_NN = matrix_NN * matrix_NN.transpose(); // Ensuring semi-positive definite

  Matrix<double, MATRIX_SIZE, 1> v_Nd = MatrixXd::Random(MATRIX_SIZE, 1);

  clock_t time_start = clock();  // Setting start clock  

  Matrix<double, MATRIX_SIZE, 1> x = matrix_NN.inverse() * v_Nd;
  cout << "Time of using Normal Inverse  is " << 1000 * (clock() - time_start) / (double) CLOCKS_PER_SEC << "ms\n";
  cout << "x = [" << x.transpose() << "]\n";

  // Using Qr Decomposition
  time_start = clock();
  x = matrix_NN.colPivHouseholderQr().solve(v_Nd);
  cout << "Time of using QR Decomposition is " << 1000 * (clock() - time_start) / (double) CLOCKS_PER_SEC << "ms\n";
  cout << "x = [" << x.transpose() << "]\n";

  // Cholesky Decomposition
  time_start = clock();
  x = matrix_NN.ldlt().solve(v_Nd);
  cout << "Time of using ldlt Decomposition is " << 1000 * (clock() - time_start) / (double) CLOCKS_PER_SEC << "ms\n";
  cout << "x = [" << x.transpose() << "]\n";
  return 0;
}