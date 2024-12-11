#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "sophus/se3.hpp"

using namespace std;

int main(int argc, char **argv) {
  // Rotate 90 degree along with z-axis
  Eigen::Matrix3d R = Eigen::AngleAxis(M_PI / 2, Eigen::Vector3d(0,0,1)).toRotationMatrix();
  Eigen::Quaterniond q(R);

  // Sophus::SO3d
  Sophus::SO3d SO3_R(R);
  Sophus::SO3d SO3_q(q);

  cout << "SO(3) from Matrix \n" << SO3_R.matrix() << endl;
  cout << "SO(3) from Quaternion \n" << SO3_q.matrix() << endl;
  cout << "They are equal" << endl;

  // Log mapping
  Eigen::Vector3d so3 = SO3_R.log();
  cout << "so3 = " << so3.transpose() << endl;

  cout << "so3 hat = \n" << Sophus::SO3d::hat(so3) << endl;
  cout << "so3 hat vee = " << Sophus::SO3d::vee(Sophus::SO3d::hat(so3)).transpose() << endl;

  // Perterbation Model
  Eigen::Vector3d update_so3(1e-4, 0, 0); // small perterbation
  Sophus::SO3d SO3_updated = Sophus::SO3d::exp(update_so3) * SO3_R;
  cout << "SO3 updated =\n" << SO3_updated.matrix() << endl;

  cout << "*******************************" << endl;

  Eigen::Vector3d t(1,0,0);
  Sophus::SE3d SE3_Rt(R,t);
  Sophus::SE3d SE3_qt(q,t);

  cout << "SE3 from R, t= \n" << SE3_Rt.matrix() << endl;
  cout << "SE3 from q, t= \n" << SE3_qt.matrix() << endl;

  // se(3) lie algebra is 6 dimension
  typedef Eigen::Matrix<double, 6, 1> Vector6d;
  Vector6d se3 = SE3_Rt.log();
  cout << "se3 = " << se3.transpose() << endl;
  // Sophus에서 변환이 앞에 있고, 회전이 뒤에 있음을 볼 수 있다.

  cout << "se3 hat = \n" << Sophus::SE3d::hat(se3) << endl;
  cout << "se3 hat vee = " << Sophus::SE3d::vee(Sophus::SE3d::hat(se3)).transpose() << endl;

  // Perterbation Model
  Vector6d update_se3; 
  update_se3.setZero();
  update_se3(0,0) = 1e-4;
  Sophus::SE3d SE3_updated = Sophus::SE3d::exp(update_se3) * SE3_Rt;
  cout << "SE3 updated = " << endl << SE3_updated.matrix() << endl;

  return 0;
}
