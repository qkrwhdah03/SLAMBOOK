#define _USE_MATH_DEFINES // For using M_PI
#pragma warning(disable: 4819) // Disable waring 4819

#include <iostream>
#include <cmath>

using namespace std;

#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace Eigen;


int main(int argc, char **argv) {

  Matrix3d rotation_matrix = Matrix3d::Identity();
  AngleAxisd rotation_vector(M_PI/4, Vector3d(0,0,1)); // Rotate 90 degrees along with z-axis


  // 1. Rotation Matrix
  cout.precision(3); // sets or returns the current number of digits that is displayed for floating-point variables
  cout << "Rotation Matrix =\n" << rotation_vector.matrix() << endl;  // Transform rotation vector to roation matrix 

  rotation_matrix = rotation_vector.toRotationMatrix();

  // 2. Roation using Roation Vector
  Vector3d v(1,0,0);
  Vector3d v_rotated = rotation_vector * v;
  cout << "(1,0,0) after rotation (by angle axis) = " << v_rotated.transpose() << endl;

  // 3. Rotation using Rotation Matrix
  v_rotated = rotation_matrix * v; 
  cout << "(1,0,0) after rotation (by rotation matrix) = " << v_rotated.transpose() << endl;

  // 4. Euler Angles
  Vector3d euler_angles = rotation_matrix.eulerAngles(2, 1, 0); // ZYX = Roll Pitch Yaw
  cout << "Yaw Pitch Roll = " << euler_angles.transpose() << endl;

  // 5. Tansform Matrix
  Isometry3d T = Isometry3d::Identity(); // It is called 3D, but it is actually 4x4 matrix
  T.rotate(rotation_vector); // Rotate using roation vector
  T.pretranslate(Vector3d(1,3,4)); // Set translation vector with (1, 3, 4)
  cout << "Transform matrix = \n" << T.matrix() << endl; 

  // For Affine Transformation -> Eigen::Affine3d
  // For Projection Transformation -> Eigen::Projective3d

  // 6. Apply Transform Matrix
  Vector3d v_transformed = T * v; // Same with R*v+t
  cout << "v transformed = " << v_transformed.transpose() << endl;

  // 7. Quaternion
  Quaterniond q = Quaterniond(rotation_vector); // Using angle axis
  cout << "Quaternion from rotation vector = " << q.coeffs().transpose() << endl; // coef : (x, y, z, w) 

  q = Quaterniond(rotation_matrix);
  cout << "Quaternion from rotation matrix = " << q.coeffs().transpose() << endl; 

  // 8. Roation using Quaternion 
  v_rotated = q * v; // same with qvq^{-1} in math

  cout << "(1,0,0) after rotation = " << v_rotated.transpose() << endl;
  cout << "Should be equal to " << (q * Quaterniond(0, 1, 0, 0) * q.inverse()).coeffs().transpose() << endl;

  return 0;
}
