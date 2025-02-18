#include <iostream>
#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>
#include <chrono>

using namespace std;

// Cost Function Model
struct CURVE_FITTING_COST {
  CURVE_FITTING_COST(double x, double y) : _x(x), _y(y) {}

  // Calculate Residual
  template<typename T>
  bool operator()(
    const T *const abc, // Model Parameter a, b, c
    T *residual) const {
    residual[0] = T(_y) - ceres::exp(abc[0] * T(_x) * T(_x) + abc[1] * T(_x) + abc[2]); // y-exp(ax^2+bx+c)
    return true;
  }

  const double _x, _y;    // x, y data
};

int main(int argc, char **argv) {
  double ar = 1.0, br = 2.0, cr = 1.0; // Real Value
  double ae = 2.0, be = -1.0, ce = 5.0; // Estimated Value
  int N = 100;  // Number of Data
  double w_sigma = 1.0; // Standard deviation of noise
  double inv_sigma = 1.0 / w_sigma;
  cv::RNG rng; // For Gaussian generation

  vector<double> x_data, y_data;   
  for (int i = 0; i < N; i++) {
    double x = i / 100.0;
    x_data.push_back(x);
    y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
  }

  double abc[3] = {ae, be, ce};

  // Least Square Problem
  ceres::Problem problem;
  for (int i = 0; i < N; i++) {
    problem.AddResidualBlock(     // Add error term
      // Auto-Diff, Residual Type, output dim, input dim 
      new ceres::AutoDiffCostFunction<CURVE_FITTING_COST, 1, 3>(
        new CURVE_FITTING_COST(x_data[i], y_data[i])
      ),
      nullptr,  // Kernel-Function
      abc      // Parameters to be estimated
    );
  }

  // Solver
  ceres::Solver::Options options;     
  options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;  // Way to solve linear system 
  options.minimizer_progress_to_stdout = true;   // cout print

  ceres::Solver::Summary summary;                // Sumamry
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  ceres::Solve(options, &problem, &summary);  // Start optimization 
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "solve time cost = " << time_used.count() << " seconds. " << endl;

  // Print Result
  cout << summary.BriefReport() << endl;
  cout << "estimated a,b,c = ";
  for (auto a:abc) cout << a << " ";
  cout << endl;

  return 0;
}