#include <iostream>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <ceres/ceres.h>
#include <vector>
#include <string> 
#include <cmath>

using namespace std;

Eigen::Matrix<double, 3, 3> Estimate_Homography(vector<Eigen::Vector2d> &M, vector<Eigen::Vector2d> &m, bool optimize = true, int verbose = 1);
Eigen::Matrix<double, 3, 3> Estimate_Intrinsic (vector<Eigen::Matrix<double, 3, 3>> &homographies);
Eigen::Matrix<double, 3, 4> Estimate_Extrinsic (Eigen::Matrix<double, 3, 3> H, Eigen::Matrix<double, 3, 3> K);
Eigen::Vector2d Estimate_Distortion(Eigen::Matrix<double, 3, 3> K, vector<Eigen::Matrix<double, 3, 4>> &extrinsics, vector<Eigen::Vector2d> &M, vector<vector<Eigen::Vector2d>> &m);
void Evaluate(Eigen::Matrix<double, 3, 3> K, Eigen::Vector2d k, vector<Eigen::Matrix<double, 3, 4>> &extrinsics, vector<Eigen::Vector2d> &M, vector<vector<Eigen::Vector2d>> &m);
void MLE_Optimize(Eigen::Matrix<double, 3, 3> &K, Eigen::Vector2d &k, vector<Eigen::Matrix<double, 3, 4>> &extrinsics, vector<Eigen::Vector2d> &M, vector<vector<Eigen::Vector2d>> &m);

struct HomographyResidualFunction{
    HomographyResidualFunction(const Eigen::Vector2d p, const Eigen::Vector2d q): p_(p), q_(q){}

    template<typename T>
    bool operator()(const T* const parameters, T* residual) const{
        Eigen::Map<const Eigen::Matrix<T, 3, 3, Eigen::RowMajor>> H(parameters);
        Eigen::Matrix<T, 3, 1> p;
        p << T(p_(0)), T(p_(1)),T(1.0);

        Eigen::Matrix<T, 3, 1> q_h = H * p;
        q_h /= q_h(2);

        residual[0] = q_h(0) - T(q_(0));
        residual[1] = q_h(1) - T(q_(1));
        return true;
    }
    private:
        const Eigen::Vector2d p_, q_;
};

struct MLEResidualFunction{
    MLEResidualFunction(const Eigen::Vector2d p, const Eigen::Vector2d q): p_(p), q_(q){}

    template<typename T>
    bool operator()(const T* const intrinsic_vector,const T* const distortion_vector, const T* const extrinsic_vector, T* residual) const{

        Eigen::Matrix<T, 3, 3> K;
        K.row(0) << intrinsic_vector[0], intrinsic_vector[2], intrinsic_vector[3];
        K.row(1) << T(0.), intrinsic_vector[1], intrinsic_vector[4];
        K.row(2) << T(0.), T(0.), T(1.);

        T u0 = intrinsic_vector[3];
        T v0 = intrinsic_vector[4];

        T k1 = distortion_vector[0];
        T k2 = distortion_vector[1];

        Eigen::Matrix<T, 3, 1> rotation_vector, translation_vector;
        rotation_vector << extrinsic_vector[0], extrinsic_vector[1], extrinsic_vector[2];
        translation_vector << extrinsic_vector[3], extrinsic_vector[4], extrinsic_vector[5];

        // Rotation
        Eigen::Matrix<T, 3, 1> p_homo(T(p_(0)), T(p_(1)), T(0)); // Homogeneous coordinates
        Eigen::AngleAxis<T> angle_axis(rotation_vector.norm(), rotation_vector.normalized());
        Eigen::Matrix<T, 3, 3> rotation_matrix = angle_axis.toRotationMatrix();
        Eigen::Matrix<T, 3, 1> p_rotated = rotation_matrix * p_homo;

        // Apply translation
        Eigen::Matrix<T, 3, 1> p_translated = p_rotated + translation_vector;

        // Project the 3D point onto the image plane
        T x = p_translated(0) / p_translated(2);
        T y = p_translated(1) / p_translated(2);
        T dis = x*x + y*y;

        Eigen::Matrix<T, 3, 1> p_changed = K * p_translated;
        T u = p_changed(0) / p_changed(2);
        T v = p_changed(1) / p_changed(2);

        T q_u_hat = u + (u-u0)*(k1*dis + k2*dis*dis);
        T q_v_hat = v + (v-v0)*(k1*dis + k2*dis*dis);

        residual[0] = q_u_hat - T(q_(0));
        residual[1] = q_v_hat - T(q_(1));

        return true;
    }

    private:
        const Eigen::Vector2d p_, q_;
};


int main(int argc, char ** argv){

    // 0. Settings 
    string base_path = "C:/Users/qkrwh/OneDrive/Desktop/VClab/visual-slam-2024-fall-indi2-code/revision/ch5/calibration/image/";
    int num_images = 5; // Number of Checker Board Images
    int checker_board_rows = 7; // Number of rows to detect in the checker board
    int checker_board_cols = 10; // Number of columns to detect in the checker board
    double checker_board_length = 0.025; // Length of checker board square (m)

    // 1. Read Image & 2. Detect Feature Point
    vector<string> image_paths;
    for(int i=1; i<=num_images; ++i){
        string path = base_path + to_string(i) + ".jpg";
        image_paths.push_back(path);
    }

    vector<vector<cv::Point2f>> cv_feature_points(num_images);
    for(int i=0; i<num_images; ++i){
        cv::Mat image = cv::imread(image_paths[i]);
        if(image.empty()){
            cerr << "[Error] Fail to read image " << image_paths[i] << endl;
            return -1;
        }

        cv::Size patternSize(checker_board_cols, checker_board_rows);
        int flags =  cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE;
        bool found = cv::findChessboardCorners(image, patternSize, cv_feature_points[i], flags);

        if(found){
            cv::drawChessboardCorners(image, patternSize, cv_feature_points[i], found);
            cv::imshow("Checker Board with Corners", image);
            cv::waitKey(0); 
        }
        else{
            cerr << "[Error] Fail to Detect Checker Board with cv::findChessboardCorners" << endl;
            return 1;
        }
    }

    // Coordinate Data Setting
    vector<vector<Eigen::Vector2d>> eigen_feature_points(num_images);
    for(int i=0; i<num_images; ++i){
        for(int j=0; j<cv_feature_points[i].size(); ++j){
            cv::Point2f p = cv_feature_points[i][j];
            eigen_feature_points[i].push_back(Eigen::Vector2d(p.x, p.y));
        }
    }

    // Assuming detecting internal points of the checker board
    // left-top point of the checker board is origin point of the world coordinate
    // i.e. checker board is z=0 plane in the world coordinate 
    vector<Eigen::Vector2d> eigen_model_points;
    for(int i=0; i<checker_board_rows; ++i){
        for(int j=0; j<checker_board_cols; ++j){
            eigen_model_points.push_back(checker_board_length * Eigen::Vector2d(i+1, j+1));
        }
    }
    
    // 3. Calculate Homography for each image 
    vector<Eigen::Matrix<double, 3, 3>> homographies;
    for(int i=0; i<num_images; ++i){
        Eigen::Matrix<double, 3, 3> H = Estimate_Homography(eigen_model_points, eigen_feature_points[i]);
        homographies.push_back(H);
    } 
   
    // 4. Calculate Intrinsic Parameters with Closed Form
    Eigen::Matrix<double, 3, 3> K;
    K = Estimate_Intrinsic(homographies);
    cout << "##### Estimated Intrinsic Matrix #####" << endl;
    cout << K << endl;

    // 5. Calculate Extrinsic Parameters with Closed Form
    vector<Eigen::Matrix<double, 3, 4>> extrinsics;
    for(int i=0; i<num_images; ++i){
        Eigen::Matrix<double, 3, 4> E = Estimate_Extrinsic(homographies[i], K);
        extrinsics.push_back(E);
    }

    // 6. Calculate Distortion Coefficients with Least Squares
    Eigen::Vector2d k = Estimate_Distortion(K, extrinsics, eigen_model_points, eigen_feature_points);
    cout << "##### Estimated Distortion Parameters ##### " << endl;
    cout << k.transpose() << endl;
    
    Evaluate(K, k, extrinsics, eigen_model_points, eigen_feature_points);

    // 7. MLE for optimization
    MLE_Optimize(K, k, extrinsics, eigen_model_points, eigen_feature_points);

    // 8. Evaluation : Calculate Reprojection Error
    Evaluate(K, k, extrinsics, eigen_model_points, eigen_feature_points);

    return 0; 
}

Eigen::Matrix<double, 3, 3> 
Estimate_Homography
(vector<Eigen::Vector2d> &M, vector<Eigen::Vector2d> &m,  bool optimize, int verbose){

    size_t size = M.size(); // number of pair of points

    Eigen::MatrixXd A(2*size, 9); 

    // Calculate DLT solution
    for(size_t i=0; i<size; ++i){
        Eigen::Vector3d tmp(M[i](0), M[i](1), 1.0);
        tmp *= -m[i](0);
        A.row(2*i) << M[i](0), M[i](1), 1., 0., 0., 0., tmp(0), tmp(1), tmp(2);

        tmp << M[i](0), M[i](1), 1.0;
        tmp *= -m[i](1);
        A.row(2*i+1) << 0., 0., 0., M[i](0), M[i](1), 1., tmp(0), tmp(1), tmp(2);
    }

    // Solution for Ax=0;
    Eigen::Matrix<double, 9, 9> B = A.transpose() * A;
    Eigen::EigenSolver<Eigen::Matrix<double, 9, 9>> solver(B);
    Eigen::VectorXd eigenvalues = solver.eigenvalues().real();
    Eigen::MatrixXd eigenvectors = solver.eigenvectors().real();


    int argmin; 
    eigenvalues.minCoeff(&argmin);
    Eigen::VectorXd solution = eigenvectors.col(argmin);  // 9x1 Matrix

    // Optimization using MLE
    if(optimize){
        vector<double> parameters(9);
        for(size_t i=0; i<9; ++i){
            parameters[i] = solution(i);
        }
        
        ceres::Problem problem;
        for(size_t i=0; i<size; ++i){
            ceres::CostFunction * cost_function = new ceres::AutoDiffCostFunction<HomographyResidualFunction, 2, 9>(
                new HomographyResidualFunction(M[i], m[i])
            );
            problem.AddResidualBlock(cost_function, nullptr, parameters.data());
        }

        ceres::Solver::Options options;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        
        cout << summary.BriefReport() << endl;
        cout << "Initial : " << solution.transpose() << endl;

        for(size_t i=0; i<9; ++i){
            solution(i) = parameters[i];
        }
        cout << "After : " << solution.transpose() << endl;
    }

    Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> H(solution.data()); 

    if(verbose){
        cout << "##### Estimated Homography #####" << endl;
        cout << H << endl; 

        cout << "##### Calculate Reprojection Error #####" << endl;
        
        double error = 0.;
        for(size_t i=0; i<size; ++i){
            Eigen::Vector3d p(M[i](0), M[i](1), 1.0);
            Eigen::Vector3d reproject;
            reproject = H * p;
            reproject /= reproject(2);
            double dx = reproject(0) - m[i](0);
            double dy = reproject(1) - m[i](1);
            error += (dx*dx + dy*dy);
            
            if(verbose == 2){
                cout << "Ans " << to_string(i+1) << ": " << m[i].transpose() << endl;
                cout << "Pred"<< to_string(i+1) << ": " << reproject(0) << " " << reproject(1) << endl;
            }
        }
        error /= size;
        cout << "MSE : " << error << endl;
    }
    cout << endl;

    return H;
}

Eigen::Matrix<double, 3, 3> 
Estimate_Intrinsic 
(vector<Eigen::Matrix<double, 3, 3>>& homographies){
    assert(homographies.size() >= 3);

    size_t size = homographies.size(); // number of homography matrices
    Eigen::MatrixXd A(2*size, 6); 

    for(size_t i=0; i<size; ++i){
        Eigen::Vector3d h1= homographies[i].col(0);
        Eigen::Vector3d h2= homographies[i].col(1);
        Eigen::Vector3d h3= homographies[i].col(2);

        A.row(2*i) << h1(0)*h2(0), h1(0)*h2(1) + h1(1)*h2(0), h1(1)*h2(1), h1(0)*h2(2) + h1(2)*h2(0), h1(1)*h2(2) + h1(2)*h2(1),  h1(2)*h2(2);

        A.row(2*i+1) <<  h1(0)*h1(0)- h2(0)*h2(0), 2*(h1(0)*h1(1)-h2(0)*h2(1)), h1(1)*h1(1)- h2(1)*h2(1), 2*(h1(0)*h1(2)-h2(0)*h2(2)), 2*(h1(1)*h1(2)-h2(1)*h2(2)),  h1(2)*h1(2)- h2(2)*h2(2);
    }

     // Solution for Ax=0;
    Eigen::Matrix<double, 6, 6> B = A.transpose() * A;
    Eigen::EigenSolver<Eigen::Matrix<double, 6, 6>> solver(B);
    Eigen::VectorXd eigenvalues = solver.eigenvalues().real();
    Eigen::MatrixXd eigenvectors = solver.eigenvectors().real();

    int argmin; 
    eigenvalues.minCoeff(&argmin);
    Eigen::VectorXd solution = eigenvectors.col(argmin);  // 6x1 Matrix

    // Intrinsic Parameter
    double v = (solution(1)*solution(3) - solution(0)*solution(4))/(solution(0)*solution(2) - solution(1)*solution(1));
    double l = solution(5) - (solution(3) * solution(3) + v * (solution(1)*solution(3) - solution(0)*solution(4)) ) / solution(0);
    double a = sqrt(l / solution(0));
    double b = sqrt((l * solution(0)) / (solution(0)*solution(2) - solution(1)*solution(1)));
    double c = -1.0 * solution(1) * a * a * (b / l);
    double u = c * (v / b) - solution(3) * a * (a / l);

    Eigen::Matrix<double, 3, 3> K;
    K.row(0) << a, c, u;
    K.row(1) << 0., b, v;
    K.row(2) << 0., 0., 1.;

    return K;
}

Eigen::Matrix<double, 3, 4> 
Estimate_Extrinsic 
(Eigen::Matrix<double, 3, 3> H, Eigen::Matrix<double, 3, 3> K){
    Eigen::Matrix<double, 3, 4> E; // Extrinsic Matrix
    Eigen::Vector3d r1, r2, r3, t; // Rotation, Translation 
    Eigen::Matrix<double, 3, 3> K_inverse = K.inverse(); // Inverse of Intrinsic Matrix 
    Eigen::Vector3d h1 = H.col(0);
    Eigen::Vector3d h2 = H.col(1);
    Eigen::Vector3d h3 = H.col(2);
    double e = 1e-3; // Tolerance
    double norm1, norm2;
    
    r1  = K_inverse * h1;
    norm1 = r1.norm();
    r1 /= norm1;

    r2 = K_inverse * h2;
    norm2 = r2.norm();
    r2 /= norm2;
    
    r3 = r1.cross(r2);
    t = (K_inverse * h3) / norm1;

    if(abs(norm1 - norm2) > e){
        cout << "[Warn] Difference of Norm is greater than the tolerance 1e-6" << endl; 
    }

    Eigen::Matrix<double, 3, 3> R;
    R << r1, r2, r3;

    // Find closest Rotation Matrix
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::MatrixXd U = svd.matrixU();
    Eigen::MatrixXd V = svd.matrixV();
    R = U * V.transpose();

    E << R, t;
    return E;
}

Eigen::Vector2d 
Estimate_Distortion
(Eigen::Matrix<double, 3, 3> K, vector<Eigen::Matrix<double, 3, 4>> &extrinsics, vector<Eigen::Vector2d> &M, vector<vector<Eigen::Vector2d>> &m){
    size_t num_points = M.size();
    size_t num_images = m.size();
    size_t size = 2 * num_points * num_images;

    // Intrinsic Parameter - ignore skewness parameter gamma
    double u0 = K.row(0)(2);
    double v0 = K.row(1)(2);

    Eigen::MatrixXd A(size, 2);
    Eigen::MatrixXd d(size, 1);

    for(size_t i=0; i<num_images; ++i){
        for(size_t j=0; j<m.size(); ++j){

            // Distorted Image pixel Coordinates
            double u_distorted = m[i][j](0);
            double v_distorted = m[i][j](1);
            
            // Ideal Image Normalized Coordinates
            Eigen::Vector3d extened_p(M[j](0), M[j](1), 0.);
            Eigen::Vector3d p = extrinsics[i] * extened_p.homogeneous(); // 3D points in Camera Coordinates
            double x = p(0) / p(2);
            double y = p(1) / p(2);
            double dis = x*x + y*y;
            
            // Ideal Image pixel Coordinates
            Eigen::Vector3d q = K*p;
            double u = q(0) / q(2);
            double v = q(1) / q(2);

            // Constraints
            A.row(2*i*num_points + 2*j) << (u-u0)*dis, (u-u0)*dis*dis;
            d.row(2*i*num_points + 2*j) << u_distorted - u;
            
            A.row(2*i*num_points + 2*j +1) << (v-v0)*dis, (v-v0)*dis*dis;
            d.row(2*i*num_points + 2*j +1) << v_distorted - v; 
        }
    }
    Eigen::Vector2d k = A.fullPivHouseholderQr().solve(d);
    if(isnan(k(0)) || isnan(k(1))){
        k(0) = 0.;
        k(1) = 0.;
    }
    return k; 
}


void 
MLE_Optimize
(Eigen::Matrix<double, 3, 3> &K, Eigen::Vector2d &k, vector<Eigen::Matrix<double, 3, 4>> &extrinsics, vector<Eigen::Vector2d> &M, vector<vector<Eigen::Vector2d>> &m){
    size_t num_images = m.size();
    size_t num_points = M.size();

    vector<double> intrinsic_vector(5);
    vector<vector<double>> extrinsic_vectors(num_images);
    vector<double> distortion_vector(2); 

    intrinsic_vector[0] = K.row(0)(0); // a
    intrinsic_vector[1] = K.row(1)(1); // b
    intrinsic_vector[2] = K.row(0)(1); // c
    intrinsic_vector[3] = K.row(0)(2); // u0
    intrinsic_vector[4] = K.row(1)(2); // v0

    distortion_vector[0]= k(0);
    distortion_vector[1]= k(1);

    for(int w=0; w<num_images; ++w){
        Eigen::Matrix<double,3,3> rotation_matrix = extrinsics[w].block<3, 3>(0, 0);
        Eigen::Vector3d translation_vector = extrinsics[w].col(3);

        Eigen::AngleAxisd angle_axis(rotation_matrix);
        Eigen::Vector3d angle_axis_vector = angle_axis.axis() * angle_axis.angle(); 

        for(size_t p=0; p<3; ++p){
                extrinsic_vectors[w].push_back(angle_axis_vector(p));
        }
        for(size_t p=0; p<3; ++p){
                extrinsic_vectors[w].push_back(translation_vector(p));
        }
    }

    ceres::Problem problem;
    for(size_t i=0; i<num_images; ++i){
        for(size_t j=0; j<num_points; ++j){
            ceres::CostFunction * cost_function = new ceres::AutoDiffCostFunction<MLEResidualFunction, 2, 5, 2, 6>(
                new MLEResidualFunction(M[j], m[i][j])
            );
            problem.AddResidualBlock(cost_function, nullptr, intrinsic_vector.data(), distortion_vector.data(), extrinsic_vectors[i].data());
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;  
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.max_num_iterations = 1000;  // 최대 반복 횟수
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    cout << summary.BriefReport() << endl;

    K.row(0) << intrinsic_vector[0], intrinsic_vector[2], intrinsic_vector[3];
    K.row(1) << 0., intrinsic_vector[1], intrinsic_vector[4];
    K.row(2) << 0., 0.,1.;

    k(0) =  distortion_vector[0];
    k(1) =  distortion_vector[1];

    for (size_t i=0; i<num_images; ++i) {
        Eigen::Vector3d optimized_rotation_vector(
            extrinsic_vectors[i][0], extrinsic_vectors[i][1], extrinsic_vectors[i][2]);
        Eigen::Vector3d optimized_translation_vector(
            extrinsic_vectors[i][3], extrinsic_vectors[i][4], extrinsic_vectors[i][5]);
        
        Eigen::AngleAxisd optimized_angle_axis(optimized_rotation_vector.norm(), optimized_rotation_vector.normalized());
        Eigen::Matrix3d optimized_rotation_matrix = optimized_angle_axis.toRotationMatrix();
        
        extrinsics[i].block<3, 3>(0, 0) = optimized_rotation_matrix;
        extrinsics[i].col(3) = optimized_translation_vector;
        
    }

    cout << "##### Optimized Intrinsic Matrix #####" << endl;
    cout << K << endl;
    cout << k.transpose() << endl;

    return;
}

void 
Evaluate
(Eigen::Matrix<double, 3, 3> K, Eigen::Vector2d k, vector<Eigen::Matrix<double, 3, 4>> &extrinsics, vector<Eigen::Vector2d> &M, vector<vector<Eigen::Vector2d>> &m){
    size_t num_images = m.size();
    size_t num_points = M.size();
    
    // Camera Intrinsic Parameters
    double u0 = K.row(0)(2); 
    double v0 = K.row(1)(2);

    double error = 0;
    for(size_t i=0; i<num_images; ++i){
        for(size_t j=0; j<num_points; ++j){
            Eigen::Vector3d p(M[j](0), M[j](1), 0.);
            Eigen::Vector3d reproject = extrinsics[i] * p.homogeneous();
            
            double x = reproject(0) / reproject(2);
            double y = reproject(1) / reproject(2);
            double dis = x*x + y*y;

            reproject = K * reproject; 
            double u = reproject(0) / reproject(2);
            double v = reproject(1) / reproject(2);

            // Radial Distortion Model
            double u_hat = u + (u-u0)*(k(0)*dis + k(1)*dis*dis);
            double v_hat = v + (v-u0)*(k(0)*dis + k(1)*dis*dis);

            double du = u_hat - m[i][j](0);
            double dv = v_hat - m[i][j](1);
            error += du*du + dv*dv;
        }
    }
    error /= num_images * num_points;

    cout << "##### Calibration Reprojection Error #####" << endl;
    cout << "RMSE : " << sqrt(error) << endl; 
    return;
}