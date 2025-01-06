#include <opencv2/opencv.hpp>
#include <string>

#define DEFAULT 0
#define AVG 1
#define L1 2
#define L2 3

using namespace std;

string image_file = "C:/Users/qkrwh/OneDrive/Desktop/VClab/visual-slam-2024-fall-indi2-code/revision/ch5/imageBasics/distorted.png";


uchar calculate_pixel(cv::Mat image, double v_distorted, double u_distorted, int mode){
  int y = (int) v_distorted;
  int x = (int) u_distorted;
  int rows = image.rows;
  int cols = image.cols;
  
  if(mode == DEFAULT) {
    return image.at<uchar>(y, x);
  }
  if(mode == AVG){ 
    double val = 0;
    int cnt = 0;
    for(int i=x; i<=x+1; ++i){
      for(int j=y; j<=y+1; ++j){
        if(i >= cols || j >= rows) continue;
        val += image.at<uchar>(j,i);
        cnt++;
      }
    }
    return (uchar)(val/cnt);
  }
  if(mode == L1){ // Weighted average of neighboring pixels based on L2 distance
    double val = 0;
    double cnt = 0;
    for(int i=x; i<=x+1; ++i){
      for(int j=y; j<=y+1; ++j){
        if(i >= cols || j >= rows) continue;
        double dx = cv::abs(i-u_distorted);
        double dy = cv::abs(j-v_distorted);
        double dis = dx + dy;
        val += 1/dis * image.at<uchar>(j,i);
        cnt += 1/dis;
      }
    }
    return (uchar)(val/cnt);
  }
  if(mode == L2){ // Weighted average of neighboring pixels based on L2 distance
    double val = 0;
    double cnt = 0;
    for(int i=x; i<=x+1; ++i){
      for(int j=y; j<=y+1; ++j){
        if(i >= cols || j >= rows) continue;
        double dx = i-u_distorted;
        double dy = j-v_distorted;
        double dis = cv::sqrt(dx*dx + dy*dy);
        val += 1/dis * image.at<uchar>(j,i);
        cnt += 1/dis;
      }
    }
    return (uchar)(val/cnt);
  }
  else return (uchar) 0;
}

int main(int argc, char **argv) {

  double k1 = -0.28340811, k2 = 0.07395907, p1 = 0.00019359, p2 = 1.76187114e-05;
  double fx = 458.654, fy = 457.296, cx = 367.215, cy = 248.375;

  cv::Mat image = cv::imread(image_file, 0);
  int rows = image.rows, cols = image.cols;
  cv::Mat image_undistort = cv::Mat(rows, cols, CV_8UC1); // unsigned int with channel 1

  for (int v = 0; v < rows; v++) {
    for (int u = 0; u < cols; u++) {
      double x = (u - cx) / fx, y = (v - cy) / fy;
      double r = sqrt(x * x + y * y);
      double x_distorted = x * (1 + k1 * r * r + k2 * r * r * r * r) + 2 * p1 * x * y + p2 * (r * r + 2 * x * x);
      double y_distorted = y * (1 + k1 * r * r + k2 * r * r * r * r) + p1 * (r * r + 2 * y * y) + 2 * p2 * x * y;
      double u_distorted = fx * x_distorted + cx;
      double v_distorted = fy * y_distorted + cy;

      if (u_distorted >= 0 && v_distorted >= 0 && u_distorted < cols && v_distorted < rows) {
        image_undistort.at<uchar>(v, u) = calculate_pixel(image, v_distorted, u_distorted, L1);
      } else {
        image_undistort.at<uchar>(v, u) = 0;
      }
    }
  }

  cv::imshow("distorted", image);
  cv::imshow("undistorted", image_undistort);
  cv::waitKey();
  return 0;
}
