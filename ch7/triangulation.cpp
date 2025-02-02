#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

void find_feature_matches(
  const cv::Mat &img_1, const cv::Mat &img_2,
  vector<cv::KeyPoint> &keypoints_1,
  vector<cv::KeyPoint> &keypoints_2,
  vector<cv::DMatch> &matches);

void pose_estimation_2d2d(
  const vector<cv::KeyPoint> &keypoints_1,
  const vector<cv::KeyPoint> &keypoints_2,
  const vector<cv::DMatch> &matches,
  cv::Mat &R, cv::Mat &t);

void triangulation(
  const vector<cv::KeyPoint> &keypoint_1,
  const vector<cv::KeyPoint> &keypoint_2,
  const vector<cv::DMatch> &matches,
  const cv::Mat &R, const cv::Mat &t,
  vector<cv::Point3d> &points
);

inline cv::Scalar get_color(float depth) {
  float up_th = 50, low_th = 10, th_range = up_th - low_th;
  if (depth > up_th) depth = up_th;
  if (depth < low_th) depth = low_th;
  return cv::Scalar(255 * depth / th_range, 0, 255 * (1 - depth / th_range));
}

cv::Point2f pixel2cam(const cv::Point2d &p, const cv::Mat &K);

int main(int argc, char **argv) {
  if (argc != 3) {
    cout << "usage: triangulation img1 img2" << endl;
    return 1;
  }

  // Read Image 
  cv::Mat img_1 = cv::imread(argv[1], cv::IMREAD_COLOR);
  cv::Mat img_2 = cv::imread(argv[2], cv::IMREAD_COLOR);

  vector<cv::KeyPoint> keypoints_1, keypoints_2;
  vector<cv::DMatch> matches;
  find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
  cout << "Found" << matches.size() << " number of feature matches!" << endl;

  cv::Mat R, t;
  pose_estimation_2d2d(keypoints_1, keypoints_2, matches, R, t);

  vector<cv::Point3d> points;
  triangulation(keypoints_1, keypoints_2, matches, R, t, points);

  cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
  cv::Mat img1_plot = img_1.clone();
  cv::Mat img2_plot = img_2.clone();
  for (int i = 0; i < matches.size(); i++) {
    float depth1 = points[i].z;
    cout << "depth: " << depth1 << endl;
    cv::Point2d pt1_cam = pixel2cam(keypoints_1[matches[i].queryIdx].pt, K);
    cv::circle(img1_plot, keypoints_1[matches[i].queryIdx].pt, 2, get_color(depth1), 2);

    cv::Mat pt2_trans = R * (cv::Mat_<double>(3, 1) << points[i].x, points[i].y, points[i].z) + t;
    float depth2 = pt2_trans.at<double>(2, 0);
    cv::circle(img2_plot, keypoints_2[matches[i].trainIdx].pt, 2, get_color(depth2), 2);
  }
  cv::imshow("img 1", img1_plot);
  cv::imshow("img 2", img2_plot);
  cv::waitKey();

  return 0;
}

void find_feature_matches(const cv::Mat &img_1, const cv::Mat &img_2,
                          vector<cv::KeyPoint> &keypoints_1,
                          vector<cv::KeyPoint> &keypoints_2,
                          vector<cv::DMatch> &matches) {
  
  cv::Mat descriptors_1, descriptors_2;
  cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
  cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
  cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");
 
  // Oritented FAST
  detector->detect(img_1, keypoints_1);
  detector->detect(img_2, keypoints_2);

  //BRIEF descriptor
  descriptor->compute(img_1, keypoints_1, descriptors_1);
  descriptor->compute(img_2, keypoints_2, descriptors_2);

  // Feature Matching
  vector<cv::DMatch> match;
  // BFMatcher matcher ( NORM_HAMMING );
  matcher->match(descriptors_1, descriptors_2, match);

  // Calculate Min, Max Distance
  double min_dist = 10000, max_dist = 0;
  for (int i = 0; i < descriptors_1.rows; i++) {
    double dist = match[i].distance;
    if (dist < min_dist) min_dist = dist;
    if (dist > max_dist) max_dist = dist;
  }

  printf("-- Max dist : %f \n", max_dist);
  printf("-- Min dist : %f \n", min_dist);

  // Thresholding
  for (int i = 0; i < descriptors_1.rows; i++) {
    if (match[i].distance <= max(2 * min_dist, 30.0)) {
      matches.push_back(match[i]);
    }
  }
}

void pose_estimation_2d2d(
  const vector<cv::KeyPoint> &keypoints_1,
  const vector<cv::KeyPoint> &keypoints_2,
  const vector<cv::DMatch> &matches,
  cv::Mat &R, cv::Mat &t) {

  cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

  // Convert to vector<cv::Point2f>
  vector<cv::Point2f> points1;
  vector<cv::Point2f> points2;

  for (int i = 0; i < (int) matches.size(); i++) {
    points1.push_back(keypoints_1[matches[i].queryIdx].pt);
    points2.push_back(keypoints_2[matches[i].trainIdx].pt);
  }

  // Essential Matrix
  cv::Point2d principal_point(325.1, 249.7);        //相机主点, TUM dataset标定值
  int focal_length = 521;            //相机焦距, TUM dataset标定值
  cv::Mat essential_matrix;
  essential_matrix = cv::findEssentialMat(points1, points2, focal_length, principal_point);

  cv::recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
}

void triangulation(
  const vector<cv::KeyPoint> &keypoint_1,
  const vector<cv::KeyPoint> &keypoint_2,
  const vector<cv::DMatch> &matches,
  const cv::Mat &R, const cv::Mat &t,
  vector<cv::Point3d> &points) {
  cv::Mat T1 = (cv::Mat_<float>(3, 4) <<
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0);
  cv::Mat T2 = (cv::Mat_<float>(3, 4) <<
    R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
    R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
    R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0)
  );

  cv::Mat K = (cv::Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
  vector<cv::Point2f> pts_1, pts_2;
  for (cv::DMatch m:matches) {
    pts_1.push_back(pixel2cam(keypoint_1[m.queryIdx].pt, K));
    pts_2.push_back(pixel2cam(keypoint_2[m.trainIdx].pt, K));
  }

  cv::Mat pts_4d;
  cv::triangulatePoints(T1, T2, pts_1, pts_2, pts_4d);

  for (int i = 0; i < pts_4d.cols; i++) {
    cv::Mat x = pts_4d.col(i);
    x /= x.at<float>(3, 0); // 归一化
    cv::Point3d p(
      x.at<float>(0, 0),
      x.at<float>(1, 0),
      x.at<float>(2, 0)
    );
    points.push_back(p);
  }
}

cv::Point2f pixel2cam(const cv::Point2d &p, const cv::Mat &K) {
  return cv::Point2f
    (
      (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
      (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
}

