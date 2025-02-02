#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>

using namespace std;

int main(int argc, char **argv) {
  if (argc != 3) {
    cout << "usage: feature_extraction img1 img2" << endl;
    return 1;
  }
  
  // Read Image
  cv::Mat img_1 = cv::imread(argv[1], cv::IMREAD_COLOR);
  cv::Mat img_2 = cv::imread(argv[2], cv::IMREAD_COLOR);
  assert(img_1.data != nullptr && img_2.data != nullptr);

  //Initialize
  vector<cv::KeyPoint> keypoints_1, keypoints_2;
  cv::Mat descriptors_1, descriptors_2;
  cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create(); // key point
  cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create(); // descriptor
  cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming"); // Featrue Matching

  // Calculate oriented FAST corner
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  detector->detect(img_1, keypoints_1);
  detector->detect(img_2, keypoints_2);

  // Calculate BRIEF descriptor based on corner position 
  descriptor->compute(img_1, keypoints_1, descriptors_1);
  descriptor->compute(img_2, keypoints_2, descriptors_2);
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "extract ORB cost = " << time_used.count() << " seconds. " << endl;

  cv::Mat outimg1;
  drawKeypoints(img_1, keypoints_1, outimg1, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
  imshow("ORB features", outimg1);

  // Matching BRIEF descriptor using hamming distance
  vector<cv::DMatch> matches;
  t1 = chrono::steady_clock::now();
  matcher->match(descriptors_1, descriptors_2, matches);
  t2 = chrono::steady_clock::now();
  time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "match ORB cost = " << time_used.count() << " seconds. " << endl;

  // Calculate Min, Max Distance
  auto min_max = minmax_element(matches.begin(), matches.end(),
                                [](const cv::DMatch &m1, const cv::DMatch &m2) { return m1.distance < m2.distance; });
  double min_dist = min_max.first->distance;
  double max_dist = min_max.second->distance;

  printf("-- Max dist : %f \n", max_dist);
  printf("-- Min dist : %f \n", min_dist);

  // Filtering based on the distance
  std::vector<cv::DMatch> good_matches;
  for (int i = 0; i < descriptors_1.rows; i++) {
    if (matches[i].distance <= max(2 * min_dist, 30.0)) {
      good_matches.push_back(matches[i]);
    }
  }

  // Visualize the Result
  cv::Mat img_match;
  cv::Mat img_goodmatch;
  cv::drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
  cv::drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch);
  cv::imshow("all matches", img_match);
  cv::imshow("good matches", img_goodmatch);
  cv::waitKey(0);

  return 0;
}
