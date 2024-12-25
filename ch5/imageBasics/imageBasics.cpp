#include <iostream>
#include <chrono>
#include <string>

using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

int main(int argc, char **argv) {
  string img_path = "C:/Users/qkrwh/OneDrive/Desktop/VClab/visual-slam-2024-fall-indi2-code/revision/ch5/imageBasics/cat.jpg";
  
  cv::Mat image;
  image = cv::imread(img_path); 

  if(image.data == nullptr){
    cerr << "Document " << img_path  << " doesn't exist" << endl;
    return 0;
  }

  // Image Type - only grayscale and color image 
  if(image.type() != CV_8UC1 && image.type() != CV_8UC3){
    cout << "Image should be grayscale or color image" << endl;
    return 0;
  }

  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  for(int y = 0; y < image.rows; y++){
    unsigned char * row_ptr = image.ptr<unsigned char>(y); // y-th row pointer
    for(int x = 0; x < image.cols; x++){
      unsigned char * data_ptr = &row_ptr[x * image.channels()];
      for(int c=0; c!=image.channels(); c++){
        unsigned char data = data_ptr[c];
      }
    }
  }
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);

  cout << "Loop over each pixel takes " << time_used.count() << " sec" << endl;

  // cv::Mat Copy

  // 1. Shallow Copy
  cv::Mat another = image;
  another(cv::Rect(0,0, 100, 100)).setTo(0);
  cv::imshow("image", image);
  cv::waitKey(0);

  // 2. Deep copy
  cv::Mat clone = image.clone();
  clone(cv::Rect(100, 100, 200, 200)).setTo(0);
  cv::imshow("image", image);
  cv::imshow("clone", clone);
  cv::waitKey(0);

  cv::destroyAllWindows();
  return 0;
}
