#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/persistence.hpp>
#include <stdio.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace cv;

int main(int argc, const char *argv[]) 
{
  
  // The following matrix definition come from the .txt   
  // in the params/ directory
  float cm1[] = { 1295.05391064956, 0, 852.059290832478, \
                  0, 1305.03656463523, 449.820384967797, \
                  0, 0, 1
                };
  cv::Mat cameraMatrix1 = cv::Mat(3, 3, CV_32F, cm1);
  float cm2[] = {1282.12633358256, 0, 1064.55729787994, \
                0, 1283.63634507551, 411.109562608613, \
                0, 0, 1 
                };
  cv::Mat cameraMatrix2 = cv::Mat(3, 3, CV_32F, cm2);
 
  float dc1[] = {-0.157631083249387,-0.926860306246985,0,0,0};
  cv::Mat distCoeffs1 = cv::Mat(1, 5, CV_32F, dc1);

  float dc2[] = {-0.177583840228775,-0.403888257544192,0,0,0};
  cv::Mat distCoeffs2 = cv::Mat(1, 5, CV_32F, dc2);


  // cv::FileStorage im1fs("../params/intrinsicMatrix1.xml", cv::FileStorage::READ);
  // im1fs << "intrinsicMatrix1" << cameraMatrix1;
  // im1fs.release();  

  // cv::FileStorage im2fs("../params/intrinsicMatrix2.xml", cv::FileStorage::READ);
  // im2fs << "intrinsicMatrix2" << cameraMatrix2;
  // im2fs.release();  

  // cv::FileStorage dc1fs("../params/distortionCoefficients1.xml", cv::FileStorage::READ);
  // dc1fs << "distortionCoefficients1" << distCoeffs1;
  // dc1fs.release();  

  // cv::FileStorage dc2fs("../params/distortionCoefficients2.xml", cv::FileStorage::READ);
  // dc2fs << "distortionCoefficients2" << distCoeffs2;
  // dc2fs.release();  

  std::cout << "cameraMatrix1 : " << cameraMatrix1 << std::endl;
  std::cout << "cameraMatrix2 : " << cameraMatrix2 << std::endl;
  std::cout << "distCoeffs1 : " << distCoeffs1 << std::endl;
  std::cout << "distCoeffs2 : " << distCoeffs2 << std::endl;

  // Trying to undistort the image using the camera parameters obtained from calibration
  cv::Mat image;
  image = cv::imread("../sample.jpg");
  cv::Size imageSize(cv::Size(image.cols,image.rows));

  cv::Mat map_x1, map_x2, map_y1, map_y2;

  // Computes the undistortion and rectification transformation map for the left camera
  cv::initUndistortRectifyMap(cameraMatrix1, distCoeffs1, cv::Mat(),cv::getOptimalNewCameraMatrix(cameraMatrix1, \
  distCoeffs1, imageSize, 1, imageSize, 0),imageSize, CV_16SC2, map_x1, map_y1);

  // Computes the undistortion and rectification transformation map for the right camera
  cv::initUndistortRectifyMap(cameraMatrix2, distCoeffs2, cv::Mat(),cv::getOptimalNewCameraMatrix(cameraMatrix2, \
  distCoeffs2, imageSize, 1, imageSize, 0),imageSize, CV_16SC2, map_x2, map_y2);

  // Save X-map for the left camera
  cv::FileStorage lxmap_storage("../data/left/map_X.xml", cv::FileStorage::WRITE);
  lxmap_storage << "map_X" << map_x1;
  lxmap_storage.release();  

  // Save Y-map for the left camera
  cv::FileStorage lymap_storage("../data/left/map_Y.xml", cv::FileStorage::WRITE);
  lymap_storage << "map_Y" << map_y1;
  lymap_storage.release();  

  // Save X-map for the right camera
  cv::FileStorage rxmap_storage("../data/right/map_X.xml", cv::FileStorage::WRITE);
  rxmap_storage << "map_X" << map_x2;
  rxmap_storage.release();  

  // Save Y-map for the right camera
  cv::FileStorage rymap_storage("../data/right/map_Y.xml", cv::FileStorage::WRITE);
  rymap_storage << "map_Y" << map_y2;
  rymap_storage.release();  

  return 0;
}
