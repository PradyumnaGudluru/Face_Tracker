#include <opencv2/opencv.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/persistence.hpp>
#include <stdio.h>
#include <iostream>

using namespace std;
using namespace cv;

// Defining the dimensions of checkerboard
int CHECKERBOARD[2]{7,10}; 

const char* params
    = "{ help h         |            | Print usage }"
      "{ orientation      |    left    | Camera orientation }";


int main(int argc, const char *argv[]) 
{
  CommandLineParser parser(argc, argv, params);
  if (parser.has("help"))
  {
      //print help information
      parser.printMessage();
      exit(0);
  }

  // obtain the camera orientation
  String orientation = parser.get<String>("orientation");
  
  // Creating vector to store vectors of 3D points for each checkerboard image
  std::vector<std::vector<cv::Point3f> > objpoints;

  // Creating vector to store vectors of 2D points for each checkerboard image
  std::vector<std::vector<cv::Point2f> > imgpoints;

  // Defining the world coordinates for 3D points
  std::vector<cv::Point3f> objp;
  for(int i{0}; i<CHECKERBOARD[1]; i++)
  {
    for(int j{0}; j<CHECKERBOARD[0]; j++)
      objp.push_back(cv::Point3f(j,i,0));
  }

  // Extracting path of individual image stored in a given directory
  std::vector<cv::String> images;
  // Path of the folder containing checkerboard images
  std::string img_path = "../images/" + orientation + "/*.jpg";

  cv::glob(img_path, images);

  cv::Mat frame, gray;
  // vector to store the pixel coordinates of detected checker board corners 
  std::vector<cv::Point2f> corner_pts;
  bool success;

  // Looping over all the images in the directory
  for(int i{0}; i<images.size(); i++)
  {
    frame = cv::imread(images[i]);
    cv::cvtColor(frame,gray,cv::COLOR_BGR2GRAY);

    // Finding checker board corners
    // If desired number of corners are found in the image then success = true  
    success = cv::findChessboardCorners(gray,cv::Size(CHECKERBOARD[0],CHECKERBOARD[1]), corner_pts, \
    cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_FAST_CHECK | cv::CALIB_CB_NORMALIZE_IMAGE);

    std::cout << "i = " << i << std::endl;
    /*
     * If desired number of corner are detected,
     * we refine the pixel coordinates and display 
     * them on the images of checker board
    */
    if(success)
    {
      cv::TermCriteria criteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 30, 0.001);

      // refining pixel coordinates for given 2d points.
      // cv::cornerSubPix(gray,corner_pts,cv::Size(11,11), cv::Size(-1,-1),criteria);
      cv::cornerSubPix(gray,corner_pts,cv::Size(5,5), cv::Size(-1,-1),criteria);

      // Displaying the detected corner points on the checker board
      cv::drawChessboardCorners(frame, cv::Size(CHECKERBOARD[0],CHECKERBOARD[1]), corner_pts,success);

      objpoints.push_back(objp);
      imgpoints.push_back(corner_pts);
    }

    // cv::imshow("Image",frame);
    // cv::waitKey(0);
  }

  // cv::destroyAllWindows();

  cv::Mat cameraMatrix,distCoeffs,R,T;

  std::cout << "calibrateCamera now " << std::endl;

  /*
   * Performing camera calibration by 
   * passing the value of known 3D points (objpoints)
   * and corresponding pixel coordinates of the 
   * detected corners (imgpoints)
  */
  cv::calibrateCamera(objpoints, imgpoints,cv::Size(gray.rows,gray.cols),cameraMatrix,distCoeffs,R,T);

  std::cout << "cameraMatrix : " << cameraMatrix << std::endl;
  std::cout << "distCoeffs : " << distCoeffs << std::endl;
  std::cout << "Rotation vector : " << R << std::endl;
  std::cout << "Translation vector : " << T << std::endl;

  // Trying to undistort the image using the camera parameters obtained from calibration
  cv::Mat image;
  image = cv::imread(images[0]);
  cv::Mat dst, mapx, mapy,new_camera_matrix;
  cv::Size imageSize(cv::Size(image.cols,image.rows));

  std::cout << "Computes the undistortion and rectification transformation map now " << std::endl;

  // Computes the undistortion and rectification transformation map.
  cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Mat(),cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, \
  imageSize, 1, imageSize, 0),imageSize, CV_16SC2, mapx, mapy);

  std::string data_path = "../data/" + orientation + "/";

  std::cout << "save maps now " << std::endl;

  // Save X-map
  cv::FileStorage xmap_storage(data_path + "map_X.xml", cv::FileStorage::WRITE);
  xmap_storage << "map_X" << mapx;
  xmap_storage.release();  

  // Save Y-map
  cv::FileStorage ymap_storage(data_path + "map_Y.xml", cv::FileStorage::WRITE);
  ymap_storage << "map_Y" << mapy;
  ymap_storage.release();  

  return 0;
}
