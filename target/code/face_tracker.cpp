
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <ctime> 
#include <atomic>
#include <syslog.h>
#include <thread>
#include <map>
#include <opencv2/core/base.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/tracking/tracker.hpp>
#include <opencv2/core/ocl.hpp>
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/face.hpp"
#include "concurrent_queue_cv.h"
#include "cv_barrier.h"


#define WIDTH                   (640)
#define HEIGHT                  (480)
#define LEFT_DISPLAY_WINDOW     "Left-Tracking"
#define RIGHT_DISPLAY_WINDOW    "Right-Tracking"
#define TEST_IMG_WIDTH          (320)
#define TEST_IMG_HEIGHT         (243)
#define MIN_DISTANCE            (5)

#define INFO_LOG(...) \
do {    \
    syslog(LOG_INFO, ##__VA_ARGS__); \
}while(0)

#define ERR_LOG(...) \
do {    \
    syslog(LOG_ERR, ##__VA_ARGS__); \
}while(0)

#define CRIT_LOG(...) \
do {    \
    syslog(LOG_CRIT, ##__VA_ARGS__); \
}while(0)


using namespace std::chrono;
using namespace std;
using namespace cv;
using namespace cv::face;


// initialize values for StereoBM parameters
int numDisparities = 144;
int blockSize = 39;
int preFilterType = 1;
int preFilterSize = 9;
int preFilterCap = 48;
int textureThreshold = 9;
int uniquenessRatio = 16;
int speckleRange = 6;
int speckleWindowSize = 24;
int disp12MaxDiff = 0;
float minDisparity = 0.0f;
float M = 4.1320842742919922e+01;


//Initialize variables to store the maps for stereo rectification
cv::Mat stereo_Map_LX, stereo_Map_LY;
cv::Mat stereo_Map_RX, stereo_Map_RY;

// declare atomic bool variable as the program terminartion flag
atomic<bool>terminate_flag{false};

// Global concurrent barrier to synchornize the two facial recognition threads
CVBarrier barrier {2u};

// vector<std::string>faces{"P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", \
// "P9", "P10", "P11", "P12", "P13", "P14", "P15", "Shuran", "P17", "P18", "P19", "P20", "P21"};

vector<std::string>faces{"P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", \
"P9", "P10", "P11", "P12", "P13", "P14", "P15", "Shuran"};

// declare a map of (string, int) pairs for detected labels of both cameras
std::map<std::string,int> label_map;
std::map<std::string,concurrent_queue_cv<std::string>> dist_map;
std::map<std::string,concurrent_queue_cv<cv::Mat>> mat_map;

// Convert to string
#define SSTR( x ) static_cast< std::ostringstream & >( \
( std::ostringstream() << std::dec << x ) ).str()

// Program Status Enum types
typedef enum {
    RECOGNIZE=1,
    TRACK
}prog_status_t;


void estimate_distance(cv::Ptr<cv::StereoSGBM>&& stereo)
{
    cv::Mat disp, disparity, depth_map;
    cv::Mat imgL, imgR;
    cv::Mat imgL_gray, imgR_gray;
    cv::Mat mean, stddev, mask2;
    int min_cols, min_rows;

    while(true) {

        // return the function if the termination flag is set
        if(terminate_flag){
            break;
        }

        // Pop matries from both left and right cameras
        mat_map[LEFT_DISPLAY_WINDOW].pop(imgL);
        mat_map[RIGHT_DISPLAY_WINDOW].pop(imgR);

        // Convert matries to gray-scaled
        cv::cvtColor(imgL, imgL_gray, cv::COLOR_BGR2GRAY);
        cv::cvtColor(imgR, imgR_gray, cv::COLOR_BGR2GRAY);

        // Applying stereo image rectification on the left image
        cv::remap(imgL_gray,
                imgL_gray,
                stereo_Map_LX,
                stereo_Map_LY,
                cv::INTER_LANCZOS4,
                cv::BORDER_CONSTANT,
                0);

        // Applying stereo image rectification on the right image
        cv::remap(imgR_gray,
                imgR_gray,
                stereo_Map_RX,
                stereo_Map_RY,
                cv::INTER_LANCZOS4,
                cv::BORDER_CONSTANT,
                0);
                
        // Resize matrices
        min_cols = min(imgL.cols, imgR.cols);
        min_rows = min(imgL.rows, imgL.rows);
        resize(imgL_gray, imgL_gray, Size(min_rows, min_cols), INTER_LINEAR);
        resize(imgR_gray, imgR_gray, Size(min_rows, min_cols), INTER_LINEAR);

        try{
            stereo->compute(imgL_gray,imgR_gray,disp);
        }
        catch(exception &e){
            cout << "exception caught: " << e.what() << endl;
            break;
        }
        
        // NOTE: compute returns a 16bit signed single channel image,
        // CV_16S containing a disparity map scaled by 16. Hence it 
        // is essential to convert it to CV_16S and scale it down 16 times.

        disp.convertTo(disparity,CV_8U);

        // Scaling down the disparity values and normalizing them
        // disparity = (disparity/(float)16.0 - (float)minDisparity)/((float)numDisparities);

        // Calculating disparity to depth map using the following equation
        // ||    depth = M * (1/disparity)   ||
        depth_map = (float)M/disparity;

        // Calculating the average depth of the object closer than the safe distance
        cv::meanStdDev(depth_map, mean, stddev, mask2);

        // Obtain the min and max of mean
        double min, max;
        cv::minMaxLoc(mean, &min, &max);

        // Printing the warning text with object distance
        char dist[10];
        std::sprintf(dist, "%.2f",mean.at<double>(0,0));

        // Push the dist text to distance queue twice as there are two threads consuming its components 
        dist_map[LEFT_DISPLAY_WINDOW].push(dist);
        dist_map[RIGHT_DISPLAY_WINDOW].push(dist);
    }
}

Rect2d&& detect_face( CascadeClassifier &face_cascade, Mat frame )
{
    Mat frame_gray;
    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    // Detect faces
    std::vector<Rect> faces;
    face_cascade.detectMultiScale( frame_gray, faces );
    if(faces.size() == 0){
        return std::move(Rect2d(0,0,0,0)); // mark Rect (0,0,0,0) as the failure to detect faces
    }

    // Return the coordinate of the first detected face
    return std::move(Rect2d(faces[0].x, faces[0].y, faces[0].width, faces[0].height));
}

int inline recognize_face(Mat face, Ptr<FisherFaceRecognizer> &model)
{
    // Convert the matrix from RGB to Grayscale
    Mat face_gray;
    cvtColor( face, face_gray, COLOR_BGR2GRAY );
    // Resize the face_mat to be the same size as the train image size
    resize(face_gray, face_gray, Size(TEST_IMG_WIDTH, TEST_IMG_HEIGHT), INTER_LINEAR);
    // Predict and return the label
    return model->predict(face_gray);
}

std::string get_time()
{
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    return std::ctime(&now_time);
}

void recognise_and_track(VideoCapture&& cap, std::string orientation)
{
    Mat img;
    Rect2d face;
    Rect2d bbox;
    Ptr<Tracker> tracker;
    double timer{0.0};
    float fps{0.0};
    int predictedLabel;
    prog_status_t thread_mode {RECOGNIZE};
    // Create the recognizer
    Ptr<FisherFaceRecognizer> model = FisherFaceRecognizer::create();
    // Load the trainer model
    model->read("../../train.yml");
    // Create the face cascade detector
    CascadeClassifier face_cascade;
    String face_cascade_name = "../haarcascade_frontalface_alt.xml";
    // Load the cascades
    if( !face_cascade.load( face_cascade_name ) )
    {
        cout << "(!)Error loading face cascade\n";
        exit(-1);
    };

    while (true)
    {
        // return the function if the termination flag is set
        if(terminate_flag){
            // destroy the current display window
            destroyWindow(orientation);
            // push a dummy matrix to lcq and rcq to make the disparity map thread proceed
            mat_map[LEFT_DISPLAY_WINDOW].push(img);
            mat_map[RIGHT_DISPLAY_WINDOW].push(img);
            return;
        }

        timer = (double)getTickCount();
        cap >> img;

        if( img.empty())
        {
            cout << "No captured frame !\n";
            continue;
        }

        // Display tracker type on frame
        putText(img, "KCF Tracker", Point(100,20), FONT_HERSHEY_SIMPLEX, \
            0.75, Scalar(50,170,50),2);

        switch(thread_mode)
        {
            case RECOGNIZE:
            {
                // Apply the classifier to img
                face = detect_face(std::ref(face_cascade), img);
        
                if(face.width != 0 && face.height != 0 ){

                    // Recognize the face by predicting the label
                    predictedLabel = recognize_face(cv::Mat(img, face), model);
                    barrier.Wait();
                    // Update the corresponding mapper entry
                    label_map[orientation] = predictedLabel;
                    // Update the bounding box 
                    bbox.x = face.x;
                    bbox.y = face.y;
                    bbox.width = face.width;
                    bbox.height = face.height;
                    // Create a KCF tracker
                    tracker = TrackerKCF::create();
                    tracker->init(img, std::ref(bbox));
                    // update thread mode
                    thread_mode = TRACK;
                }
                break;
            }
            case TRACK:
            {
                bool success = tracker->update(img, std::ref(bbox));
                if(!success || (label_map[LEFT_DISPLAY_WINDOW] != label_map[RIGHT_DISPLAY_WINDOW])){

                    // print the error message if tracking update failed
                    if(!success){
                        // cout << orientation << "tracking failed" << endl;
                        putText(img, "Tracking failure detected", Point(100,80), \
                            FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,255),2);
                    }

                    // cout << orientation << "left label = " << label_map[LEFT_DISPLAY_WINDOW] << endl;
                    // cout << orientation << "right label = " << label_map[RIGHT_DISPLAY_WINDOW] << endl;

                    // Clear the label_map value
                    label_map[orientation] = -1;
                    // Clear the tracker
                    tracker->clear();
                    // update thread mode
                    thread_mode = RECOGNIZE;
                    
                    break;
                }

                // Tracking success : Draw the tracked object
                rectangle(img, bbox, Scalar( 255, 0, 0 ), 2, 1 );
                // Create cropped matrix based on the bbox size
                Rect roi{(int)bbox.x, (int)bbox.y, (int)bbox.width, (int)bbox.height};
                Mat cropped_img = cv::Mat(img, roi);
                mat_map[orientation].push(cropped_img);

                // Display recognition anotation on frame 
                if(predictedLabel > faces.size()){
                    putText(img, "Stranger", Point(bbox.x-6,bbox.y), FONT_HERSHEY_SIMPLEX, \
                    0.75, Scalar(50,170,50),2);
                }
                else{
                    putText(img, faces[predictedLabel - 1], Point(bbox.x-6,bbox.y), FONT_HERSHEY_SIMPLEX, \
                    0.75, Scalar(50,170,50),2);
                }
                
                // Display the distance value if possible
                putText(img, "Distance : ", Point(100,70), FONT_HERSHEY_SIMPLEX, \
                    0.75, Scalar(50,170,50),2);

                if(dist_map[orientation].size()){
                    std::string dist;
                    dist_map[orientation].pop(dist);
                    putText(img, dist + " cm", Point(235,70), FONT_HERSHEY_SIMPLEX, \
                    0.75, Scalar(50,170,50),2);
                    if(stoi(dist) < MIN_DISTANCE){
                        putText(img, "Warning, too close !", Point(100,93), FONT_HERSHEY_SIMPLEX, \
                    0.75, Scalar(0,0,255),2);
                    }
                }
                break;
            }

            default:
                break;
        }

        // Calculate Frames per second (FPS)
        fps = getTickFrequency() / ((double)getTickCount() - timer);
        // Display FPS on frame
        putText(img, "FPS : " + SSTR(int(fps)), Point(100,50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50), 2);
        // Log the FPS value 
        INFO_LOG("camera=%s FPS=%.2f",orientation.c_str(), fps);
        // Display frame.
        imshow(orientation, img);
        // Exit if ESC pressed.
        int k = waitKey(1);
        if(k == 27)
        {
            terminate_flag = true;
        }
    }
}


int main(int argc, char **argv)
{
    int CamL_id{1}; // Camera ID for left camera
    int CamR_id{0}; // Camera ID for right camera

    // Set terminate_flag to false
    terminate_flag = false;
   
    // Reading the mapping values for stereo image rectification
    cv::FileStorage lmapx = cv::FileStorage("../../data/left/map_X.xml", cv::FileStorage::READ);
    lmapx["map_X"] >> stereo_Map_LX;
    lmapx.release();

    cv::FileStorage lmapy = cv::FileStorage("../../data/left/map_Y.xml", cv::FileStorage::READ);
    lmapy["map_Y"] >> stereo_Map_LY;
    lmapy.release();

    cv::FileStorage rmapx = cv::FileStorage("../../data/right/map_X.xml", cv::FileStorage::READ);
    rmapx["map_X"] >> stereo_Map_RX;
    rmapx.release();

    cv::FileStorage rmapy = cv::FileStorage("../../data/right/map_Y.xml", cv::FileStorage::READ);
    rmapy["map_Y"] >> stereo_Map_RY;
    rmapy.release();

    // Ensure the rectification matrices are not empty
    CV_Assert(stereo_Map_LX.size() != cv::Size(0,0));
    CV_Assert(stereo_Map_RX.size() != cv::Size(0,0));
    CV_Assert(stereo_Map_LY.size() != cv::Size(0,0));
    CV_Assert(stereo_Map_RY.size() != cv::Size(0,0));

    // Reading the stored the StereoBM parameters
    cv::FileStorage cv_file = cv::FileStorage("../../data/depth_estimation_params_cpp.xml", cv::FileStorage::READ);
    cv_file["numDisparities"] >> numDisparities;
    cv_file["blockSize"] >> blockSize;
    cv_file["preFilterType"] >> preFilterType;
    cv_file["preFilterSize"] >> preFilterSize;
    cv_file["preFilterCap"] >> preFilterCap;
    cv_file["minDisparity"] >> minDisparity;
    cv_file["textureThreshold"] >> textureThreshold;
    cv_file["uniquenessRatio"] >> uniquenessRatio;
    cv_file["speckleRange"] >> speckleRange;
    cv_file["speckleWindowSize"] >> speckleWindowSize;
    cv_file["disp12MaxDiff"] >> disp12MaxDiff;
    
    cv::Ptr<cv::StereoSGBM> stereo = cv::StereoSGBM::create();

    // updating the parameter values of the StereoSGBM algorithm
    stereo->setNumDisparities(numDisparities);
	stereo->setBlockSize(blockSize);
	stereo->setUniquenessRatio(uniquenessRatio);
	stereo->setSpeckleRange(speckleRange);
	stereo->setSpeckleWindowSize(speckleWindowSize);
	stereo->setDisp12MaxDiff(disp12MaxDiff);
	stereo->setMinDisparity(minDisparity);

    // initialize label map
    label_map[LEFT_DISPLAY_WINDOW] = -1;
    label_map[RIGHT_DISPLAY_WINDOW] = -1;

    cv::VideoCapture camL(CamL_id), camR(CamR_id);

    // Check if left camera is attched
    if (!camL.isOpened())
    {
        std::cout << "Could not open camera with index : " << CamL_id << std::endl;
        return -1;
    }

    // Check if right camera is attached
    if (!camR.isOpened())
    {
        std::cout << "Could not open camera with index : " << CamR_id << std::endl;
        return -1;
    }

    // Configure the camera resolutions
    camL.set(CAP_PROP_FRAME_WIDTH, WIDTH);//Setting the width of the video
    camL.set(CAP_PROP_FRAME_HEIGHT, HEIGHT);//Setting the height of the video//

    camR.set(CAP_PROP_FRAME_WIDTH, WIDTH);//Setting the width of the video
    camR.set(CAP_PROP_FRAME_HEIGHT, HEIGHT);//Setting the height of the video//
    
    thread tl{ recognise_and_track, std::move(camL) , LEFT_DISPLAY_WINDOW };
    thread tr{ recognise_and_track, std::move(camR) , RIGHT_DISPLAY_WINDOW };
    thread td{ estimate_distance, std::move(stereo) };

    tl.join();
    tr.join();
    td.join();

}

