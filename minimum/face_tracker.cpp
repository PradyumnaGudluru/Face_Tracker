/**
 * @file face_tracker.cpp
 * @author Shuran Xu (shxu6388@colorado.edu)
 * @brief This program adopts the Haar-cascade detector and the KCF tracker
 * to continously detect and track a human face. In case the tracking fails,
 * the program resumes to detect the face before jumping to track the face.
 * @ref The implementation of the program uses the reference implementation of
 * the use of the Haar-cascade detector and the KCF tracker.
 * 
 * The reference implementation of the use of the Haar-cascade detector can
 * be found from the following link:
 * https://docs.opencv.org/4.1.1/db/d28/tutorial_cascade_classifier.html
 * 
 * The reference implementation of the use of the KCF tracker can
 * be found from the following link:
 * https://docs.opencv.org/4.x/d2/d0a/tutorial_introduction_to_tracker.html
 * 
 * @version 0.1
 * @date 2022-07-20
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/tracking/tracker.hpp>
#include <opencv2/core/ocl.hpp>
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <chrono>
#include <ctime> 


#define WIDTH                   (640)
#define HEIGHT                  (480)


using namespace cv;
using namespace std;

CascadeClassifier face_cascade;

// Convert to string
#define SSTR( x ) static_cast< std::ostringstream & >( \
( std::ostringstream() << std::dec << x ) ).str()

typedef enum {
    DETECT=1,
    TRACK
}prog_status_t;


Rect detect_face( Mat frame )
{
    Mat frame_gray;
    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );
    //-- Detect faces
    std::vector<Rect> faces;
    face_cascade.detectMultiScale( frame_gray, faces );
    if(faces.size() == 0){
        return Rect(0,0,0,0); // mark Rect (0,0,0,0) as the failure to detect faces
    }

    // Return the coordinate of the first detected face
    return Rect(faces[0].x, faces[0].y, faces[0].width, faces[0].height);
}


int main(int argc, char **argv)
{
    int camera_device = 0;
    Mat frame;
    String face_cascade_name = "../haarcascade_frontalface_alt.xml";
    // Load the cascades
    if( !face_cascade.load( face_cascade_name ) )
    {
        cout << "--(!)Error loading face cascade\n";
        return -1;
    };

    Ptr<Tracker> tracker;
    VideoCapture capture;
    Rect2d bbox;
    prog_status_t program_mode = DETECT;
    Rect face;
    
    // Read the video stream
    capture.open( camera_device );
    if ( ! capture.isOpened() )
    {
        cout << "Error opening video capture\n";
        return -1;
    }

    // Configure the camera resolutions
    capture.set(CAP_PROP_FRAME_WIDTH, WIDTH);//Setting the width of the video
    capture.set(CAP_PROP_FRAME_HEIGHT, HEIGHT);//Setting the height of the video//
    
    while ( capture.read(frame) )
    {
        if( frame.empty() )
        {
            cout << "No captured frame -- Break!\n";
            continue;
        }

        // Start timer
        double timer = (double)getTickCount();

        if(program_mode == DETECT){
            // Apply the classifier to the frame
            face = detect_face( frame );
            if(face.width != 0 && face.height != 0 ){
                // Update the bounding box 
                bbox.x = face.x;
                bbox.y = face.y;
                bbox.width = face.width;
                bbox.height = face.height;

                auto now = std::chrono::system_clock::now();
                std::time_t now_time = std::chrono::system_clock::to_time_t(now);
                cout << "Face detected at " << std::ctime(&now_time) << endl;
               
                tracker = TrackerKCF::create();
                tracker->init(frame, bbox);
                // Change the mode to TRACK
                program_mode = TRACK;
            }
        }
        else{
            // Update the tracking result
            if (tracker->update(frame, bbox))
            {
                // Tracking success : Draw the tracked object
                rectangle(frame, bbox, Scalar( 255, 0, 0 ), 2, 1 );
            }
            else
            {
                // Tracking failure detected.
                putText(frame, "Tracking failure detected", Point(100,80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,255),2);
                // Clear the tracker
                tracker->clear();
                // Change the mode to DETECT
                program_mode = DETECT;
            }
            
            // Display tracker type on frame
            putText(frame, "KCF Tracker", Point(100,20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50),2);
        }

        // Calculate Frames per second (FPS)
        float fps = getTickFrequency() / ((double)getTickCount() - timer);
        // Display FPS on frame
        putText(frame, "FPS : " + SSTR(int(fps)), Point(100,50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50), 2);

        // Display frame.
        imshow("Tracking", frame);

        // Exit if ESC pressed.
        int k = waitKey(1);
        if(k == 27)
        {
            break;
        }
    }
}