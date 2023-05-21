/**
 * @file fisherface.cpp
 * @author Shuran Xu (shxu6388@colorado.edu)
 * @brief This program demonstrates the use of the fisherface method by
 * training the model with training images and testing the model with
 * the input test images. 
 * @ref The implementation is based on the Face Recognition tutorial
 * offered by OpenCV at the following link:
 * https://docs.opencv.org/4.1.1/da/d60/tutorial_face_main.html
 * @version 0.1
 * @date 2022-07-10
 * 
 * @copyright Copyright (c) 2022
 */

#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"


#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace cv::face;
using namespace std;


static Mat norm_0_255(InputArray _src) {
    Mat src = _src.getMat();
    // Create and return normalized image:
    Mat dst;
    switch(src.channels()) {
    case 1:
        cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
        break;
    case 3:
        cv::normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
        break;
    default:
        src.copyTo(dst);
        break;
    }
    return dst;
}


static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ',') {
    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(Error::StsBadArg, error_message);
    }
    string line, path, classlabel;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            images.push_back(imread(path, 0));
            labels.push_back(atoi(classlabel.c_str()));
        }
    }
}

const char* params
    = "{ help h         |                         | Print usage }"
      "{ train          |    ../yalefaces.csv     | Path to the training .csv file }"
      "{ test           |    ../testfaces.csv     | Path to the testing .csv file }"
      "{ save           |                         | Path for saving the trained model }";


int main(int argc, const char *argv[]) 
{
    CommandLineParser parser(argc, argv, params);
    parser.about( "This program demonstrates the fisherface model provided by "
                  " OpenCV.\n Users can pass in .csv files for both training and testing purposes.\n" );
    if (parser.has("help"))
    {
        //print help information
        parser.printMessage();
        exit(0);
    }

    // A cv::VideoCapture object is used to read the input video if provided
    String fn_csv = parser.get<String>("train");
    
    if (fn_csv.empty()){
        cerr << "No .csv file specified" << endl;
        return 1;
    }

    // These vectors hold the images and corresponding labels.
    vector<Mat> images;
    vector<int> labels;
    // Read in the data. This can fail if no valid
    // input filename is given.
    try {
        read_csv(fn_csv, images, labels);
    } catch (const cv::Exception& e) {
        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
        // nothing more we can do
        exit(1);
    }

    // Quit if there are not enough images for this demo.
    if(images.size() <= 1) {
        string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
        CV_Error(Error::StsError, error_message);
    }

    // Get the height from the first image. We'll need this
    // later in code to reshape the images to their original
    // size:
    int height = images[0].rows;
    // Create the recognizer
    Ptr<FisherFaceRecognizer> model = FisherFaceRecognizer::create();

    // Read in the data. 
    String test_fn = parser.get<String>("test");
    vector<Mat> testImages;
    vector<int> testLabels;

    try {
        read_csv(test_fn, testImages, testLabels);
    } catch (const cv::Exception& e) {
        cerr << "Error opening file \"" << test_fn << "\". Reason: " << e.msg << endl;
        // nothing more we can do
        exit(1);
    }

    // Train the recognizer
    model->train(images, labels);
    int successCnt = 0;

    // Test the recognizer using the whole test image set
    for(int i = 0; i < testImages.size(); i++)
    {
        Mat testImage = testImages[i];
        int testLabel = testLabels[i];

        int predictedLabel = model->predict(testImage);
        if(predictedLabel == testLabel){
            successCnt++;
            // imshow("Correct Image", testImage);
            // waitKey();
        }
    }

    string result_message = format("Successful Predicted Images = %d / Total Test Images = %u.", \
    successCnt, (unsigned int)testLabels.size());
    cout << result_message << endl;

    // Save the trained model
    String save_path = parser.get<String>("save");
    if(!save_path.empty()){
        model->write(save_path);
    }
    
    // Display if we are not writing to an output folder:
    if(argc == 2) {
        waitKey(0);
    }

    return 0;
}