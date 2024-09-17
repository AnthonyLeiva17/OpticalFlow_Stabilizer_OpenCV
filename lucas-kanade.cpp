#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <chrono>
using namespace cv;
using namespace std;

int lucas_kanade(const string& filename, bool save)
{
    VideoCapture capture(filename);
    if (!capture.isOpened()){
        cerr << "Unable to open file!" << endl;
        return 0;
    }
    
    vector<Scalar> colors;
    RNG rng;
    for(int i = 0; i < 100; i++)
    {
        int r = rng.uniform(0, 256);
        int g = rng.uniform(0, 256);
        int b = rng.uniform(0, 256);
        colors.push_back(Scalar(r,g,b));
    }
    int number_of_points = 50;
    auto start = std::chrono::high_resolution_clock::now();

    Mat old_frame, old_gray;
    vector<Point2f> p0, p1;

    capture >> old_frame;
    if (old_frame.empty()) {
        cerr << "Error: Unable to read the first frame." << endl;
        return 0;
    }

    cvtColor(old_frame, old_gray, COLOR_BGR2GRAY);
    goodFeaturesToTrack(old_gray, p0, number_of_points, 0.3, 7, Mat(), 7, false, 0.04);
    cout << "Number of points to track: " << p0.size() << endl;

    Mat mask = Mat::zeros(old_frame.size(), CV_8UC3);
    int counter = 0;

    while(true) {
        Mat frame, frame_gray;
        capture >> frame;
        if (frame.empty())
            break;
        cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

        vector<uchar> status;
        vector<float> err;
        TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);

        if (p0.empty()) {
            cerr << "Error: No points to track." << endl;
            break;
        }

        calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p1, status, err, Size(15,15), 2, criteria);

        vector<Point2f> good_new;
        for(size_t i = 0; i < p0.size(); i++)
        {
            if(status[i] == 1) {
                good_new.push_back(p1[i]);
                line(mask, p1[i], p0[i], colors[i % colors.size()], 2);
                circle(frame, p1[i], 5, colors[i % colors.size()], -1);
            }
        }
        //cout << "Number of good points: " << good_new.size() << endl;
        if (good_new.empty()) {
            cerr << "Warning: No good points to track. Re-detecting features." << endl;
            goodFeaturesToTrack(frame_gray, p0, number_of_points, 0.01, 7, Mat(), 7, false, 0.04);
            continue;
        }

       

        Mat img;
        add(frame, mask, img);

        if (save) {
            string save_path = "./optical_flow_frames/frame_" + to_string(counter) + ".jpg";
            imwrite(save_path, img);
        }

        imshow("flow", img);
        int keyboard = waitKey(25);
        if (keyboard == 'q' || keyboard == 27)
            break;

        old_gray = frame_gray.clone();
        p0 = good_new;
        counter++;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "sparse_lk " << ": " << duration.count() << " segundos" << std::endl;
    return 0;
}
