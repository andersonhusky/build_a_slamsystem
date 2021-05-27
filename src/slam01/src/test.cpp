#include<iostream>
#include<vector>

#include<opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>                                        // cvtColor
#include <opencv2/core/types.hpp>                                   // KetPoint
#include<opencv2/features2d.hpp>                                    // FAST
#include <opencv2/core/mat.hpp>                                     // InputArray, OutputArray

using namespace std;
using namespace cv;

void feature_detect_show(const Mat img)
{
    Mat img_gray = img;
    Mat img_output = img;
    if(img_gray.channels() == 3)
    {
        cvtColor(img_gray, img_gray, CV_RGB2GRAY);            // 输出图像，输入图像，转换码
    }
    imwrite("GRAY.jpg", img_gray);

    vector<cv::KeyPoint> keypoints;
    Mat mDescriptors;
    FAST(img_gray, keypoints, 20);
    cout << !keypoints.empty() << endl;
    if(!keypoints.empty())
    {
        drawKeypoints(img_gray, keypoints, img_output);
        imwrite("result1.jpg", img_output);
    }
    imwrite("result2.jpg", img_output);


    // imshow("灰度图片", im_gray);
}

int main()
{
    Mat img = imread("test.jpg", CV_LOAD_IMAGE_COLOR);
    feature_detect_show(img);
}