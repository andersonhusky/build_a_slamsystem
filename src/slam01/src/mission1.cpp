/**
 * 任务1：实现一个通过ros读取照片，提取完特征点最终显示的系统
 * start：20/5/26
 * end:20/5/27
 * end：...
*/

#include<iostream>
#include<vector>
#include<chrono>                                                                        // chrono

#include<ros/ros.h>
#include<sensor_msgs/Image.h>
#include<cv_bridge/cv_bridge.h>
#include<opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/core/types.hpp>                                   // KetPoint
#include<opencv2/features2d.hpp>                                    // FAST
#include <opencv2/core/mat.hpp>                                     // InputArray, OutputArray

using namespace std;
using namespace cv;

// int orbextractor(InputArray _img, vector<KeyPoint>& _keypoint, OutputArray _mdescriptor)
// {
//     return 0;
// }

void feature_detect_show(const Mat img)
{
    Mat img_gray = img;
    Mat img_output = img;
    if(img_gray.channels() == 3)
    {
        cvtColor(img_gray, img_gray, CV_BGR2GRAY);            // 输出图像，输入图像，转换码
    }

    vector<cv::KeyPoint> keypoints;
    Mat mDescriptors;
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    FAST(img_gray, keypoints, 20);
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    cout << "spend time " << (t0-t1).count() << " seconds" << endl;
    if(!keypoints.empty())
    {
        drawKeypoints(img_gray, keypoints, img_output);
    }

    cv::imshow("图片", img_output);
    cv::waitKey(1);
}

void GetImage(const sensor_msgs::ImageConstPtr& msg)
{
    cv_bridge::CvImageConstPtr cv_ptr;          // 用于ros图像和opencv图像的转换
    try
    {
        cv_ptr = cv_bridge::toCvShare(msg);
    }
    catch(const std::exception& e)
    {
        // std::cerr << e.what() << '\n';
        ROS_ERROR("cv_bridge exception:%s", e.what());
        return;
    }
    feature_detect_show(cv_ptr->image);
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "hongfeng");                  // 节点初始化
    ros::start();                                                             // 节点启动

    if(argc != 3)
    {
        cerr << endl << "usage: need ...";
    }

    ros::NodeHandle nodeHandler;                  // 创建节点句柄
    // 创建一个subscriber，
    //订阅名为/camera/color/image_raw的话题，处理队列大小为1， 注册回调函数以及所其所属的类,
    ros::Subscriber sub = nodeHandler.subscribe("/camera/color/image_raw", 1, &GetImage);

    ros::spin();                                                            // 循环等待回调函数
    cv:cvDestroyWindow("view");
    return 0;
}