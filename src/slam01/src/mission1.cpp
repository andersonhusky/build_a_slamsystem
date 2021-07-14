/**
 * 任务1：实现一个通过ros读取照片，提取完特征点最终显示的系统
 * start：20/5/26
 * end:20/5/27
 * end：...
*/

#include<iostream>
#include<vector>
#include<chrono>                                                                        // chrono
#include<torch/torch.h>
#include<torch/script.h>

#include<ros/ros.h>
#include<sensor_msgs/Image.h>
#include<cv_bridge/cv_bridge.h>
#include<opencv2/highgui.hpp>

#include<opencv2/core/types.hpp>                                   // KetPoint
#include<opencv2/features2d.hpp>                                    // FAST
#include<opencv2/core/mat.hpp>                                     // InputArray, OutputArray

#include"viosystem/Vio.h"
#include"viosystem/Frame.h"
#include"viosystem/Camera.h"

using namespace std;

class ProcessImg
{
    public:
        ProcessImg(Vio* PVio, bool net):mPVio(PVio), mnet(net){}
        void GetImage(const sensor_msgs::ImageConstPtr& msg);
        Vio* mPVio;
        bool mnet;
        bool mbRGB=false;                                           // true为RGB，false为BGR
        int count = 0;
    private:
};

void ProcessImg::GetImage(const sensor_msgs::ImageConstPtr& msg)
{
    // cv_ptr->header.stamp.toSec() 时间戳
    cv_bridge::CvImageConstPtr cv_ptr;          // 用于ros图像和opencv图像的转换
    try
    {
        cv_ptr = cv_bridge::toCvShare(msg);
    }
    catch(const std::exception& e)
    {
        ROS_ERROR("cv_bridge exception:%s", e.what());
        return;
    }
    
    // img作为处理的mat，处理过程中的两个结果存为const送给函数
    cv::Mat img = cv_ptr->image;
    if(img.channels() == 3)
    {
        // 输出图像，输入图像，转换码
        if(mbRGB)
            cvtColor(img,img,CV_RGB2GRAY);
        else
            cvtColor(img,img,CV_BGR2GRAY);
    }
    else if(img.channels()==4)
    {
        if(mbRGB)
            cvtColor(img,img,CV_RGBA2GRAY);
        else
            cvtColor(img,img,CV_BGRA2GRAY);
    }

    // 模式切换
    if(mnet)
    {
        const cv::Mat img_gray = img;                       // 灰度图
        if(count==0||count == 5 || count==10)
        {
            string title = to_string(count);
            cv::imwrite(title+".png", img);
        }
        count++;
        img.convertTo(img, CV_32FC3, 1.0f/255.0f);
        // tensor,(1,1,480,640),使用gpu
        const torch::Tensor img_tensor = torch::from_blob(img.data,{1, 480,640, 1}).permute({0, 3, 1, 2}).cuda();
        mPVio->Track(img_tensor, img_gray, cv_ptr->header.stamp.toSec());
    }
    else
    {
        // to do.. 传统模式下的
    }
}

int main(int argc, char **argv)
{
    // torch::autograd::AutoGradMode guard(false_type);
    bool net = true;                                                    // 是否调用神经网络
    ros::init(argc, argv, "hongfeng");                  // 节点初始化
    ros::start();                                                             // 节点启动

    if(argc != 3)
    {
        cerr << endl << "usage: need ...";
        // ros::shutdown();
        // return 1;
    }

    ros::NodeHandle nodeHandler;                  // 创建节点句柄
    // 创建当前vio进程
    Vio working(net);

    ProcessImg pimg(&working, net);
    // 创建一个subscriber，
    //订阅名为/camera/color/image_raw的话题，处理队列大小为1， 注册回调函数以及所其所属的类
    // ros::Subscriber sub = nodeHandler.subscribe("/camera/color/image_raw", 1, &ProcessImg::GetImage, &pimg);
    ros::Subscriber sub = nodeHandler.subscribe("/color", 1, &ProcessImg::GetImage, &pimg);

    ros::spin();                                                            // 循环等待回调函数
    ros::shutdown();
    cv:cvDestroyWindow("view");
    return 0;
}