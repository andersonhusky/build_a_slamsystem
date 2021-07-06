#ifndef FRAME_H
#define FRAME_H

#include<ros/ros.h>
#include<iostream>                                                                        // vector
#include<chrono>                                                                            // chrono

#include<viosystem/Mappoint.h>

#include<opencv2/core/mat.hpp>                                          // Mat
#include<opencv2/features2d.hpp>                                        // FAST
#include<cv_bridge/cv_bridge.h>                                            // 
#include<opencv2/highgui.hpp>
#include<torch/torch.h>
#include<torch/script.h>

using namespace std;
class Frame
{
    public:
        Frame();
        Frame(const cv::Mat &im,
                        int &nNextId,
                        const double &timestamp);                                            // 构造函数
        Frame(const torch::Tensor &im_tensor,
                        const cv::Mat &im,
                        int &nNextId,
                        const double &timestamp,
                        bool net);

        void FeatureDetect(const cv::Mat& im);
        void AddPoints(at::Tensor feature_tensor, at::Tensor score_tensor, at::Tensor descriptors_tensor,
                                            int time_usd, int n);
        void AddMatching(at::Tensor &kpts0, at::Tensor &kpts1, int time_usd, int n);

        void Show();

        bool mnet;                                                                  // 神经网络/传统方法标志位
        long unsigned int mindex;
        double mtimestamp;
        int N_points, N_matching;                                                                              // 特征点数目
        vector<MapPoint*> mmappoints;                   // 构建的地图点

        int mtime_findusd, mtime_matchusd;

        vector<cv::KeyPoint> mkeypoints;
        cv::Mat mdescriptor;
        cv::Mat mimg_gray;

        at::Tensor mkeypoints_tensor;                       // 关键点tensor
        at::Tensor mscores_tensor;                              // 关键点得分tensor
        at::Tensor mdescriptors_tensor;                     // 关键点描述子tensor
        at::Tensor mimg_tensor;                                    // 图像tensor
        at::Tensor mkpts0, mkpts1;                              // 上一帧，当前帧匹配特征点的坐标

    private:

};

#endif