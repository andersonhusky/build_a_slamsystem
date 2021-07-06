#ifndef CONSTRUCT_H
#define CONSTRUCT_H

#include<string>
#include<iostream>
#include<opencv2/core/persistence.hpp>
#include<opencv2/highgui.hpp>

#include<torch/torch.h>
#include<torch/script.h>
// #include "Thirdparty/DBoW2/DUtils/Random.h"

using namespace std;
class Construct
{
    public:
        Construct();
        Construct(const string &strSettingsFile);
        void ReconstructFromTwoFrames(
            at::Tensor &kpts0,                                 // 上一帧匹配关键点张量
            at::Tensor &kpts1,                                 // 当前帧匹配关键点张量
            cv::Mat &R21,                                           // 旋转向量
            cv::Mat &t21,                                           // 平移
            vector<cv::Point3f> &cP3D,              // 3d点
            vector<bool> &cTriangulated);       // 三角化成功与否

        cv::Mat cK;
        cv::Mat cDist_coef;

        // ReconstructFromTwoFrames计算R、t
        at::Tensor cKpts0, cKpts1;
        int cMax_iterations;
        vector<vector<size_t>> cSet;
    private:

};

#endif