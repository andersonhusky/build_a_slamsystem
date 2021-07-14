#ifndef CONSTRUCT_H
#define CONSTRUCT_H

#include<string>
#include<iostream>
#include<opencv2/core/persistence.hpp>
#include<opencv2/highgui.hpp>

#include<torch/torch.h>
#include<torch/script.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include "Thirdparty/DBoW2/DUtils/Random.h"

using namespace std;
class Construct
{
    public:
        Construct();
        Construct(cv::Mat &K, float sigma=1.0);
        bool ReconstructFromTwoFrames(
            vector<cv::Point2f> points0,                                 // 上一帧匹配关键点张量
            vector<cv::Point2f> points1,                                 // 当前帧匹配关键点张量
            cv::Mat &R21,                                           // 旋转向量
            cv::Mat &t21,                                           // 平移
            vector<cv::Point3f> &cP3D,              // 3d点
            vector<bool> &cTriangulated);       // 三角化成功与否
        void CalculateHomography(vector<bool> &Match_H, float &score, cv::Mat &H21);
        void CalculateFundamental(vector<bool> &Match_F, float &score, cv::Mat &F21);

        cv::Mat ComputeH10(const vector<cv::Point2f> &P0, const vector<cv::Point2f> &P1);
        cv::Mat ComputeF10(const vector<cv::Point2f> &P0, const vector<cv::Point2f> &P1);

        float CheckHomography(const cv::Mat &H21, const cv::Mat &H12, vector<bool> &MatchInliers, float sigma);
        float CheckFundamental(const cv::Mat &F21, vector<bool> &MatchInliers, float sigma);

        bool ReconstructFromH(vector<bool> &MatchInliers, cv::Mat &H10,cv::Mat &K, cv::Mat &R10, cv::Mat &t10,
                                                        vector<cv::Point3f> &P3d, vector<bool> &Trangulated, float MinParallax, int MinTriangulated);
        bool ReconstructFromF(vector<bool> &MatchInliers, cv::Mat &F10, cv::Mat &K, cv::Mat &R10, cv::Mat &t10,
                                                        vector<cv::Point3f> &P3d, vector<bool> &Trangulated, float MinParallax, int MinTriangulated);

        void Normalize(const vector<cv::Point2f> points, vector<cv::Point2f> &cNormalizedPoints, cv::Mat &T);
        int CheckRt(const cv::Mat &Ri, const cv::Mat &ti, const vector<cv::Point2f> &points0, const vector<cv::Point2f> &points1, vector<bool> &cMatchInliers,
                                                        const cv::Mat &K, vector<cv::Point3f> &cP3D, float th2, vector<bool> &cGood, float &cParallax);
        void Triangulate(const cv::Point2f &point0, const cv::Point2f &point1, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D);
        void DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t);

        void TestH(const cv::Mat &H10, const cv::Mat &H01, vector<bool> &cMatchInliers, float sigma, vector<cv::Point2f> cPosition0i_, vector<cv::Point2f> cPosition1i_);
        void TestHi(const cv::Mat &H10, const cv::Mat &H01, vector<bool> &cMatchInliers, float sigma, vector<int> idx_vec);

    private:
        cv::Mat cK;
        cv::Mat cDist_coef;

        // ReconstructFromTwoFrames计算R、t
        vector<cv::Point2f> cPoints0, cPoints1;
        at::Tensor cKpts0, cKpts1;
        int cMax_iterations;
        vector<vector<size_t>> cSet;            // 计算R、t所用需要随机采样序号存储与此
        float cSigma, cSigma_2;                                           // 样本标准差和方差

};

#endif