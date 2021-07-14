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
class Construct2
{
    public:
        Construct2();
        Construct2(const string &strSettingsFile, float sigma=1.0);
    
        bool Reconstruct(const at::Tensor &cKpts1, const at::Tensor &cKpts2,
                    cv::Mat &R21, cv::Mat &t21, std::vector<cv::Point3f> &vP3D, std::vector<bool> &vbTriangulated);

    private:
        void FindFundamental(std::vector<bool> &vbInliers, float &score, cv::Mat &F21);
        void Normalize(const at::Tensor kpts, std::vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T);
        cv::Mat ComputeF21(const std::vector<cv::Point2f> &vP1, const std::vector<cv::Point2f> &vP2);
        float CheckFundamental(const cv::Mat &F21, std::vector<bool> &vbMatchesInliers, float sigma);
        bool ReconstructF(std::vector<bool> &vbMatchesInliers, cv::Mat &F21, cv::Mat &K,
                      cv::Mat &R21, cv::Mat &t21, std::vector<cv::Point3f> &vP3D, std::vector<bool> &vbTriangulated, float minParallax, int minTriangulated);
        void DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t);
        int CheckRT(const cv::Mat &R, const cv::Mat &t, const at::Tensor Kpts1, const at::Tensor Kpts2,
                        std::vector<bool> &vbInliers,
                       const cv::Mat &K, std::vector<cv::Point3f> &vP3D, float th2, std::vector<bool> &vbGood, float &parallax);
        void Triangulate(const at::Tensor Kpt1, const at::Tensor Kpt2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D);

        cv::Mat mK;
        cv::Mat cDist_coef;

        // ReconstructFromTwoFrames计算R、t
        at::Tensor cKpts1, cKpts2;
        int mMaxIterations;
        std::vector<std::vector<size_t> > mvSets;            // 计算R、t所用需要随机采样序号存储与此
        float mSigma, mSigma2;                                           // 样本标准差和方差
};

#endif