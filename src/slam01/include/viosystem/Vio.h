#ifndef VIO_H
#define VIO_H

#include "Frame.h"
#include "Construct.h"
#include "Camera.h"
#include <c10/cuda/CUDACachingAllocator.h>

class Vio
{
    public:
        // torch::NoGradGuard no_grad;
        Vio(bool net):start(false), frame_index(0), Mnet(net), State(NO_IMAGES_YET){
            if(Mnet)
            {
                cout << "\nloading superpoint" << endl;
                torch::Device m_device(torch::kCUDA);
                SuperPoints = torch::jit::load(superpoint_path);
                cout << "load superpoint successful!" << endl;
                SuperPoints.to(m_device);
                SuperPoints.eval();
                c10::cuda::CUDACachingAllocator::emptyCache();

                cout << "\nloading superglue" << endl;
                SuperGlue = torch::jit::load(superglue_path);
                cout << "load superpoint successful!" << endl;
                SuperGlue.to(m_device);
                SuperGlue.eval();
                c10::cuda::CUDACachingAllocator::emptyCache();
            }
            CameraModel = Camera(camera_path);
            MoveConstruct = Construct(CameraModel.cK);
        };

        void FratureMatch(const cv::Mat &im, const double &timestamp);
        void Track(const torch::Tensor &im_tensor, const cv::Mat &im, const double &timestamp);
        void SuperPointFind(const torch::Tensor &im_tensor, const cv::Mat &im, const double &timestamp);                  // 找到superpoint点
        void SuperGlueMatching();           // 用supergule进行匹配
        void Initialization();

        enum eTrackingState{
        SYSTEM_NOT_READY=-1,
        NO_IMAGES_YET=0,
        NOT_INITIALIZED=1,
        OK=2,
        LOST=3
        };

        Camera CameraModel;
        eTrackingState State;
        bool start;                                                                 // 帧起点标志位
        int frame_index;                                                    // 关键帧信息

        torch::jit::script::Module SuperPoints;
        vector<torch::jit::IValue> inputs;                  // 实时保存supergulue的输入
        torch::jit::script::Module SuperGlue;
        int Nmatch;
        at::Tensor Mkpts0, Mkpts1;

        cv::Mat VioImgGray;                         // 传统模式时的存放img的cv::Mat
        torch::Tensor VioImgTensor;           // 使用神经网络时存放img的tensor

        Frame NowFrame;
        Frame LastFrame;
        Construct MoveConstruct;
        vector<cv::Point3f> IniP3D;       // 从匹配中恢复的3d点坐标

        bool Mnet;                                               // 确定传统模式还是神经网络模式
        int count=0;                                        // 调试用

    private:
        string superpoint_path = "/home/hongfeng/CV/SuperGluePretrainedNetwork/superpoint_gpu_flow.pt";
        string superglue_path = "/home/hongfeng/CV/SuperGluePretrainedNetwork/superglue_gpu.pt";
        string camera_path = "/home/hongfeng/CV/build_a_slam_bymyself/src/slam01/D435i.yaml";
};

#endif