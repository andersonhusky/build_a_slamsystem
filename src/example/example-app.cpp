#include<torch/torch.h>
#include<torch/script.h>
#include <iostream>
#include<vector>
#include<opencv2/highgui.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/opencv.hpp>
#include"viosystem/Construct.h"

using namespace std;
using namespace cv;


void find_feature_match(const cv::Mat &img1, const cv::Mat &img2, at::Tensor &Mkpts0, at::Tensor &Mkpts1)
{
    torch::Device m_device(torch::kCUDA);
    torch::jit::script::Module SuperPoints= torch::jit::load(
        "/home/hongfeng/CV/SuperGluePretrainedNetwork/superpoint_gpu_flow.pt"
    );
    torch::jit::script::Module SuperGlue = torch::jit::load(
        "/home/hongfeng/CV/SuperGluePretrainedNetwork/superglue_gpu.pt"
    );
    SuperPoints.to(m_device);
    SuperPoints.eval();
    SuperGlue.to(m_device);
    SuperGlue.eval();
    cout << "load model successful!" << endl;

    cv::Mat img_1 = img1;
    cv::Mat img_2 = img2;
    cvtColor(img_1,img_1,CV_BGR2GRAY);                          // 160*120
    cvtColor(img_2, img_2, CV_BGR2GRAY);
    const cv::Mat img_gray1 = img_1;
    const cv::Mat img_gray2 = img_2;
    img_1.convertTo(img_1, CV_32FC3, 1.0f/255.0f);
    img_2.convertTo(img_2, CV_32FC3, 1.0f/255.0f);
    const torch::Tensor img_tensor1 = torch::from_blob(img_1.data,{1, 120,160, 1}).permute({0, 3, 1, 2}).cuda();
    const torch::Tensor img_tensor2 = torch::from_blob(img_2.data,{1, 120,160, 1}).permute({0, 3, 1, 2}).cuda();

    auto output1 = SuperPoints.forward({img_tensor1}).toTuple();
    auto output2 = SuperPoints.forward({img_tensor1}).toTuple();
    vector<torch::jit::IValue> inputs;
    inputs.push_back(output1->elements()[0].toTensor().unsqueeze(0));
    inputs.push_back(output1->elements()[1].toTensor().unsqueeze(0));
    inputs.push_back(output1->elements()[2].toTensor().unsqueeze(0));
    inputs.push_back(img_tensor1);
    inputs.push_back(output2->elements()[0].toTensor().unsqueeze(0));
    inputs.push_back(output2->elements()[1].toTensor().unsqueeze(0));
    inputs.push_back(output2->elements()[2].toTensor().unsqueeze(0));
    inputs.push_back(img_tensor2);

    auto output = SuperGlue.forward(inputs).toTuple();
    at::Tensor indices0 = output->elements()[0].toTensor();
    at::Tensor valid0 = (indices0 > -1).toType(at::kBool);                      // 由上一帧特征点找到的有效匹配n个(用掩模)
    at::Tensor valid1 = indices0.index({valid0});                                       // 对应的有效索引序号(用行索引)
    Mkpts0 = inputs[0].toTensor().index({valid0});                                  // (1, 300, 2) 转化为 (n, 2)  有效匹配的坐标
    Mkpts1 = inputs[4].toTensor().squeeze(0).index({valid1});           // (1, 300, 2) 转化为(n, 2)  有效匹配的坐标

    return;
}

int main(int argc, char **argv)
{
    if(argc!=3)
    {
        cout << "usage::pose_estimation_2d2d img1 img2" << endl;
        return 1;
    }
    Mat img1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat img2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);
    at::Tensor Mkpts0;
    at::Tensor Mkpts1;

    find_feature_match(img1, img2, Mkpts0, Mkpts1);

    cout << "找到了" << Mkpts0.size(0) << "组匹配点" << endl;

    cv::Mat R, t;
    vector<cv::Point3f> P3D;
    vector<bool> Triangulated;
    Construct CCC;
    CCC = Construct("/home/hongfeng/CV/build_a_slam_bymyself/src/slam01/D435i.yaml");
    CCC.ReconstructFromTwoFrames(Mkpts0, Mkpts1, R, t, P3D, Triangulated);

    return 0;
}
