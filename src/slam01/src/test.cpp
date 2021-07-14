#include<torch/torch.h>
#include<torch/script.h>
#include <iostream>
#include<vector>
#include<opencv2/highgui.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/opencv.hpp>
#include"viosystem/Construct.h"
#include"viosystem/Camera.h"

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
    cv::Mat img_gray1 = img_1;
    cv::Mat img_gray2 = img_2;
    img_1.convertTo(img_1, CV_32FC3, 1.0f/255.0f);
    img_2.convertTo(img_2, CV_32FC3, 1.0f/255.0f);
    const torch::Tensor img_tensor1 = torch::from_blob(img_1.data,{1, 480, 640, 1}).permute({0, 3, 1, 2}).cuda();
    const torch::Tensor img_tensor2 = torch::from_blob(img_2.data,{1, 480, 640, 1}).permute({0, 3, 1, 2}).cuda();

    auto output1 = SuperPoints.forward({img_tensor1}).toTuple();
    auto output2 = SuperPoints.forward({img_tensor2}).toTuple();
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

    for(int i=0; i<Mkpts0.size(0); i++)
        {
            // to do..  在这里把提取的特征点直接存储下来
            cv::Point point_draw(Mkpts0[i][0].item().toInt(), Mkpts0[i][1].item().toInt());
            cv::circle(img_gray1, point_draw, 2, cv::Scalar(255, 0, 0), -1);
        }
    cv::imshow("图片", img_gray1);
    cv::waitKey(0);

    for(int i=0; i<Mkpts0.size(0); i++)
        {
            // to do..  在这里把匹配结果直接存储下来
            cv::Point point_now(Mkpts0[i][0].item().toInt(), Mkpts0[i][1].item().toInt());
            cv::Point point_last(Mkpts1[i][0].item().toInt(), Mkpts1[i][1].item().toInt());
            cv::circle(img_gray2, point_now, 2, cv::Scalar(0, 255, 0), -1);
            cv::line(img_gray2, point_last, point_now, cv::Scalar(255, 0, 0), 2);
        }
    cv::imshow("图片", img_gray2);
    cv::waitKey(0);

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
    vector<cv::Point2f> points0;
    vector<cv::Point2f> points1;
    points0.reserve(Mkpts0.size(0));
    points1.reserve(Mkpts1.size(0));
    for(int i=0; i<Mkpts0.size(0); i++)
    {
        cv::Point2f point;
        point.x = Mkpts0[i][0].item().toFloat();
        point.y = Mkpts0[i][1].item().toFloat();
        points0.push_back(point);
    }
    for(int i=0; i<Mkpts1.size(0); i++)
    {
        cv::Point2f point;
        point.x = Mkpts1[i][0].item().toFloat();
        point.y = Mkpts1[i][1].item().toFloat();
        points1.push_back(point);
    }
    for(vector<cv::Point2f>::iterator itr = points0.begin(); itr!=points0.end(); itr++)
    {
        cout << 1 << endl;
    }

    cout << "找到了" << Mkpts0.size(0) << "组匹配点" << endl;

    cv::Mat R, t;
    vector<cv::Point3f> P3D;
    vector<bool> Triangulated;
    Camera CameraModel;
    Construct CCC;
    CameraModel = Camera("/home/hongfeng/CV/build_a_slam_bymyself/src/slam01/D435i.yaml");
    CCC = Construct(CameraModel.cK);
    CCC.ReconstructFromTwoFrames(points0, points1, R, t, P3D, Triangulated);

    return 0;
}