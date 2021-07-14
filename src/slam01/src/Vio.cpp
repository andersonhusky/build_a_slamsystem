#include "viosystem/Vio.h"

vector<cv::KeyPoint> Tensor2Vector(torch::Tensor &input_tensor)
{
    input_tensor = input_tensor.to(torch::kCPU);
    vector<cv::KeyPoint> output_vec;
}

/*用传统方法提取特征点*/
void Vio::FratureMatch(const cv::Mat &im, const double &timestamp)
{
    VioImgGray = im;

    if(start==false)            // 拿到第一张图片
    {
        NowFrame = Frame(VioImgGray, frame_index, timestamp);
        start = true;
        NowFrame.Show();
        return;
    }
    else                                // 拿到最新的图片呢，和上一张进行匹配
    {
        cout << 1 << endl;
        frame_index++;
        LastFrame = NowFrame;
        NowFrame = Frame(VioImgGray, frame_index, timestamp);
        NowFrame.Show();
    }
}

/*用神经网络方法提取特征点*/
void Vio::Track(const torch::Tensor &im_tensor, const cv::Mat &im, const double &timestamp)
{
    VioImgTensor = im_tensor;
    VioImgGray = im;
    
    if(State==NO_IMAGES_YET)    State = NOT_INITIALIZED;

    if(State==NOT_INITIALIZED)
    {
        // Initialization();
        NowFrame = Frame(im_tensor, im, frame_index, timestamp, Mnet);
        State = OK;
        SuperPointFind(im_tensor, im, timestamp);
        NowFrame.Show();
        return;
    }

    if(State==OK)
    {
        LastFrame = NowFrame;
        NowFrame = Frame(im_tensor, im, frame_index, timestamp, Mnet);
        SuperPointFind(im_tensor, im, timestamp);
        SuperGlueMatching();
    }

    // 提取和匹配结束开始计算位姿，旋转、位移、通过三角化的匹配
    cv::Mat Rcw;
    cv::Mat tcw;
    vector<bool> Triangulated;

    vector<cv::Point2f> Points0;
    vector<cv::Point2f> Points1;
    Points0.reserve(NowFrame.mkpts0.size(0));
    Points1.reserve(NowFrame.mkpts1.size(0));
    for(int i=0; i<Mkpts0.size(0); i++)
    {
        cv::Point2f point;
        point.x = Mkpts0[i][0].item().toFloat();
        point.y = Mkpts0[i][1].item().toFloat();
        Points0.push_back(point);
    }
    for(int i=0; i<Mkpts1.size(0); i++)
    {
        cv::Point2f point;
        point.x = Mkpts1[i][0].item().toFloat();
        point.y = Mkpts1[i][1].item().toFloat();
        Points1.push_back(point);
    }

    MoveConstruct.ReconstructFromTwoFrames(Points0, Points1, Rcw, tcw, IniP3D, Triangulated);
    NowFrame.Show();
}

/*****************************************************************
 *     函数功能：寻找SuperPoint特征点
 *     函数参数介绍：im_tensor：输入图片张量格式；im：输入图片mat形式；timestamp：图片时间戳
 *     备注：运行之后
 *      inputs：存放上一时刻和当前时刻计算出来的特征张量、得分张量、描述子张量和图片张量，用作SuperGlue的输入
 *      格式为（0~3）上一时刻，（4~7）当前时刻
 *      NowFrame：特征张量、得分张量、描述子张量，花费时间，特征点个数
 ******************************************************************/
void Vio::SuperPointFind(const torch::Tensor &im_tensor, const cv::Mat &im, const double &timestamp)
{
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    auto output = SuperPoints.forward({im_tensor}).toTuple();
    c10::cuda::CUDACachingAllocator::emptyCache();
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    // 特征、得分、描述子
    at::Tensor feature_tensor = output->elements()[0].toTensor().unsqueeze(0);               // (1, n, 2)
    at::Tensor score_tensor = output->elements()[1].toTensor().unsqueeze(0);                 // (1, n, 1)
    at::Tensor descriptors_tensor = output->elements()[2].toTensor().unsqueeze(0);       // （1， 256， n）
    if(frame_index>2)
        inputs.erase(inputs.begin(), inputs.begin()+4);
    inputs.push_back(feature_tensor);
    inputs.push_back(score_tensor);
    inputs.push_back(descriptors_tensor);
    inputs.push_back(im_tensor);

    int time_usd = (t1-t0).count();
    int n = feature_tensor.size(1);
    NowFrame.AddPoints(feature_tensor, score_tensor, descriptors_tensor, time_usd, n);
    c10::cuda::CUDACachingAllocator::emptyCache();
}

// 利用SuperGlue进行匹配
void Vio::SuperGlueMatching()
{
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    auto output = SuperGlue.forward(inputs).toTuple();
    c10::cuda::CUDACachingAllocator::emptyCache();
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    at::Tensor indices0 = output->elements()[0].toTensor();
    // at::Tensor indices1 = output->elements()[1].toTensor().to(torch::kCPU);
    // at::Tensor mscores0 = output->elements()[2].toTensor().to(torch::kCPU);
    // at::Tensor mscores1 = output->elements()[2].toTensor().to(torch::kCPU);
    c10::cuda::CUDACachingAllocator::emptyCache();
    // superglue源码转过来
    at::Tensor valid0 = (indices0 > -1).toType(at::kBool);                      // 由上一帧特征点找到的有效匹配n个(用掩模)
    at::Tensor valid1 = indices0.index({valid0});                                       // 对应的有效索引序号(用行索引)
    Mkpts0 = inputs[0].toTensor().index({valid0});                                  // (1, 300, 2) 转化为 (n, 2)  有效匹配的坐标
    Mkpts1 = inputs[4].toTensor().squeeze(0).index({valid1});           // (1, 300, 2) 转化为(n, 2)  有效匹配的坐标
    int time_usd = (t1-t0).count();
    int n = Mkpts1.size(0);
    NowFrame.AddMatching(Mkpts0, Mkpts1, time_usd, n);
    c10::cuda::CUDACachingAllocator::emptyCache();
}

void Vio::Initialization()
{
}