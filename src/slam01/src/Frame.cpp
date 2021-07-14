#include "viosystem/Frame.h"

Frame::Frame()
{

}

Frame::Frame(const cv::Mat &im, int &nNextId, const double &timestamp)
{
    mindex = nNextId;
    mtimestamp = timestamp;
    FeatureDetect(im);
}

Frame::Frame(const torch::Tensor &im_tensor,
                                const cv::Mat &im,
                                int &nNextId,
                                const double &timestamp,
                                bool net):
                                mimg_tensor(im_tensor), mimg_gray(im),
                                mindex(nNextId++), mtimestamp(timestamp), mnet(net)
{
}

// 将tensor恢复成图片
cv::Mat Tensor2Mat(torch::Tensor &input_tensor)
{
    torch::Tensor input_tensor_trans = input_tensor;
    // height取到的是二维矩阵的行数，width取到的是二维矩阵的列数
    input_tensor_trans = input_tensor_trans.squeeze();
    input_tensor_trans = input_tensor_trans.to(torch::kCPU);
    int height = input_tensor_trans.size(0);
    int width = input_tensor_trans.size(1);
    // mat创建时的cv::Size(长对应列数，宽对应行数)
    cv::Mat output_mat(cv::Size(width, height), CV_32F, input_tensor_trans.data_ptr());
    return output_mat;
}

void Frame::FeatureDetect(const cv::Mat &im)
{
    mimg_gray = im;

    vector<cv::KeyPoint> keypoints;
    cv::Mat Descriptors;
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    FAST(mimg_gray, keypoints, 10);
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    mtime_findusd = (t1-t0).count();
    N_points = keypoints.size();
    mkeypoints = keypoints;
}

void Frame::AddPoints(at::Tensor feature_tensor, at::Tensor score_tensor, at::Tensor descriptors_tensor,
                                                int time_usd, int n)
{
    mkeypoints_tensor = feature_tensor;
    mscores_tensor = score_tensor;
    mdescriptors_tensor = descriptors_tensor;
    mtime_findusd = time_usd;   N_points = n;
    for(int i=0; i<mkeypoints_tensor.size(1); i++)
        {
            // to do..  在这里把提取的特征点直接存储下来
            cv::Point point_draw(mkeypoints_tensor[0][i][0].item().toInt(), mkeypoints_tensor[0][i][1].item().toInt());
            cv::circle(mimg_gray, point_draw, 2, cv::Scalar(255, 0, 0), -1);
        }
}

// 添加匹配的结果
void Frame::AddMatching(at::Tensor &kpts0, at::Tensor &kpts1, int time_usd, int n)
{
    mkpts0 = kpts0;
    mkpts1 = kpts1;
    mtime_matchusd = time_usd, N_matching = n;
    for(int i=0; i<mkpts0.size(0); i++)
        {
            // to do..  在这里把匹配结果直接存储下来
            cv::Point point_now(mkpts0[i][0].item().toInt(), mkpts0[i][1].item().toInt());
            cv::Point point_last(mkpts1[i][0].item().toInt(), mkpts1[i][1].item().toInt());
            cv::circle(mimg_gray, point_now, 2, cv::Scalar(0, 255, 0), -1);
            cv::line(mimg_gray, point_last, point_now, cv::Scalar(0, 0, 255), 2);
        }
}

void Frame::Show()
{
    cout << "frame index:" << mindex << endl;
    cout << "frame timestamp:" << mtimestamp << endl;
    cout << "number of keypoint:" << N_points << endl;
    cout << "number of matches:" << N_matching << endl;
    cout << "time used:" << mtime_findusd << endl;
    cout << "time used:" << mtime_matchusd << "\n" << endl;
    cv::imshow("图片", mimg_gray);
    cv::waitKey(1);
}