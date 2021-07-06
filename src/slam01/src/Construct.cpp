#include"viosystem/Construct.h"

Construct::Construct(){}

Construct::Construct(const string &strSettingsFile)
{
    cv::FileStorage cSettings(strSettingsFile, cv::FileStorage::READ);
    if(!cSettings.isOpened())
    {
        cerr << "Failed to open settings file at: " << strSettingsFile << endl;
        exit(-1);
    }

    cv::Mat dist_coef = cv::Mat::zeros(4, 1, CV_32F);

    auto  ccamera_name = cSettings["Camera.type"];
    if(ccamera_name == "PinHole")
    {
        dist_coef.at<float>(0) = cSettings["Camera.k1"];
        dist_coef.at<float>(1) = cSettings["Camera.k2"];
        dist_coef.at<float>(2) = cSettings["Camera.p1"];
        dist_coef.at<float>(3) = cSettings["Camera.p2"];
    }
    float fx =cSettings["Camera.fx"];
    float fy = cSettings["Camera.fy"];
    float cx = cSettings["Camera.cx"];
    float cy = cSettings["Camera.cy"];
    // 构造内参矩阵K
    cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
    K.at<float>(0, 0) = fx;
    K.at<float>(1, 1) = fy;
    K.at<float>(0, 2) = cx;
    K.at<float>(1, 2) = cy;
    K.copyTo(cK);
    // 构造几遍参数矩阵dist_coef
    float k3 = cSettings["Camera.k3"];
    if(k3!=0)
    {
        dist_coef.resize(5);
        dist_coef.at<float>(4) = k3;
    }
    dist_coef.copyTo(cDist_coef);

    float fps = cSettings["Camera.fps"];
    if(fps==0)  fps=30;

    cout << endl << "Camera Parameters: " << endl;
    cout << "-fx: " << fx << endl;
    cout << "-fy: " << fy << endl;
    cout << "-cx: " << cx << endl;
    cout << "-cy: " << cy << endl;
    cout << "-k1: " << dist_coef.at<float>(0) << endl;
    cout << "-k2: " << dist_coef.at<float>(1) << endl;
    cout << "-p1: " << dist_coef.at<float>(2) << endl;
    cout << "-p2" << dist_coef.at<float>(3) << endl;
    if(dist_coef.rows==5)
        cout << "-k3: " << dist_coef.at<float>(4) << endl;
    cout << "fps: " << fps << endl;
    cout << "K: " << cK << endl;

    cMax_iterations = 30;
}

void Construct::ReconstructFromTwoFrames(at::Tensor &kpts0, at::Tensor &kpts1, cv::Mat &R21, cv::Mat &t21,
                                                                                            vector<cv::Point3f> &cP3D, vector<bool> &cTriangulated)
{
    cKpts0 = kpts0;
    cKpts1 = kpts1;
    if(kpts1.size(0)==0)    return;

    const int N = kpts1.size(0);
    vector<size_t> cIndices_all;                    // 所有的序号
    cIndices_all.reserve(N);
    vector<size_t> cIndices_avaliable;               // 最后使用的关键点序号
    // 随机取cMax_iterations组8个的数据用于计算R，t
    cSet = vector<vector<size_t>>(cMax_iterations, vector<size_t>(8, 0));
    // DUtils::Random::SeedRandOnce(0);    // 设置随机数种子
    // for(int it=0; it<cMax_iterations; it++)
    // {
    //     cIndices_avaliable = cIndices_all;
    //     for(size_t j=0; j<8; j++)
    //     {
    //         int randnum = DUtils::Random::RandomInt(0, N-1);
    //         int idx = cIndices_avaliable[randnum];
    //         cSet[it][j] = idx;
    //         cIndices_avaliable[randnum] = cIndices_avaliable.back();
    //         cIndices_avaliable.pop_back();
    //     }
    // }
    // cout << cIndices_avaliable[0] << endl;
}