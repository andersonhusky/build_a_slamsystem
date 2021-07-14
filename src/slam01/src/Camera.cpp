#include"viosystem/Camera.h"

Camera::Camera(const string &strSettingsFile)
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
    // K.at<float>(0, 0) = 520.9;
    // K.at<float>(1, 1) = 521.0;
    // K.at<float>(0, 2) = 325.1;
    // K.at<float>(1, 2) = 249.7;
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
}