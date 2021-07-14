#ifndef CAMERA_H
#define CAMERA_H

#include<string>
#include<iostream>
#include<opencv2/core/persistence.hpp>
#include<opencv2/highgui.hpp>

using namespace std;
class Camera
{
    public:
    Camera(){};
    Camera(const string &strSettingsFile);

    cv::Mat cK;
    cv::Mat cDist_coef;

    private:
};

#endif