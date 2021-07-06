#include<torch/torch.h>
#include<torch/script.h>
#include <iostream>
#include<vector>
#include<opencv2/highgui.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main()
{
    torch::Device m_device(torch::kCUDA);
    torch::jit::script::Module SuperPoints= torch::jit::load(
        "/home/hongfeng/CV/SuperGluePretrainedNetwork/superpoint.pt"
    );
    cout << "load model successful!" << endl;

    torch::Tensor input = torch::rand({1, 1, 480, 640}).to(m_device);
    input.print();

    SuperPoints.to(m_device);
    SuperPoints.eval();
    auto output = SuperPoints.forward({input});
    auto output_tuple = output.toTuple();
    auto feature = output_tuple->elements()[0].toTensor();
    auto score = output_tuple->elements()[1].toTensor();
    auto descriptors = output_tuple->elements()[2].toTensor();

    vector<int> inputs;
    cout << inputs << endl;
    inputs.push_back(1);
    inputs.push_back(2);
    inputs.push_back(3);
    cout << inputs << endl;
    inputs.erase(inputs.begin(), inputs.begin()+2);
    cout << inputs << endl;

    return 0;
}