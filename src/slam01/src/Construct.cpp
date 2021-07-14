#include"viosystem/Construct.h"
#include<thread>

Construct::Construct(){}

Construct::Construct(cv::Mat &K, float sigma)
{
    cK = K.clone();
    cSigma = sigma;
    cSigma_2 = sigma*sigma;
}

bool Construct::ReconstructFromTwoFrames(vector<cv::Point2f> points0, vector<cv::Point2f> points1, cv::Mat &R10, cv::Mat &t10,
                                                                                            vector<cv::Point3f> &cP3D, vector<bool> &cTriangulated)
{
    cPoints0 = points0;
    cPoints1 = points1;
    if(points0.size()==0)    return false;

    const int N = points0.size();
    cout << "N: " << N << endl;
    vector<size_t> cIndices_all;                    // 所有的序号
    cIndices_all.reserve(N);
    for(int i=0; i<N; i++)
    {
        cIndices_all.push_back(i);
    }

    vector<size_t> cIndices_avaliable;               // 暂存没有被随机采样到的序号
    // 随机取cMax_iterations组8个的数据用于计算R，t
    cMax_iterations = min(30, N/8);
    cSet = vector<vector<size_t>>(cMax_iterations, vector<size_t>(8, 0));
    DUtils::Random::SeedRandOnce(0);    // 设置随机数种子
    for(int it=0; it<cMax_iterations; it++)
    {
        cIndices_avaliable = cIndices_all;
        for(size_t j=0; j<8; j++)
        {
            int randnum = DUtils::Random::RandomInt(0, cIndices_avaliable.size()-1);            // 这里上线设为size()-1而不能设为N-1，因为size()在动态变化
            int idx = cIndices_avaliable[randnum];
            cSet[it][j] = idx;
            cIndices_avaliable[randnum] = cIndices_avaliable.back();
            cIndices_avaliable.pop_back();
        }
    }

    vector<bool> Match_H, Match_F;
    float SH, SF;
    cv::Mat H, F;
    vector<bool> cTriangulated_H, cTriangulated_F;
    thread calculateH(&Construct::CalculateHomography, this, ref(Match_H), ref(SH), ref(H));
    thread calculateF(&Construct::CalculateFundamental, this, ref(Match_F), ref(SF), ref(F));
    calculateH.join();
    calculateF.join();

    float cMinParallax = 1.0;
    // cout<<"homography_matrix is "<<endl<< H <<endl;
    if(ReconstructFromH(Match_H, H, cK, R10, t10, cP3D, cTriangulated_H, cMinParallax, 50))
        cout << 1010101010 << endl;

    // for(int i=0; i<cKpts0.size(0); i++)
    // {
    //     cv::Mat p1(3, 1, CV_32F);
    //     cv::Mat p2(1, 3, CV_32F);
    //     p1.at<float>(0) = cKpts0[i][0].item().toFloat();
    //     p1.at<float>(1) = cKpts0[i][1].item().toFloat();
    //     p1.at<float>(2) = 0;
    //     p2.at<float>(0) = cKpts1[i][0].item().toFloat();
    //     p2.at<float>(1) = cKpts1[i][1].item().toFloat();
    //     p2.at<float>(2) = 0;
    //     auto result = p2*F*p1;
    //     cout << result << endl;
    // }

    // cout<<"fundamental_matrix is "<<endl<< F <<endl;
    if(ReconstructFromF(Match_F, F, cK, R10, t10, cP3D, cTriangulated_F, cMinParallax, 50))
        cout << 111111111111 << endl;
    return true;
}

void Construct::CalculateHomography(vector<bool> &Match_H, float &score, cv::Mat &H10)
{
    const int N = cPoints0.size();

    // 坐标归一化
    vector<cv::Point2f> cPosition0, cPosition1;
    cv::Mat T0, T1;
    Normalize(cPoints0, cPosition0, T0);
    Normalize(cPoints1, cPosition1, T1);
    cv::Mat T1inv = T1.inv();                                           // 这里的求逆是由H的形式决定的，P1=H*P0

    score = 0.0;
    Match_H = vector<bool>(N,false);
    vector<cv::Point2f> cPosition0i(8);
    vector<cv::Point2f> cPosition1i(8);
    cv::Mat H10i, H01i;
    vector<bool> cCurrentInliers(N, false);
    float cCurrentScore;

    for(int it=0; it<cMax_iterations; it++)
    {
        for(size_t j=0; j<8; j++)
        {
            int idx = cSet[it][j];
            cPosition0i[j] = cPosition0[idx];
            cPosition1i[j] = cPosition1[idx];
        }
        cv::Mat H_it = ComputeH10(cPosition0i, cPosition1i);
        H10i = T1inv*H_it*T0;               // 去归一化
        H01i = H10i.inv();

        cCurrentScore = CheckHomography(H10i, H01i, cCurrentInliers, cSigma);
        if(cCurrentScore>score)
        {
            Match_H = cCurrentInliers;
            score = cCurrentScore;
            H10 = H10i.clone();
        }
    }
}

void Construct::CalculateFundamental(vector<bool> &Match_F, float &score, cv::Mat &F10)
{
    const int N = cPoints0.size();

    vector<cv::Point2f> cPosition0, cPosition1;
    cv::Mat T0,T1;
    Normalize(cPoints0, cPosition0, T0);
    Normalize(cPoints1, cPosition1, T1);
    cv::Mat T1t = T1.t();                                                       // 这里为什么只需要转置,是由F的形式决定的P1.t()*F*P0=0

    score = 0.0;
    Match_F = vector<bool>(N, false);
    vector<cv::Point2f> cPosition0i(8);
    vector<cv::Point2f> cPosition1i(8);
    cv::Mat F10i;
    vector<bool> cCurrentInliers(N, false);
    float cCurrentScore;

    for(int it=0; it<cMax_iterations; it++)
    {
        for(size_t j=0; j<8; j++)
        {
            int idx = cSet[it][j];
            cPosition0i[j] = cPosition0[idx];
            cPosition1i[j] = cPosition1[idx];
        }
        cv::Mat F_it = ComputeF10(cPosition0i, cPosition1i);
        F10i = T1t*F_it*T0;

        cCurrentScore = CheckFundamental(F10i, cCurrentInliers, cSigma);
        if(cCurrentScore>score)
        {
            Match_F = cCurrentInliers;
            score = cCurrentScore;
            F10 = F10i.clone();
        }
    }
}

/*****************************************************************
 *     函数功能：对坐标归一化
 *     函数参数介绍：ckpts：特征点坐标tensor形式；vNormalizedPoints：归一化后的坐标；T：归一化的数据与原数据差一个矩阵变换；
 *     备注：运行之后
 *      cNormalizedPoints：存放归一化之后的坐标;
 *      T:NormalizedPoints[i] = T * kpts[i].item().toFloat()    矩阵形式
 ******************************************************************/
void Construct::Normalize(const vector<cv::Point2f> points, vector<cv::Point2f> &NormalizedPoints, cv::Mat &T)
{
    float meanX = 0;                        // x,y坐标平均值
    float meanY = 0;
    const int N = points.size();
    for(int i=0; i<N; i++)
    {
        meanX += points[i].x;
        meanY += points[i].y;
    }
    meanX = meanX/N;
    meanY = meanY/N;

    NormalizedPoints.resize(N);
    float meanDevX = 0;             // x,y坐标的中心距值
    float meanDevY = 0;
    for(int i=0; i<N; i++)
    {
        NormalizedPoints[i].x = points[i].x - meanX;
        NormalizedPoints[i].y = points[i].y -meanY;
        meanDevX += fabs(NormalizedPoints[i].x);
        meanDevY += fabs(NormalizedPoints[i].y);
    }
    meanDevX = meanDevX/N;
    meanDevY = meanDevY/N;

    float sX = 1.0/meanDevX;
    float sY = 1.0/meanDevY;
    for(int i=0; i<N; i++)
    {
        NormalizedPoints[i].x = NormalizedPoints[i].x *sX;
        NormalizedPoints[i].y = NormalizedPoints[i].y *sY;
    }
    
    T = cv::Mat::eye(3, 3, CV_32F);
    T.at<float>(0, 0) = sX;
    T.at<float>(1, 1) = sY;
    T.at<float>(0, 2) = -meanX*sX;
    T.at<float>(1, 2) = -meanY*sY;
}

/*****************************************************************
 *     函数功能：计算单应矩阵（14讲P171）
 *     函数参数介绍：P0：归一化后特征点1的坐标；P1：归一化后特征点2的坐标；
 *     备注：运行之后   返回计算的单应矩阵H
 ******************************************************************/
cv::Mat Construct::ComputeH10(const vector<cv::Point2f> &P0, const vector<cv::Point2f> &P1)
{
    const int N = P0.size();
    cv::Mat A(2*N, 9, CV_32F);
    for(int i=0; i<N; i++)
    {
        const float u0 = P0[i].x;
        const float v0 = P0[i].y;
        const float u1 = P1[i].x;
        const float v1 = P1[i].y;
        // 此处相对书上多了个负号，不影响
        A.at<float>(2*i, 0) = 0.0;
        A.at<float>(2*i, 1) = 0.0;
        A.at<float>(2*i, 2) = 0.0;
        A.at<float>(2*i, 3) = -u0;
        A.at<float>(2*i, 4) = -v0;
        A.at<float>(2*i, 5) = -1;
        A.at<float>(2*i, 6) = u0*v1;
        A.at<float>(2*i, 7) = v0*v1;
        A.at<float>(2*i, 8) = v1;

        A.at<float>(2*i+1, 0) = u0;
        A.at<float>(2*i+1, 1) = v0;
        A.at<float>(2*i+1, 2) = 1;
        A.at<float>(2*i+1, 3) = 0.0;
        A.at<float>(2*i+1, 4) = 0.0;
        A.at<float>(2*i+1, 5) = 0.0;
        A.at<float>(2*i+1, 6) = -u0*u1;
        A.at<float>(2*i+1, 7) = -v0*u1;
        A.at<float>(2*i+1, 8) = -u1;
    }
    // 将求解Ah=uv转化为求解Ax=0
    cv::Mat w, u, vt;
    cv::SVDecomp(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
    return vt.row(8).reshape(0, 3);
}

cv::Mat Construct::ComputeF10(const vector<cv::Point2f> &P0, const vector<cv::Point2f> &P1)
{
    const int N = P0.size();
    cv::Mat A(N, 9, CV_32F);
    for(int i=0; i<N; i++)
    {
        const float u0 = P0[i].x;
        const float v0 = P0[i].y;
        const float u1 = P1[i].x;
        const float v1 = P1[i].y;

        A.at<float>(i, 0) = u1*u0;
        A.at<float>(i, 1) = u1*v0;
        A.at<float>(i, 2) = u1;
        A.at<float>(i, 3) = v1*u0;
        A.at<float>(i, 4) = v1*v0;
        A.at<float>(i, 5) = v1;
        A.at<float>(i, 6) = u0;
        A.at<float>(i, 7) = v0;
        A.at<float>(i, 8) = 1;
    }

    cv::Mat w, u, vt;

    cv::SVDecomp(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
    // F的值应为Ax=0的解，同时第三个特征值应该为0
    cv::Mat Fpre = vt.row(8).reshape(0, 3);
    cv::SVDecomp(Fpre, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
    w.at<float>(2) = 0;
    return u*cv::Mat::diag(w)*vt;
} 

/*****************************************************************
 *     函数功能：检验基础矩阵函数
 *     函数参数介绍：H10：计算出的单应矩阵；H01：计算出的单应矩阵的逆矩阵；cMatchesInliers：检测匹配对通过与否；sigma：样本标准差；
 *     备注：运行之后
 *      vbMatchesInliers：记录在基础矩阵H10第i个匹配对通过检测与否
 *      score：基础矩阵H10的质量
 ******************************************************************/
float Construct::CheckHomography(const cv::Mat &H10, const cv::Mat &H01, vector<bool> &cMatchInliers, float sigma)
{
    const int N = cPoints0.size();
    
    const float h11 = H10.at<float>(0, 0);
    const float h12 = H10.at<float>(0, 1);
    const float h13 = H10.at<float>(0, 2);
    const float h21 = H10.at<float>(1, 0);
    const float h22 = H10.at<float>(1, 1);
    const float h23 = H10.at<float>(1, 2);
    const float h31 = H10.at<float>(2, 0);
    const float h32 = H10.at<float>(2, 1);
    const float h33 = H10.at<float>(2, 2);

    const float hinv11 = H01.at<float>(0, 0);
    const float hinv12 = H01.at<float>(0, 1);
    const float hinv13 = H01.at<float>(0, 2);
    const float hinv21 = H01.at<float>(1, 0);
    const float hinv22 = H01.at<float>(1, 1);
    const float hinv23 = H01.at<float>(1, 2);
    const float hinv31 = H01.at<float>(2, 0);
    const float hinv32 = H01.at<float>(2, 1);
    const float hinv33 = H01.at<float>(2, 2);

    cMatchInliers.resize(N);
    float score = 0;
    const float th = 5.991;
    const float invSigmaSquare = 1.0/(sigma*sigma);

    for(int i=0; i<N; i++)
    {
        bool cPassed = true;

        const float u0 = cPoints0[i].x;
        const float v0 = cPoints0[i].y;
        const float u1 = cPoints1[i].x;
        const float v1 = cPoints1[i].y;

        const float cDenominator01 = 1.0/(hinv31*u1+hinv32*v1+hinv33);
        const float u0from1 = (hinv11*u1+hinv12*v1+hinv13)*cDenominator01;
        const float v0from1 = (hinv21*u1+hinv22*v1+hinv23)*cDenominator01;
        const float cSquareDist01 = (u0-u0from1)*(u0-u0from1)+(v0-v0from1)*(v0-v0from1);
        const float cChiSquare01 = cSquareDist01*invSigmaSquare;                    // 误差平方和/随机项方差的结果服从卡方分布
        if(cChiSquare01>th)    cPassed = false;
        else                                                 score += th - cChiSquare01;

        const float cDenominator10 = 1.0/(h31*u0+h32*v0+h33);
        const float u1from0 = (h11*u0+h12*v0+h13)*cDenominator10;
        const float v1from0 = (h21*u0+h22*v0+h23)*cDenominator10;
        const float cSquareDist10 = (u1-u1from0)*(u1-u1from0)+(v1-v1from0)*(v1-v1from0);
        const float cChiSquare10 = cSquareDist10*invSigmaSquare;
        if(cChiSquare10>th) cPassed = false;
        else                                                score += th - cChiSquare10;

        if(cPassed)     cMatchInliers[i] = true;
        else                    cMatchInliers[i] = false;
    }
    return score;
}

/*****************************************************************
 *     函数功能：检验基础矩阵函数
 *     函数参数介绍：F10：计算出的基础矩阵；cMatchesInliers：检测匹配对通过与否；sigma：样本标准差；
 *     备注：运行之后
 *      vbMatchesInliers：记录在基础矩阵F21第i个匹配对通过检测与否
 *      score：基础矩阵F21的得分
 ******************************************************************/
float Construct::CheckFundamental(const cv::Mat &F10, vector<bool> &cMatchInliers, float sigma)
{
    const int N = cPoints0.size();

    const float f11 = F10.at<float>(0, 0);
    const float f12 = F10.at<float>(0, 1);
    const float f13 = F10.at<float>(0, 2);
    const float f21 = F10.at<float>(1, 0);
    const float f22 = F10.at<float>(1, 1);
    const float f23 = F10.at<float>(1, 2);
    const float f31 = F10.at<float>(2, 0);
    const float f32 = F10.at<float>(2, 1);
    const float f33 = F10.at<float>(2, 2);

    cMatchInliers.resize(N);
    float score = 0;
    const float th = 3.841;
    const float thScore = 5.991;
    const float invSigmaSquare = 1.0/(sigma*sigma);

    for(int i=0; i<N; i++)
    {
        bool cPassed =true;

        const float u0 = cPoints0[i].x;
        const float v0 = cPoints0[i].y;
        const float u1 = cPoints1[i].x;
        const float v1 = cPoints1[i].y;

        // 对极约束误差,关键帧0到1
        const float a1 = f11*u0+f12*v0+f13;
        const float b1 = f21*u0+f22*v0+f23;
        const float c1 = f31*u0+f32*v0+f33;
        const float cShouldBe0_1 = a1*u1+b1*v1+c1;
        const float cSquareDist10 = cShouldBe0_1*cShouldBe0_1/(a1*a1+b1*b1);
        const float cChiSquare10 = cSquareDist10*invSigmaSquare;
        if(cChiSquare10>th)         cPassed = false;
        else                                          score += thScore - cChiSquare10;

        const float a0 = f11*u1+f21*v1+f31;
        const float b0 = f12*u1+f22*v1+f32;
        const float c0 = f13*u1+f23*v1+f33;
        const float cShouldBe0_0 = a0*u0+b0*v0+c0;
        const float cSquareDist01 = cShouldBe0_0*cShouldBe0_0/(a0*a0+b0*b0);
        const float cChiSquare01 = cSquareDist01*invSigmaSquare;
        if(cChiSquare01>th)         cPassed = false;
        else                                          score += thScore-cChiSquare01;

        if(cPassed)                            cMatchInliers[i] = true;
        else                                          cMatchInliers[i] = false;
    }
    return score;
}

bool Construct::ReconstructFromH(vector<bool> &cMatchInliers, cv::Mat &H10,cv::Mat &K, cv::Mat &R10, cv::Mat &t10,
                                                        vector<cv::Point3f> &cP3d, vector<bool> &cTrangulated, float cMinParallax, int cMinTriangulated)
{
    int N = 0;                                                                                              // 通过H检测的数量
    for(size_t i=0; i<cMatchInliers.size(); i++)
        if(cMatchInliers[i])    N++;
    cout << "H-passed number: " << N << "; 0.9*N: " << 0.9*N << endl;

    // 对单应矩阵
    cv::Mat cKinv = cK.inv();
    cv::Mat A = cKinv*H10*cK;
    cv::Mat w,U,Vt,V;
    cv::SVD::compute(A, w, U, Vt, cv::SVD::FULL_UV);
    V = Vt.t();

    float s = cv::determinant(U)*cv::determinant(Vt);           // U和V行列式相乘，理论上应该为1
    float d1 = w.at<float>(0);
    float d2 = w.at<float>(1);
    float d3 = w.at<float>(2);
    if(d1/d2<1.00001 || d2/d3 < 1.00001)
    {
        cout << "d1/d2= " << d1/d2 << ", d2/d3= " << d2/d3 << endl;
        return false;       // 特征值检测
    }

    vector<cv::Mat> cR, ct, cn;
    cR.reserve(8);
    ct.reserve(8);
    cn.reserve(8);

    float cAux1 = sqrt((d1*d1-d2*d2)/(d1*d1-d3*d3));
    float cAux3 = sqrt((d2*d2-d3*d3)/(d1*d1-d3*d3));
    float x1[] = {cAux1, cAux1, -cAux1, -cAux1};
    float x3[] = {cAux3, -cAux3, cAux3, -cAux3};

    //case d'=d2    情况1
    float cAuxSinThta = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1+d3)*d2);         // 正负为定
    float cCosThta = (d2*d2+d1*d3)/((d1+d3)*d2);
    float cSinThta[] = {cAuxSinThta, -cAuxSinThta, -cAuxSinThta, cAuxSinThta};
    for(int i=0; i<4; i++)
    {
        cv::Mat Rp = cv::Mat::eye(3, 3, CV_32F);
        Rp.at<float>(0, 0) = cCosThta;
        Rp.at<float>(0, 2) = -cSinThta[i];
        Rp.at<float>(2, 0) = cSinThta[i];
        Rp.at<float>(2, 2) = cCosThta;
        cv::Mat R = s*U*Rp*Vt;
        cR.push_back(R);

        cv::Mat tp(3, 1, CV_32F);
        tp.at<float>(0) = x1[i];
        tp.at<float>(1) = 0;
        tp.at<float>(2) = x3[i];
        tp *= (d1-d3);
        cv::Mat t = U*tp;
        ct.push_back(t/cv::norm(t));

        cv::Mat np(3, 1, CV_32F);
        np.at<float>(0) = x1[i];
        np.at<float>(1) = 0;
        np.at<float>(2) = x3[i];
        cv::Mat n = V*np;
        if(n.at<float>(2)<0)    n = -n;                 // 保证z轴正向
        cn.push_back(n);
    }

    //case d'=-d2    情况2
    float cAuxSinPhi = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1-d3)*d2);
    float cCosPhi = (d1*d3-d2*d2)/((d1-d3)*d2);
    float cSinPhi[] = {cAuxSinPhi, -cAuxSinPhi, -cAuxSinPhi, cAuxSinPhi};
    for(int i=0; i<4; i++)
    {
        cv::Mat Rp = cv::Mat::eye(3, 3, CV_32F);
        Rp.at<float>(0, 0) = cCosPhi;
        Rp.at<float>(0, 2) = cSinPhi[i];
        Rp.at<float>(1, 1) = -1;
        Rp.at<float>(2, 0) = cSinPhi[i];
        Rp.at<float>(2, 2) = -cCosPhi;
        cv::Mat R = s*U*Rp*Vt;
        cR.push_back(R);

        cv::Mat tp(3, 1, CV_32F);
        tp.at<float>(0) = x1[i];
        tp.at<float>(1) = 0;
        tp.at<float>(2) = x3[i];
        tp *= (d1+d3);
        cv::Mat t = U*tp;
        ct.push_back(t/cv::norm(t));

        cv::Mat np(3, 1, CV_32F);
        np.at<float>(0) = x1[i];
        np.at<float>(1) = 0;
        np.at<float>(2) = x3[i];
        cv::Mat n = V*np;
        if(n.at<float>(2)>0)    n = -n;
        cn.push_back(n);
    }

    int cBest = 0;
    int cScond = 0;
    int cBestIdx = -1;
    float cBestParallx = -1;
    vector<cv::Point3f> cBestP3D;
    vector<bool> cBestTriangulated;
    cout << "========Homography========" << endl;
    for(size_t i=0; i<8; i++)
    {
        // cout << "======== " << i << " ========" << endl;
        // cout << "rotation" << i << " = " << endl;
        // cout << cR[i] << endl;
        // cout << "translation" << i << " = " << endl;
        // cout << ct[i] << endl;

        float cPrallax_i;
        vector<cv::Point3f> cP3D_i;
        vector<bool> cTriangulated_i;
        int cGood =CheckRt(cR[i], ct[i], cPoints0, cPoints1, cMatchInliers, cK, cP3D_i, 4.0*cSigma_2, cTriangulated_i, cPrallax_i);
        cout << "cGood: " << cGood << endl;

        if(cGood>cBest)
        {
            cScond = cBest;
            cBest = cGood;
            cBestIdx = i;
            cBestParallx = cPrallax_i;
            cBestP3D = cP3D_i;
            cBestTriangulated = cTriangulated_i;
        }
        else if(cGood>cScond)
        {
            cScond = cGood;
        }
    }

    if(cScond<0.75*cBest && cBestParallx>=cMinParallax && cBest>cMinTriangulated && cBest>0.9*N)
    {
        cR[cBestIdx].copyTo(R10);
        ct[cBestIdx].copyTo(t10);
        cP3d = cBestP3D;
        cTrangulated = cBestTriangulated;
        return true;
    }
    
    return false;
    // to do 先写完CheckRT()
}

bool Construct::ReconstructFromF(vector<bool> &cMatchInliers, cv::Mat &F10,cv::Mat &K, cv::Mat &R10, cv::Mat &t10,
                                                        vector<cv::Point3f> &cP3d, vector<bool> &cTrangulated, float cMinParallax, int cMinTriangulated)
{
    int N=0;
    for(size_t i=0; i<cMatchInliers.size(); i++)
        if(cMatchInliers[i])    N++;

    cv::Mat E10 = K.t()*F10*K;

    // vector<cv::Point2f> points1;
    // vector<cv::Point2f> points2;
    // points1.reserve(cKpts0.size(0));
    // points2.reserve(cKpts1.size(0));
    // for(int i=0; i<cKpts0.size(0); i++)
    // {
    //     points1[i].x = cKpts0[i][0].item().toFloat();
    //     points1[i].y = cKpts0[i][1].item().toFloat();
    // }
    // for(int i=0; i<cKpts0.size(0); i++)
    // {
    //     points2[i].x = cKpts1[i][0].item().toFloat();
    //     points2[i].y = cKpts1[i][1].item().toFloat();
    // }
    // cv::Point2d principal_point ( 325.1, 249.7 );
    // double focal_length = 521;
    // cv::Mat R, t;
    // cv::recoverPose(E10, points1, points2, R, t, focal_length, principal_point );
    // cout<<"R is "<<endl<<R<<endl;
    // cout<<"t is "<<endl<<t<<endl;

    cv::Mat cR1, cR2, t;
    DecomposeE(E10, cR1, cR2, t);
    cv::Mat ct1 = t;
    cv::Mat ct2 = -t;
    // cout << cR1 << ct1 << endl;
    // cout << cR2 << ct2 << endl;

    vector<cv::Point3f> cP3d1, cP3d2, cP3d3, cP3d4;
    vector<bool> cTrangulated1, cTrangulated2, cTrangulated3, cTrangulated4;
    float cParallax1, cParallax2, cParallax3, cParallax4;
    cout << "========Fundamental========" << endl;
    int cGood1 = CheckRt(cR1, ct1, cPoints0, cPoints1, cMatchInliers,cK, cP3d1, 4.0*cSigma_2, cTrangulated1, cParallax1);
    int cGood2 = CheckRt(cR2, ct1, cPoints0, cPoints1, cMatchInliers, cK, cP3d2, 4.0*cSigma_2, cTrangulated2, cParallax2);
    int cGood3 = CheckRt(cR1, ct2, cPoints0, cPoints1, cMatchInliers, cK, cP3d3, 4.0*cSigma_2, cTrangulated3, cParallax3);
    int cGood4 = CheckRt(cR2, ct2, cPoints0, cPoints1, cMatchInliers, cK, cP3d4, 4.0*cSigma_2, cTrangulated4, cParallax4);
    int cMaxGood = max(cGood1, max(cGood2, max(cGood3, cGood4)));
    cout << "F-passed number: " << N << "; 0.9*N: " << 0.9*N << endl;
    cout << "cGood1: " << cGood1 << endl;
    cout << "cGood2: " << cGood2 << endl;
    cout << "cGood3: " << cGood3 << endl;
    cout << "cGood4: " << cGood4 <<  endl;

    R10 = cv::Mat();
    t10 = cv::Mat();
    int cMinGood = max(static_cast<int>(0.9*N), cMinTriangulated);

    int cPossibleNum = 0;
    if(cGood1>0.7*cMaxGood)         cPossibleNum++;
    if(cGood2>0.7*cMaxGood)         cPossibleNum++;
    if(cGood3>0.7*cMaxGood)         cPossibleNum++;
    if(cGood4>0.7*cMaxGood)         cPossibleNum++;
    cout << "Num: " << cPossibleNum  << endl;
    if(cMaxGood<cMinGood || cPossibleNum>1)           return false;
    if(cMaxGood==cGood1)
    {
        cout << "Parallax1: " << cParallax1 << "; MinParallax: " << cMinParallax << endl;
        if(cParallax1>cMinParallax)
        {
            cP3d = cP3d1;
            cTrangulated = cTrangulated1;
            cR1.copyTo(R10);
            ct1.copyTo(t10);
            return true;
        }
    }
    else if(cMaxGood==cGood2)
    {
        cout << "Parallax2: " << cParallax2 << "; MinParallax: " << cMinParallax << endl;
        if(cParallax2>cMinParallax)
        {
            cP3d = cP3d2;
            cTrangulated = cTrangulated2;
            cR2.copyTo(R10);
            ct1.copyTo(t10);
            return true;
        }
    }
    else if(cMaxGood==cGood3)
    {
        cout << "Parallax3: " << cParallax3 << "; MinParallax: " << cMinParallax << endl;
        if(cParallax3>cMinParallax)
        {
            cP3d = cP3d3;
            cTrangulated = cTrangulated3;
            cR1.copyTo(R10);
            ct2.copyTo(t10);
            return true;
        }
    }
    else if(cMaxGood==cGood4)
    {
        cout << "Parallax4: " << cParallax4 << "; MinParallax: " << cMinParallax << endl;
        if(cParallax4>cMinParallax)
        {
            cP3d = cP3d4;
            cTrangulated = cTrangulated4;
            cR2.copyTo(R10);
            ct2.copyTo(t10);
            return true;
        }
    }

    return false;
}

/*****************************************************************
 *     函数功能：检查R、t是否正确的
 *     函数参数介绍：cR：旋转矩阵；ct：位移；kpts0：关键帧0的特征点张量；kpts1：关键帧1的特征点张量；vbMatchesInliers：保存最好的H或F下匹配对通过与否；
 *                                    K：内参；cP3D：通过验证的3D点坐标；th2：4倍样本方差，投影误差阈值；cGood：记录通过验证且有一定视差角的匹配；cParallax：最小视差角；
 *     备注：运行之后
 *      vP3D、vbGood、parallax计算完成；返回nGood：质量好的匹配的个数
 ******************************************************************/
int Construct::CheckRt(const cv::Mat &cRi, const cv::Mat &cti, const vector<cv::Point2f> &points0, const vector<cv::Point2f> &points1, vector<bool> &cMatchInliers,
                                                        const cv::Mat &K, vector<cv::Point3f> &cP3D, float th2, vector<bool> &cGood, float &cParallax)
{
    const float fx = K.at<float>(0, 0);
    const float cx = K.at<float>(0, 2); 
    const float fy = K.at<float>(1, 1);
    const float cy = K.at<float>(1, 2);

    const int N = points0.size();
    cGood = vector<bool>(N, false);                 // 记录匹配关键点恢复出来的3D点是否有效
    cP3D.resize(N);
    vector<float> cCosParallaxs;                                                 // 记录误差角余弦值
    cCosParallaxs.reserve(N);
    cv::Mat P1(3, 4, CV_32F, cv::Scalar(0));                            // 投影矩阵P1=K[ I | O ]，世界坐标系下
    K.copyTo(P1.rowRange(0, 3).colRange(0, 3));
    cv::Mat O1 = cv::Mat::zeros(3, 1, CV_32F);                      // 世界坐标系原点坐标
    cv::Mat P2(3, 4, CV_32F);                                                       // 投影矩阵P2=K[ R | t ]
    cRi.copyTo(P2.rowRange(0, 3).colRange(0, 3));
    cti.copyTo(P2.rowRange(0, 3).col(3));
    P2 = K*P2;
    cv::Mat O2 = -cRi.t()*cti;                                                            // 相机光心在世界坐标系坐标

    int cN_Good = 0;
    int negNum = 0;
    for(size_t i=0; i<N; i++)                                   // 遍历匹配对
    {
        if(!cMatchInliers[i])                                                                       continue;           // 没通过检测，直接跳过

        cv::Mat cP3Dc0;
        Triangulate(points0[i], points1[i], P1, P2, cP3Dc0);
        if(cP3Dc0.at<float>(2)<=0)  negNum++;
        // 保留有效三角化结果
        if(!isfinite(cP3Dc0.at<float>(0)) || !isfinite(cP3Dc0.at<float>(1) || !isfinite(cP3Dc0.at<float>(2))))
        {
            cGood[i] = false;
            continue;
        }

        cv::Mat cNormal_Oto3D = cP3Dc0-O1;
        float cDist_Oto3D = cv::norm(cNormal_Oto3D);         // 坐标系原点到3D点距离（世界坐标系）
        cv::Mat cNormal_Cto3D = cP3Dc0-O2;
        float cDist_Cto3D = cv::norm(cNormal_Cto3D);         // 相机光心到3D点距离（世界坐标系）
        float cCosParallax = cNormal_Oto3D.dot(cNormal_Cto3D)/(cDist_Oto3D*cDist_Cto3D);      // 视差角余弦值

        // 检查在前后两个相机坐标系下3D点坐标是否为正
        if(cP3Dc0.at<float>(2)<=0 && cCosParallax<0.99998)      continue;
        cv::Mat cP3Dc1 = cRi*cP3Dc0+cti;
        if(cP3Dc1.at<float>(2)<=0 && cCosParallax<0.99998)      continue;

        float cImg0x, cImg0y;                                                           // 计算的3D坐标投影回去的像素坐标
        float cZ0inv = 1.0/cP3Dc0.at<float>(2);
        cImg0x = fx*cP3Dc0.at<float>(0)*cZ0inv + cx;
        cImg0y = fy*cP3Dc0.at<float>(1)*cZ0inv + cy;
        // cout << "imgx: " << cImg0x << ", x: " << kpts0[i][0].item().toFloat() << "; imgy: " <<cImg0y << ", y: " << kpts0[i][1].item().toFloat();
        float cSquareError0 = (cImg0x-points0[i].x)*(cImg0x-points0[i].x)+
                                                        (cImg0y - points0[i].y)*(cImg0y - points0[i].y);
        // cout <<"; error: " << cSquareError0 << endl;
        if(cSquareError0>th2)                                                                   continue;

        float cImg1x, cImg1y;
        float cZ1inv = 1.0/cP3Dc1.at<float>(2);
        cImg1x = fx*cP3Dc1.at<float>(0)*cZ1inv + cx;
        cImg1y = fy*cP3Dc1.at<float>(1)*cZ1inv + cy;
        float cSquareError1 = (cImg1x-points1[i].x)*(cImg1x-points1[i].x)+
                                                        (cImg1y-points1[i].y)*(cImg1y-points1[i].y);
        if(cSquareError1>th2)                                                                   continue;
        
        cCosParallaxs.push_back(cCosParallax);
        cP3D[i] = cv::Point3f(cP3Dc0.at<float>(0), cP3Dc0.at<float>(1), cP3Dc0.at<float>(2));
        cN_Good++;
        if(cCosParallax<0.99998)            cGood[i] = true;
    }
    cout << "negtive rate: " << negNum*100/N << "%" << endl;

    if(cN_Good>0)
    {
        sort(cCosParallaxs.begin(), cCosParallaxs.end());
        size_t idx = min(50, int(cCosParallaxs.size()-1));
        cParallax = acos(cCosParallaxs[idx])*180/CV_PI;
    }
    else    cParallax = 0;

    return cN_Good;
}

/*****************************************************************
 *     函数功能：三角测量（14讲P177）
 *     函数参数介绍：kpt0：特征点1；kpt1：特征点2；P1：投影矩阵1；P2：投影矩阵2；x3D：计算的3D点坐标；
 *     备注：运行之后
 *      x3D：计算的kpt0对应的3D点坐标（x，y，z）
 ******************************************************************/
void Construct::Triangulate(const cv::Point2f &point0, const cv::Point2f &point1, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D)
{
    cv::Mat A(4, 4, CV_32F);

    A.row(0) = point0.x*P1.row(2) - P1.row(0);
    A.row(1) = point0.y*P1.row(2) - P1.row(1);
    A.row(2) = point1.x*P2.row(2) - P2.row(0);
    A.row(3) = point1.y*P2.row(2) - P2.row(1);

    cv::Mat w, u, vt;
    cv::SVD::compute(A, w, u, vt, cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
    x3D = vt.row(3).t();                                                                // x3D=[X,Y,Z,W]
    x3D = x3D.rowRange(0, 3)/x3D.at<float>(3);              // 归一化x3D=[X/W,Y/W,Z/W]
}

void Construct::DecomposeE(const cv::Mat &E10, cv::Mat &cR1, cv::Mat &cR2, cv::Mat &ct)
{
    cv::Mat w, u, vt;
    cv::SVD::compute(E10, w, u, vt);

    u.col(2).copyTo(ct);
    ct = ct/cv::norm(ct);

    cv::Mat W(3, 3, CV_32F, cv::Scalar(0));
    W.at<float>(0, 1) = -1;
    W.at<float>(1, 0) = 1;
    W.at<float>(2, 2) = 1;

    cR1 = u*W*vt;
    if(cv::determinant(cR1)<0)  cR1 = -cR1;
    cR2 = u*W.t()*vt;
    if(cv::determinant(cR2)<0)  cR2 = -cR2;
}

























// 仅仅作局部测试用
        // vector<int> idx_vec;
        // for(size_t j=0; j<8; j++)
        // {
        //     int idx = cSet[it][j];
        //     idx_vec.push_back(idx);
        // }
        // TestHi(H10i, H01i, cCurrentInliers, cSigma, idx_vec);
void Construct::TestHi(const cv::Mat &H10, const cv::Mat &H01, vector<bool> &cMatchInliers, float sigma, vector<int> idx_vec)
{
    const float h11 = H10.at<float>(0, 0);
    const float h12 = H10.at<float>(0, 1);
    const float h13 = H10.at<float>(0, 2);
    const float h21 = H10.at<float>(1, 0);
    const float h22 = H10.at<float>(1, 1);
    const float h23 = H10.at<float>(1, 2);
    const float h31 = H10.at<float>(2, 0);
    const float h32 = H10.at<float>(2, 1);
    const float h33 = H10.at<float>(2, 2);

    const float hinv11 = H01.at<float>(0, 0);
    const float hinv12 = H01.at<float>(0, 1);
    const float hinv13 = H01.at<float>(0, 2);
    const float hinv21 = H01.at<float>(1, 0);
    const float hinv22 = H01.at<float>(1, 1);
    const float hinv23 = H01.at<float>(1, 2);
    const float hinv31 = H01.at<float>(2, 0);
    const float hinv32 = H01.at<float>(2, 1);
    const float hinv33 = H01.at<float>(2, 2);

    cMatchInliers.resize(8);
    for(int i=0; i<8; i++)
    {
        bool cPassed = true;
        const float u0 = cKpts0[idx_vec[i]][0].item().toFloat();
        const float v0 = cKpts0[idx_vec[i]][1].item().toFloat();
        const float u1 = cKpts1[idx_vec[i]][0].item().toFloat();
        const float v1 = cKpts1[idx_vec[i]][1].item().toFloat();

        const float cDenominator01 = 1.0/(hinv31*u1+hinv32*v1+hinv33);
        const float u0from1 = (hinv11*u1+hinv12*v1+hinv13)*cDenominator01;
        const float v0from1 = (hinv21*u1+hinv22*v1+hinv23)*cDenominator01;
        const float cSquareDist01 = (u0-u0from1)*(u0-u0from1)+(v0-v0from1)*(v0-v0from1);
        cout << "u0: " << u0 << ", " << "u0from1: " << u0from1 << ", " << "v0: " << v0 << ", " << "v0from1: " << v0from1 << "error: " << cSquareDist01 << endl;

        const float cDenominator10 = 1.0/(h31*u0+h32*v0+h33);
        const float u1from0 = (h11*u0+h12*v0+h13)*cDenominator10;
        const float v1from0 = (h21*u0+h22*v0+h23)*cDenominator10;
        const float cSquareDist10 = (u1-u1from0)*(u1-u1from0)+(v1-v1from0)*(v1-v1from0);
        cout << "u1: " << u1 << ", " << "u1from0: " << u1from0 << ", " << "v1: " << v1 << ", " << "v1from1: " << v1from0 << "error: " << cSquareDist10 << endl;
    }
}

        // cv::Mat H_itinv = H_it.inv();
        // TestH(H_it, H_itinv, cCurrentInliers, cSigma, cPosition0i, cPosition1i);
void Construct::TestH(const cv::Mat &H10, const cv::Mat &H01, vector<bool> &cMatchInliers, float sigma, vector<cv::Point2f> cPosition0i_, vector<cv::Point2f> cPosition1i_)
{
    const float h11 = H10.at<float>(0, 0);
    const float h12 = H10.at<float>(0, 1);
    const float h13 = H10.at<float>(0, 2);
    const float h21 = H10.at<float>(1, 0);
    const float h22 = H10.at<float>(1, 1);
    const float h23 = H10.at<float>(1, 2);
    const float h31 = H10.at<float>(2, 0);
    const float h32 = H10.at<float>(2, 1);
    const float h33 = H10.at<float>(2, 2);

    const float hinv11 = H01.at<float>(0, 0);
    const float hinv12 = H01.at<float>(0, 1);
    const float hinv13 = H01.at<float>(0, 2);
    const float hinv21 = H01.at<float>(1, 0);
    const float hinv22 = H01.at<float>(1, 1);
    const float hinv23 = H01.at<float>(1, 2);
    const float hinv31 = H01.at<float>(2, 0);
    const float hinv32 = H01.at<float>(2, 1);
    const float hinv33 = H01.at<float>(2, 2);

    cMatchInliers.resize(8);
    for(int i=0; i<8; i++)
    {
        bool cPassed = true;
        const float u0 = cPosition0i_[i].x;
        const float v0 = cPosition0i_[i].y;
        const float u1 = cPosition1i_[i].x;
        const float v1 = cPosition1i_[i].y;

        const float cDenominator01 = 1.0/(hinv31*u1+hinv32*v1+hinv33);
        const float u0from1 = (hinv11*u1+hinv12*v1+hinv13)*cDenominator01;
        const float v0from1 = (hinv21*u1+hinv22*v1+hinv23)*cDenominator01;
        const float cSquareDist01 = (u0-u0from1)*(u0-u0from1)+(v0-v0from1)*(v0-v0from1);
        cout << "u0: " << u0 << ", " << "u0from1: " << u0from1 << ", " << "v0: " << v0 << ", " << "v0from1: " << v0from1 << "error: " << cSquareDist01 << endl;

        const float cDenominator10 = 1.0/(h31*u0+h32*v0+h33);
        const float u1from0 = (h11*u0+h12*v0+h13)*cDenominator10;
        const float v1from0 = (h21*u0+h22*v0+h23)*cDenominator10;
        const float cSquareDist10 = (u1-u1from0)*(u1-u1from0)+(v1-v1from0)*(v1-v1from0);
        cout << "u1: " << u1 << ", " << "u1from0: " << u1from0 << ", " << "v1: " << v1 << ", " << "v1from1: " << v1from0 << "error: " << cSquareDist10 << endl;
    }
}