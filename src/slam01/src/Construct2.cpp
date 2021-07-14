#include"viosystem/Construct2.h"
#include<thread>

Construct2::Construct2(){}

Construct2::Construct2(const string &strSettingsFile, float sigma)
{
    mSigma = sigma;
    mSigma2 = sigma*sigma;

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
    K.copyTo(mK);
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
    cout << "K: " << mK << endl;
}

bool Construct2::Reconstruct(const at::Tensor &Kpts1, const at::Tensor &Kpts2,
                                        cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated)
{
    cKpts1 = Kpts1;       // 两个去畸变之后的关键点向量
    cKpts2 = Kpts2;

    const int N = cKpts1.size(0);               // 匹配对数

    // Indices for minimum set selection
    vector<size_t> vAllIndices;                             // 标准的序号
    vAllIndices.reserve(N);
    vector<size_t> vAvailableIndices;               // 用于随机采样的序号

    for(int i=0; i<N; i++)
    {
        vAllIndices.push_back(i);
    }
    mMaxIterations = max(30, N/8);
    // Generate sets of 8 points for each RANSAC iteration  为每次RANSAC迭代产生mMaxIterations组数据，每组八个点
    mvSets = vector< vector<size_t> >(mMaxIterations,vector<size_t>(8,0));

    DUtils::Random::SeedRandOnce(0);        // 设置随机数种子

    for(int it=0; it<mMaxIterations; it++)      // 遍历迭代次数
    {
        vAvailableIndices = vAllIndices;

        // Select a minimum set
        for(size_t j=0; j<8; j++)
        {
            int randi = DUtils::Random::RandomInt(0,vAvailableIndices.size()-1);        // 生成0-size的一个随机数
            int idx = vAvailableIndices[randi];     // 取关键点序号

            mvSets[it][j] = idx;                                    // 第it组第j个点为取出的序号

            vAvailableIndices[randi] = vAvailableIndices.back();        // 删掉已经取到的点
            vAvailableIndices.pop_back();
        }
    }

    // Launch threads to compute in parallel a fundamental matrix and a homography
    // 两个线程计算基础矩阵和单应矩阵
    vector<bool> vbMatchesInliersH, vbMatchesInliersF;      // 保存最好的结果下匹配对通过与否
    float SH, SF;               // 单应矩阵和基础矩阵的得分score
    cv::Mat H, F;               // 计算得到的单应矩阵和基础矩阵

    thread threadF(&Construct2::FindFundamental,this,ref(vbMatchesInliersF), ref(SF), ref(F));

    // Wait until both threads have finished等待直到两个线程都结束
    threadF.join();

    float minParallax = 1.0;

    // Try to reconstruct from homography or fundamental depending on the ratio (0.40-0.45)
    // 恢复运动R，T
    return ReconstructF(vbMatchesInliersF,F,mK,R21,t21,vP3D,vbTriangulated,minParallax,50);
}

void Construct2::FindFundamental(vector<bool> &vbMatchesInliers, float &score, cv::Mat &F21)
{
    // Number of putative matches 匹配的对数
    const int N = vbMatchesInliers.size();

    // Normalize coordinates    坐标归一化
    vector<cv::Point2f> vPn1, vPn2;     // 归一化后的坐标
    cv::Mat T1, T2;                                         // 归一化用到的矩阵
    Normalize(cKpts1,vPn1, T1);        // 归一化
    Normalize(cKpts2,vPn2, T2);
    cv::Mat T2t = T2.t();

    // Best Results variables
    score = 0.0;
    vbMatchesInliers = vector<bool>(N,false);

    // Iteration variables
    vector<cv::Point2f> vPn1i(8);
    vector<cv::Point2f> vPn2i(8);
    cv::Mat F21i;
    vector<bool> vbCurrentInliers(N,false);         // 用于离群值检验
    float currentScore;

    // Perform all RANSAC iterations and save the solution with highest score
    for(int it=0; it<mMaxIterations; it++)       // 遍历迭代次数
    {
        // Select a minimum set
        for(int j=0; j<8; j++)
        {
            int idx = mvSets[it][j];                            //取序号

            vPn1i[j] = vPn1[idx];            // 取匹配的两个特征点归一化后的坐标
            vPn2i[j] = vPn2[idx];
        }

        cv::Mat Fn = ComputeF21(vPn1i,vPn2i);                   // 计算基础矩阵F

        F21i = T2t*Fn*T1;                                                                // 归一化恢复

        currentScore = CheckFundamental(F21i, vbCurrentInliers, mSigma);        // 检验质量

        if(currentScore>score)
        {
            F21 = F21i.clone();
            vbMatchesInliers = vbCurrentInliers;
            score = currentScore;
        }
    }
}

void Construct2::Normalize(const at::Tensor kpts, vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T)
{
    float meanX = 0;
    float meanY = 0;
    const int N = kpts.size(0);                 // 特征点向量尺寸

    vNormalizedPoints.resize(N);        // 调整尺寸

    for(int i=0; i<N; i++)                              // 遍历
    {
        meanX += kpts[i][0].item().toFloat();                 // x,y坐标累加
        meanY += kpts[i][1].item().toFloat();
    }

    meanX = meanX/N;                            // x,y坐标平均值
    meanY = meanY/N;

    float meanDevX = 0;                         // 减去平均值之后的x，y坐标绝对值累加
    float meanDevY = 0;

    for(int i=0; i<N; i++)                          // 遍历
    {
        vNormalizedPoints[i].x = kpts[i][0].item().toFloat() - meanX;         // x,y坐标-平均值
        vNormalizedPoints[i].y = kpts[i][1].item().toFloat() - meanY;

        meanDevX += fabs(vNormalizedPoints[i].x);
        meanDevY += fabs(vNormalizedPoints[i].y);
    }

    meanDevX = meanDevX/N;          // 绝对值取平均
    meanDevY = meanDevY/N;

    float sX = 1.0/meanDevX;            // 倒数
    float sY = 1.0/meanDevY;

    for(int i=0; i<N; i++)
    {
        vNormalizedPoints[i].x = vNormalizedPoints[i].x * sX;           // 完成归一化
        vNormalizedPoints[i].y = vNormalizedPoints[i].y * sY;
    }

    T = cv::Mat::eye(3,3,CV_32F);               // 构造一个T矩阵不知道干嘛的
    T.at<float>(0,0) = sX;
    T.at<float>(1,1) = sY;
    T.at<float>(0,2) = -meanX*sX;
    T.at<float>(1,2) = -meanY*sY;
}

cv::Mat Construct2::ComputeF21(const vector<cv::Point2f> &vP1,const vector<cv::Point2f> &vP2)
{
    const int N = vP1.size();       // 计算F的特征点数目

    cv::Mat A(N,9,CV_32F);      // 求解AX=0，A是特征点坐标构成记得矩阵N×9；X是待求的H矩阵，9*1（3*3）

    for(int i=0; i<N; i++)              // A矩阵构造
    {
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;
        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;

        A.at<float>(i,0) = u2*u1;
        A.at<float>(i,1) = u2*v1;
        A.at<float>(i,2) = u2;
        A.at<float>(i,3) = v2*u1;
        A.at<float>(i,4) = v2*v1;
        A.at<float>(i,5) = v2;
        A.at<float>(i,6) = u1;
        A.at<float>(i,7) = v1;
        A.at<float>(i,8) = 1;
    }

    cv::Mat u,w,vt;

    // SVD分解，w：奇异值；u：左特征向量；vt：右奇异值的转置矩阵
    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);
    // 因为vt是正交阵，故vt乘最后一行的转置是[0,...,1]（9*1），而奇异值矩阵w（8*9）中最后一列全是0，所以vt最后一行是AX=0的解
    cv::Mat Fpre = vt.row(8).reshape(0, 3);             // 取vt第九行作为AX=0中X的解（1*9），转换成（3*3）

    // 对得到的F矩阵再次SVD分解，并将计算结果的第三个特征值设为0
    cv::SVDecomp(Fpre,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    w.at<float>(2)=0;

    return  u*cv::Mat::diag(w)*vt;      // 返回最终的F矩阵
}

float Construct2::CheckFundamental(const cv::Mat &F21, vector<bool> &vbMatchesInliers, float sigma)
{
    const int N = cKpts1.size(0);               // 匹配的对数

    const float f11 = F21.at<float>(0,0);           // 取基础矩阵的9个值
    const float f12 = F21.at<float>(0,1);
    const float f13 = F21.at<float>(0,2);
    const float f21 = F21.at<float>(1,0);
    const float f22 = F21.at<float>(1,1);
    const float f23 = F21.at<float>(1,2);
    const float f31 = F21.at<float>(2,0);
    const float f32 = F21.at<float>(2,1);
    const float f33 = F21.at<float>(2,2);

    vbMatchesInliers.resize(N);                         // 检测各个匹配对通过与否

    float score = 0;                // F12的得分

    const float th = 3.841;     // 设定的阈值
    const float thScore = 5.991;        // 得分阈值

    const float invSigmaSquare = 1.0/(sigma*sigma);     // 标准差平方的倒数

    for(int i=0; i<N; i++)          // 遍历匹配对数
    {
        bool bIn = true;

        const float u1 = cKpts1[i][0].item().toFloat();          // 取到像素坐标
        const float v1 = cKpts1[i][1].item().toFloat();
        const float u2 = cKpts2[i][0].item().toFloat();
        const float v2 = cKpts2[i][1].item().toFloat();

        // Reprojection error in second image
        // l2=F21x1=(a2,b2,c2)
        // 关键帧1投影到2的误差
        const float a2 = f11*u1+f12*v1+f13;
        const float b2 = f21*u1+f22*v1+f23;
        const float c2 = f31*u1+f32*v1+f33;

        const float num2 = a2*u2+b2*v2+c2;                      // 误差，－0没有表示出来

        const float squareDist1 = num2*num2/(a2*a2+b2*b2);              // 为什么/(a2*a2+b2*b2)？？？？？？？

        const float chiSquare1 = squareDist1*invSigmaSquare;            // 残差平方和除以随机项方差的结果服从卡方分布

        if(chiSquare1>th)
            bIn = false;
        else
            score += thScore - chiSquare1;              // 误差越大，得分越低

        // Reprojection error in second image
        // l1 =x2tF21=(a1,b1,c1)
        // 关键帧2投影到1的误差
        const float a1 = f11*u2+f21*v2+f31;
        const float b1 = f12*u2+f22*v2+f32;
        const float c1 = f13*u2+f23*v2+f33;

        const float num1 = a1*u1+b1*v1+c1;

        const float squareDist2 = num1*num1/(a1*a1+b1*b1);

        const float chiSquare2 = squareDist2*invSigmaSquare;

        if(chiSquare2>th)
            bIn = false;
        else
            score += thScore - chiSquare2;

        if(bIn)                                     // 通过两次检验，置为true
            vbMatchesInliers[i]=true;
        else
            vbMatchesInliers[i]=false;
    }

    return score;
}

bool Construct2::ReconstructF(vector<bool> &vbMatchesInliers, cv::Mat &F21, cv::Mat &K,
                            cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated)
{
    int N=0;            // 记录匹配对通过的数量
    for(size_t i=0, iend = vbMatchesInliers.size() ; i<iend; i++)
        if(vbMatchesInliers[i])
            N++;

    // Compute Essential Matrix from Fundamental Matrix计算本质矩阵
    cv::Mat E21 = K.t()*F21*K;

    cv::Mat R1, R2, t;

    // Recover the 4 motion hypotheses
    DecomposeE(E21,R1,R2,t);  

    cv::Mat t1=t;           // 取两个t
    cv::Mat t2=-t;
    cout << R1 << t1 << endl;
    cout << R2 << t2 << endl;

    // Reconstruct with the 4 hyphoteses and check  检查四组情况下3D点在摄像头前方且投影误差小于阈值的个数
    vector<cv::Point3f> vP3D1, vP3D2, vP3D3, vP3D4;                 // 通过验证的3D点坐标（相机1下）           
    vector<bool> vbTriangulated1,vbTriangulated2,vbTriangulated3, vbTriangulated4;          // 记录通过验证且有一定视差角的匹配
    float parallax1,parallax2, parallax3, parallax4;                        // 最小视差角

    int nGood1 = CheckRT(R1,t1,cKpts1,cKpts2,vbMatchesInliers,K, vP3D1, 4.0*mSigma2, vbTriangulated1, parallax1);
    int nGood2 = CheckRT(R2,t1,cKpts1,cKpts2,vbMatchesInliers,K, vP3D2, 4.0*mSigma2, vbTriangulated2, parallax2);
    int nGood3 = CheckRT(R1,t2,cKpts1,cKpts2,vbMatchesInliers,K, vP3D3, 4.0*mSigma2, vbTriangulated3, parallax3);
    int nGood4 = CheckRT(R2,t2,cKpts1,cKpts2,vbMatchesInliers,K, vP3D4, 4.0*mSigma2, vbTriangulated4, parallax4);
    cout << "nGood1: " << nGood1 << endl;
    cout << "nGood2: " << nGood2 << endl;
    cout << "nGood3: " << nGood3 << endl;
    cout << "nGood3: " << nGood4 << endl;

    int maxGood = max(nGood1,max(nGood2,max(nGood3,nGood4)));       // 最多的好匹配个数

    R21 = cv::Mat();
    t21 = cv::Mat();

    int nMinGood = max(static_cast<int>(0.9*N),minTriangulated);            // 阈值

    int nsimilar = 0;                                   // 四种可能性中通过验证匹配数量满足要求的可能性个数
    if(nGood1>0.7*maxGood)
        nsimilar++;
    if(nGood2>0.7*maxGood)
        nsimilar++;
    if(nGood3>0.7*maxGood)
        nsimilar++;
    if(nGood4>0.7*maxGood)
        nsimilar++;

    // If there is not a clear winner or not enough triangulated points reject initialization
    // 成功重投影的数量太少或者没有哪个结果明显胜出，失败返回
    if(maxGood<nMinGood || nsimilar>1)
    {
        return false;
    }

    // If best reconstruction has enough parallax initialize
    if(maxGood==nGood1)             // 按照对照结果给参数赋值
    {   // 最小视差角大于最低阈值
        if(parallax1>minParallax)
        {
            vP3D = vP3D1;
            vbTriangulated = vbTriangulated1;

            R1.copyTo(R21);
            t1.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood2)
    {
        if(parallax2>minParallax)
        {
            vP3D = vP3D2;
            vbTriangulated = vbTriangulated2;

            R2.copyTo(R21);
            t1.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood3)
    {
        if(parallax3>minParallax)
        {
            vP3D = vP3D3;
            vbTriangulated = vbTriangulated3;

            R1.copyTo(R21);
            t2.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood4)
    {
        if(parallax4>minParallax)
        {
            vP3D = vP3D4;
            vbTriangulated = vbTriangulated4;

            R2.copyTo(R21);
            t2.copyTo(t21);
            return true;
        }
    }

    return false;
}

void Construct2::DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t)
{
    cv::Mat u,w,vt;
    cv::SVD::compute(E,w,u,vt);         // SVD分解

    u.col(2).copyTo(t);                 // 分解得到位移t=u的最后一列，没有尺度
    t=t/cv::norm(t);                       // 归一化

    cv::Mat W(3,3,CV_32F,cv::Scalar(0));        // 沿着z轴旋转90度，角轴为[0,0,1]即以Z轴为旋转轴，转动arccos((tr[0,0,1]-1)/2)=90°
    W.at<float>(0,1)=-1;                                        // 矩阵形式
    W.at<float>(1,0)=1;
    W.at<float>(2,2)=1;

    R1 = u*W*vt;                                                       // 两个旋转量，乘-90°
    if(cv::determinant(R1)<0)                             // 行列式小于0就乘个负号（确保行列式值=1）
        R1=-R1;

    R2 = u*W.t()*vt;                                                // 两个旋转量，乘+90°
    if(cv::determinant(R2)<0)
        R2=-R2;
}

int Construct2::CheckRT(const cv::Mat &R, const cv::Mat &t, const at::Tensor Kpts1, const at::Tensor Kpts2,
                       vector<bool> &vbMatchesInliers,
                       const cv::Mat &K, vector<cv::Point3f> &vP3D, float th2, vector<bool> &vbGood, float &parallax)
{
    // Calibration parameters内参
    const float fx = K.at<float>(0,0);
    const float fy = K.at<float>(1,1);
    const float cx = K.at<float>(0,2);
    const float cy = K.at<float>(1,2);

    vbGood = vector<bool>(Kpts1.size(0),false);         // 记录匹配关键点恢复出来3D点是否有效
    vP3D.resize(Kpts1.size(0));

    vector<float> vCosParallax;                                         // 记录误差角余弦值
    vCosParallax.reserve(Kpts1.size(0));

    // Camera 1 Projection Matrix K[I|0]
    cv::Mat P1(3,4,CV_32F,cv::Scalar(0));                       // 投影矩阵P1=K[ I | 0 ]，与世界坐标系相同
    K.copyTo(P1.rowRange(0,3).colRange(0,3));

    cv::Mat O1 = cv::Mat::zeros(3,1,CV_32F);                // 相机1光心在世界坐标系坐标

    // Camera 2 Projection Matrix K[R|t]
    cv::Mat P2(3,4,CV_32F);                                                 // 投影矩阵P2=K[ R | t ]
    R.copyTo(P2.rowRange(0,3).colRange(0,3));
    t.copyTo(P2.rowRange(0,3).col(3));
    P2 = K*P2;

    cv::Mat O2 = -R.t()*t;                                                      // 相机2光心在世界坐标系的坐标

    int nGood=0;                                                                    // 记录质量好的匹配的个数

    for(size_t i=0, iend=Kpts1.size(0);i<iend;i++)          // 遍历匹配对
    {
        if(!vbMatchesInliers[i])            // 没有，跳过
            continue;

        cv::Mat p3dC1;                          // 3D点在世界坐标系的坐标

        Triangulate(Kpts1[i],Kpts2[i],P1,P2,p3dC1);           // 三角化测量坐标

        // isfinite()判断一个浮点数是否是一个有限值，检查算出来的坐标是不是正常
        if(!isfinite(p3dC1.at<float>(0)) || !isfinite(p3dC1.at<float>(1)) || !isfinite(p3dC1.at<float>(2)))
        {
            vbGood[i]=false;
            continue;
        }

        // Check parallax
        cv::Mat normal1 = p3dC1 - O1;               // 相机光心到3D点向量和距离
        float dist1 = cv::norm(normal1);

        cv::Mat normal2 = p3dC1 - O2;               // 相机光心到3D点向量和距离
        float dist2 = cv::norm(normal2);

        float cosParallax = normal1.dot(normal2)/(dist1*dist2);         // 视差角的余弦值

        // Check depth in front of first camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        // 坚持3D点在第一个相机的深度，深度<0且有一定视差角就跳过
        if(p3dC1.at<float>(2)<=0 && cosParallax<0.99998)
            continue;

        // Check depth in front of second camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        // 检查第二个相机
        cv::Mat p3dC2 = R*p3dC1+t;
        // 坚持3D点在第二个相机的深度，深度<0且有一定视差角就跳过
        if(p3dC2.at<float>(2)<=0 && cosParallax<0.99998)
            continue;

        // Check reprojection error in first image计算3D点在第一个图像上的投影误差
        float im1x, im1y;               // 计算投影到相机1的像素坐标
        float invZ1 = 1.0/p3dC1.at<float>(2);
        im1x = fx*p3dC1.at<float>(0)*invZ1+cx;
        im1y = fy*p3dC1.at<float>(1)*invZ1+cy;
        cout << "im1x: " << im1x << "; u1: " << Kpts1[i][0].item().toFloat();
        cout << "im1y: " << im1y << "; v1: " << Kpts1[i][1].item().toFloat() << endl;

        float squareError1 = (im1x-Kpts1[i][0].item().toFloat())*(im1x-Kpts1[i][0].item().toFloat())+
                                                (im1y-Kpts1[i][1].item().toFloat())*(im1y-Kpts1[i][1].item().toFloat());       // 投影误差

        if(squareError1>th2)
            continue;

        // Check reprojection error in second image计算3D点在第二个图像上的投影误差
        float im2x, im2y;               // 计算投影到相机2的像素坐标
        float invZ2 = 1.0/p3dC2.at<float>(2);
        im2x = fx*p3dC2.at<float>(0)*invZ2+cx;
        im2y = fy*p3dC2.at<float>(1)*invZ2+cy;

        float squareError2 = (im2x-Kpts2[i][0].item().toFloat())*(im2x-Kpts2[i][0].item().toFloat())+
                                                (im2y-Kpts2[i][1].item().toFloat())*(im2y-Kpts2[i][1].item().toFloat());       // 投影误差

        if(squareError2>th2)
            continue;

        vCosParallax.push_back(cosParallax);                    // 通过检验，这对匹配点三角化重投影成功，记录误差角余弦值，3D点坐标
        vP3D[i] = cv::Point3f(p3dC1.at<float>(0),p3dC1.at<float>(1),p3dC1.at<float>(2));
        nGood++;

        if(cosParallax<0.99998)                                                 // 有一定视差角
            vbGood[i]=true;                       // 对应位置记为True
    }

    if(nGood>0)
    {
        sort(vCosParallax.begin(),vCosParallax.end());
        // 取出第50个或最大的余弦值，即最小的角度
        size_t idx = min(50,int(vCosParallax.size()-1));
        parallax = acos(vCosParallax[idx])*180/CV_PI;       // 计算视差角
    }
    else
        parallax=0;

    return nGood;
}

void Construct2::Triangulate(const at::Tensor Kpt1, const at::Tensor Kpt2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D)
{
    cv::Mat A(4,4,CV_32F);

    A.row(0) = Kpt1[0].item().toFloat()*P1.row(2)-P1.row(0);            // 解AP=0的一个最小二乘问题，P是3D坐标s
    A.row(1) = Kpt1[1].item().toFloat()*P1.row(2)-P1.row(1);
    A.row(2) = Kpt2[0].item().toFloat()*P2.row(2)-P2.row(0);
    A.row(3) = Kpt2[1].item().toFloat()*P2.row(2)-P2.row(1);

    cv::Mat u,w,vt;
    cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
    x3D = vt.row(3).t();                                                            // x3D=[X,Y,Z,W]
    x3D = x3D.rowRange(0,3)/x3D.at<float>(3);           // 归一化x3D=[X/W,Y/W,Z/W]
}