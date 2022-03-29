#include "Frame.h"
#include "Converter.h"
#include <thread>
#include "sub.h"
#include <time.h>


using namespace std;
using namespace cv;
using namespace cv::line_descriptor;
using namespace Eigen;

namespace Planar_SLAM {
    bool debug = false;

    long unsigned int Frame::nNextId = 0;
    bool Frame::mbInitialComputations = true;
    float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
    float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
    float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

    Frame::Frame() {}

    //Copy Constructor
    Frame::Frame(const Frame &frame)
            : mpORBvocabulary(frame.mpORBvocabulary), mpORBextractorLeft(frame.mpORBextractorLeft),
              mpORBextractorRight(frame.mpORBextractorRight),
              mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()),
              mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
              mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn), mvuRight(frame.mvuRight),
              mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
              mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()),
              mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), mnId(frame.mnId),
              mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
              mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
              mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors),
              mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2),
              mLdesc(frame.mLdesc), NL(frame.NL), mvKeylinesUn(frame.mvKeylinesUn),
              mvpMapLines(frame.mvpMapLines),  mvbLineOutlier(frame.mvbLineOutlier), mvKeyLineFunctions(frame.mvKeyLineFunctions),
              mvDepthLine(frame.mvDepthLine), mvLines3D(frame.mvLines3D), mv3DLineforMap(frame.mv3DLineforMap), dealWithLine(frame.dealWithLine),
              blurNumber(frame.blurNumber), vSurfaceNormal(frame.vSurfaceNormal),
              vVanishingDirection(frame.vVanishingDirection), mVF3DLines(frame.mVF3DLines), mvPlaneCoefficients(frame.mvPlaneCoefficients),
              mbNewPlane(frame.mbNewPlane), mvpMapPlanes(frame.mvpMapPlanes), mnPlaneNum(frame.mnPlaneNum), mvbPlaneOutlier(frame.mvbPlaneOutlier),
              mvpParallelPlanes(frame.mvpParallelPlanes), mvpVerticalPlanes(frame.mvpVerticalPlanes),
              vSurfaceNormalx(frame.vSurfaceNormalx), vSurfaceNormaly(frame.vSurfaceNormaly), vSurfaceNormalz(frame.vSurfaceNormalz),
              vSurfacePointx(frame.vSurfacePointx), vSurfacePointy(frame.vSurfacePointy), vSurfacePointz(frame.vSurfacePointz),
              vVanishingLinex(frame.vVanishingLinex),vVanishingLiney(frame.vVanishingLiney),vVanishingLinez(frame.vVanishingLinez),
              mvPlanePoints(frame.mvPlanePoints) {
        for (int i = 0; i < FRAME_GRID_COLS; i++)
            for (int j = 0; j < FRAME_GRID_ROWS; j++)
                mGrid[i][j] = frame.mGrid[i][j];

        if (!frame.mTcw.empty())
            SetPose(frame.mTcw);
    }

    // Extract points,lines and planes by using a multi-thread manner
    /**
    * @brief 为RGBD相机准备的帧构造函数
    *
    * @param[in] imGray        对RGB图像灰度化之后得到的灰度图像
    * @param[in] imDepth       深度图像
    * @param[in] timeStamp     时间戳
    * @param[in] extractor     特征点提取器句柄
    * @param[in] voc           ORB特征点词典的句柄
    * @param[in] K             相机的内参数矩阵
    * @param[in] distCoef      相机的去畸变参数
    * @param[in] bf            baseline*bf
    * @param[in] thDepth       远点和近点的深度区分阈值
    * @param[in] depthMapFactor 深度相机的转换引子 realsense=5000
    * @param[in] max_merge_dist CAPE平面提取参数
    * @param[in] patch_size CAPE平面提取参数
    */
    Frame::Frame(const cv::Mat &imRGB, const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp,
                 ORBextractor *extractor, ORBVocabulary *voc, cv::Mat &K, cv::Mat &distCoef, const float &bf,
                 const float &thDepth, const float &depthMapFactor,float &max_merge_dist,int &patch_size,float &max_point_dist, YOLOX* Semanticer)
            : mpORBvocabulary(voc), mpORBextractorLeft(extractor),
              mpORBextractorRight(static_cast<ORBextractor *>(NULL)),
              mTimeStamp(timeStamp), mK(K.clone()), mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
              mMax_merge_dist(max_merge_dist),mPatch_size(patch_size),mMax_point_dist(max_point_dist),m_Semanticer(Semanticer){
        // Frame ID
        // Step 1 帧的ID 自增
        mnId = nNextId++;
        planeDetectionCape.MAX_MERGE_DIST = mMax_merge_dist;
        planeDetectionCape.PATCH_SIZE = mPatch_size;

        // Scale Level Info
        // Step 2 计算图像金字塔的参数
        //获取图像金字塔的层数
        mnScaleLevels = mpORBextractorLeft->GetLevels();
        //获取每层的缩放因子
        mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
        //计算每层缩放因子的自然对数
        mfLogScaleFactor = log(mfScaleFactor);
        //获取各层图像的缩放因子
        mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
        //获取各层图像的缩放因子的倒数
        mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
        //TODO 也是获取这个不知道有什么实际含义的sigma^2
        mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
        //计算上面获取的sigma^2的倒数
        mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

        //获取内参
        fx = K.at<float>(0, 0);
        fy = K.at<float>(1, 1);
        cx = K.at<float>(0, 2);
        cy = K.at<float>(1, 2);
        invfx = 1.0f / fx;
        invfy = 1.0f / fy;

        //深度图转换
        if (depthMapFactor != 1 || imDepth.type() != CV_32F) {
            imDepth.convertTo(depth, CV_32F, depthMapFactor);
        }
        // 建立内参矩阵K
        cv::Mat tmpK = (cv::Mat_<double>(3, 3) << fx, 0, cx,
                0, fy, cy,
                0, 0, 1);
        dealWithLine = true;

        // Step 3 对图像进行提取特征 点、线、面
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        thread threadLines(&Planar_SLAM::Frame::ExtractLSD, this, imGray, depth, tmpK);
        thread threadPoints(&Planar_SLAM::Frame::ExtractORB, this, 0, imGray); //参数0-左图， 1-右图
        thread threadPlanes(&Planar_SLAM::Frame::ComputePlanes, this, depth, imDepth, imRGB, K,depthMapFactor);
        thread threadYOLO(&Planar_SLAM::Frame::ExtractObject, this, imRGB);
        // 新增CAPE面提取算法
//        thread threadPlanes_CAPE(&Planar_SLAM::Frame::ComputePlanes_CAPE, this, depth, imRGB, K);
        threadPoints.join();
        threadLines.join();
//        threadPlanes_CAPE.join();
        threadPlanes.join();
        threadYOLO.join();
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        double t12= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

//       cout<<"time-frame-feature:" <<t12<<endl;

        //获取特征点的个数
        N = mvKeys.size();

        std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
        double t32= std::chrono::duration_cast<std::chrono::duration<double> >(t3 - t1).count();
//        cout<<"time-frame-line:" <<t32<<endl;

        //获取线特征的个数
        NL = mvKeylinesUn.size();

        if (mvKeys.empty())
            return;

        // Step 4 用OpenCV的矫正函数、内参对提取到的特征点进行矫正
        UndistortKeyPoints();

        // Step 5 获取图像的深度，并且根据这个深度推算其右图中匹配的特征点的视差
        ComputeStereoFromRGBD(depth);

        // 初始化本帧的地图点和线
        mvpMapPoints = vector<MapPoint *>(N, static_cast<MapPoint *>(NULL));
        mvpMapLines = vector<MapLine *>(NL, static_cast<MapLine *>(NULL));
        // 记录地图点是否为外点，初始化均为外点false
        mvbOutlier = vector<bool>(N, false);
        mvbLineOutlier = vector<bool>(NL, false);

        // This is done only for the first Frame (or after a change in the calibration)
        //  Step 6 计算去畸变后图像边界，将特征点分配到网格中。这个过程一般是在第一帧或者是相机标定参数发生变化之后进行
        if (mbInitialComputations) {
            ComputeImageBounds(imGray);
            // 表示一个图像像素相当于多少个图像网格列（宽）
            mfGridElementWidthInv = static_cast<float>(FRAME_GRID_COLS) / static_cast<float>(mnMaxX - mnMinX);
            // 表示一个图像像素相当于多少个图像网格行（高）
            mfGridElementHeightInv = static_cast<float>(FRAME_GRID_ROWS) / static_cast<float>(mnMaxY - mnMinY);
            //给类的静态成员变量幅值
            fx = K.at<float>(0, 0);
            fy = K.at<float>(1, 1);
            cx = K.at<float>(0, 2);
            cy = K.at<float>(1, 2);
            invfx = 1.0f / fx;
            invfy = 1.0f / fy;

            //特殊的初始化过程完成，标志复位
            mbInitialComputations = false;
        }

        // 计算假想的基线长度 baseline= mbf/fx
        // 后面要对从RGBD相机输入的特征点,结合相机基线长度,焦距,以及点的深度等信息来计算其在假想的"右侧图像"上的匹配点
        mb = mbf / fx;

        // 将特征点分配到图像网格中
        AssignFeaturesToGrid();

        std::chrono::steady_clock::time_point t4 = std::chrono::steady_clock::now();
        double t43= std::chrono::duration_cast<std::chrono::duration<double> >(t4 - t3).count();
        //cout<<"time-frame-plane:" <<t43<<endl;

        // 面特征检测数量
        mnPlaneNum = mvPlanePoints.size();
        //初始化本帧地图的面特征
        mvpMapPlanes = vector<MapPlane *>(mnPlaneNum, static_cast<MapPlane *>(nullptr));
        // 将其分为垂直面和平行面两大类
        mvpParallelPlanes = vector<MapPlane *>(mnPlaneNum, static_cast<MapPlane *>(nullptr));
        mvpVerticalPlanes = vector<MapPlane *>(mnPlaneNum, static_cast<MapPlane *>(nullptr));
        // 平面上的点和线
        // 没有说对面的匹配
        mvPlanePointMatches = vector<vector<MapPoint *>>(mnPlaneNum);
        mvPlaneLineMatches = vector<vector<MapLine *>>(mnPlaneNum);
        // 记录检测到平面是否为外点
        mvbPlaneOutlier = vector<bool>(mnPlaneNum, false);
        mvbVerPlaneOutlier = vector<bool>(mnPlaneNum, false);
        mvbParPlaneOutlier = vector<bool>(mnPlaneNum, false);
#ifdef DEBUG_INFO
        std::cout << "mnPlaneNum: " << mnPlaneNum << std::endl;
#endif

    }

    Frame::Frame( long unsigned int i )
    {
        mnId = i;
        nNextId = i+1;
    }

    void Frame::AssignFeaturesToGrid() {
        int nReserve = 0.5f * N / (FRAME_GRID_COLS * FRAME_GRID_ROWS);
        for (unsigned int i = 0; i < FRAME_GRID_COLS; i++)
            for (unsigned int j = 0; j < FRAME_GRID_ROWS; j++)
                mGrid[i][j].reserve(nReserve);

        for (int i = 0; i < N; i++) {
            const cv::KeyPoint &kp = mvKeysUn[i];

            int nGridPosX, nGridPosY;
            if (PosInGrid(kp, nGridPosX, nGridPosY))
                mGrid[nGridPosX][nGridPosY].push_back(i);
        }
    }

    void Frame::ExtractLSD(const cv::Mat &im, const cv::Mat &depth,cv::Mat K) {
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        //  在图像平面上进行线特征的提取
        mpLineSegment->ExtractLineSegment(im, mvKeylinesUn, mLdesc, mvKeyLineFunctions);
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        double t21= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        // 灭点计算
        // minimal sampling: the number of required samples
//        Vp_Ransac(im,3000,0.1);
        isLineGood(im, depth, K);
        std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
        double t32= std::chrono::duration_cast<std::chrono::duration<double> >(t3 - t2).count();
//        std::cout << "ExtractLSD using time " << t32 + t21 << std::endl;
    }

    float Frame::compute_vote(const Eigen::Vector3d Vp,
                       const vector<Line_normal> Lines,
                       const vector<int> Vp_candadate_index,
                       float threshold_inlier,
                       vector<bool> &VP_outlier)
    {
        // 直线方程 Ax + By + C = 0
        // 点到直线距离|Ax+By+C| / sqrt(A^2+B^2)
        // 灭点已经投影到图像平面中
        float score=0;
        int num_line = Vp_candadate_index.size();
        double x1 = Vp.x();
        double y1 = Vp.y();

        for(int i=0; i< num_line; i++)
        {
            int index = Vp_candadate_index[i];
            double x2 = Lines[index].mid_point.x();
            double y2 = Lines[index].mid_point.y();
            Eigen::Vector2d v1(x1-x2,y1-y2);
            Eigen::Vector3d v2_tmp = Lines[index].start_piont-Lines[index].end_point;
            Eigen::Vector2d v2(v2_tmp.x(),v2_tmp.y());
            v1.normalize();
            v2.normalize();

            float theta = abs(v1.transpose()*v2);

            float  A,B,C;
            A = y1-y2;
            B = x2-x1;
            C = x1*y2-x2*y1;
            Eigen::Vector3d line_coeffis(A,B,C);
            // distance
            //float distance_1,distance_2;
            //distance_1 = abs(line_coeffis.transpose()*Lines[i].start_piont)/sqrt(A*A+B*B);
            //distance_2 = abs(line_coeffis.transpose()*Lines[i].end_point)/sqrt(A*A+B*B);
            // 根据直线之间的夹角确定内外点
            if(theta > 0.999)
            {
                // 属于内点
                 score += (Lines[index].start_piont-Lines[index].end_point).norm();
                // score++;
            }
            else
            {
                VP_outlier[i] = true;
            }
        }
        return score;
    }

    Eigen::Vector3d Frame::VP_estimation(vector<Line_normal> Line_normalize,
                                  vector<bool> &Vp_outlier,
                                  vector<int> Vp_candadate_index,
                                  Eigen::Vector3d model_constrian,
                                  float threshold_inlier,
                                  int num_ransac_iter)
    {
        srand((unsigned)time(NULL));
        Eigen::Vector3d best_model;
        float best_score=-10000;
        int num_candadate = Vp_candadate_index.size();

        for(int ransac_iter =0; ransac_iter<num_ransac_iter; ransac_iter++)
        {
            vector<bool> tmp_outlier = vector<bool>(num_candadate,false);
            int index_1 = rand()%(num_candadate);
            int index_2 = rand()%(num_candadate);
            index_1 = Vp_candadate_index[index_1];
            index_2 = Vp_candadate_index[index_2];

            Eigen::Vector3d np1 = Line_normalize[index_1].normal_vector;
            Eigen::Vector3d np2 = Line_normalize[index_2].normal_vector;

            // 确定灭点
            Eigen::Vector3d vp = np1.cross(np2);
            vp.normalize();
            // 正交约束
            if(model_constrian[2] != -10)
            {
                Eigen::Vector3d tmp_model = model_constrian.normalized();
                double theta = abs(tmp_model.transpose()*vp);
                if(theta>0.1)
                    continue;
            }
            // 投影之图像平面
            vp = vp/vp.z()*fx;
            // 进行投票
            float score = compute_vote(vp,Line_normalize,Vp_candadate_index,threshold_inlier,tmp_outlier);
            if(score > best_score)
            {
                best_score =score;
                cout << "best score "<< best_score<< "   vp "<< vp.transpose()<<endl;
                best_model = vp;
                Vp_outlier = tmp_outlier;
            }
        }

        return best_model;
    }
    void Frame::Draw_vanishing(cv::Mat &im,
                        const vector<Line_normal> Keylines,
                        const vector<bool> Vp_outlier,
                        const vector<int> Vp_candadate_index,
                        const Eigen::Vector3d VP)
    {
        int num_lines = Vp_candadate_index.size();
        int cols = im.cols;
        int rows = im.rows;

        float vp_x = VP.x()+cx;
        float vp_y = VP.y()+cy;

        int x1,y1,x2,y2;

        for(size_t j=0;j<num_lines;j++)
        {
            cout << Vp_outlier[j]<< " ";
            if(!Vp_outlier[j])
            {
                int index= Vp_candadate_index[j];
                x2 = Keylines[index].mid_point.x()+cx;
                y2 = Keylines[index].mid_point.y()+cy;
                x1 = vp_x;
                y1 = vp_y;

                cv::line(im, cv::Point2f(x1,y1),cv::Point2f(x2,y2), cv::Scalar(0, 0, 0),2);
            }
        }
    }
    void Frame::Vp_Ransac(cv::Mat image,int num_ransac_iter, float threshold_inlier)
    {
        int num_lines = mvKeylinesUn.size();
        //归一化成像平面
        vector<Line_normal> Line_normalize;

        // 线段预处理
        // note: 对线段中分布过密集的点进行稀疏化操作，防止误匹配的线段过权重操作
        // 加入yolo后可以单独对动态物体中的线段进行稀疏化操作
        // 这样的好处是可以解决图像尺度问题，像图像中的轮椅而言，远看和近看他的密度不同

        for(int i=0; i<num_lines; i++)
        {
            Line_normal line_tempt;

            line_tempt.start_piont<<(mvKeylinesUn[i].startPointX-cx), (mvKeylinesUn[i].startPointY-cy), fx;

            line_tempt.end_point  << (mvKeylinesUn[i].endPointX-cx), (mvKeylinesUn[i].endPointY-cy), fx;

            line_tempt.mid_point = 0.5*(line_tempt.start_piont+line_tempt.end_point);

            line_tempt.normal_vector = line_tempt.start_piont.cross(line_tempt.end_point);
            line_tempt.normal_vector.normalize();
//            cout << "start_point " << line_tempt.start_piont.transpose()<< "end point "<<
//            line_tempt.end_point.transpose()<< "normal"<<line_tempt.normal_vector.transpose()<<endl;
            Line_normalize.push_back(line_tempt);
        }
        // 去畸变 undistort
        /*
         *
         */
        // 从数据集中随机抽样筛选2个点 确定第一个VP

        //note: 在选取第一个VP时，需要引入IMU的重力轴做候选子参照：不管是跟他平行也好，垂直也罢，总之是要求第一个VP不能找斜了
        Eigen::Vector3d best_model_1;
        Eigen::Vector3d IMU_model(-10,-10,-10);
        vector<int> Vp1_candadate_index;
        vector<bool> Vp1_outlier = vector<bool>(num_lines,false);

        for(int i=0;i<num_lines;i++)
            Vp1_candadate_index.push_back(i);

        best_model_1 = VP_estimation(Line_normalize,Vp1_outlier,Vp1_candadate_index,IMU_model,threshold_inlier,num_ransac_iter);

        // 从线段的中点到灭点画线
        cv::Mat a = image;
        Draw_vanishing(a,Line_normalize,Vp1_outlier,Vp1_candadate_index,best_model_1);

        // 确定第二个灭点
        // 重新确定数据集 vector<Line_normal> Line_normalize
        // 1、在某些极端情况 可能根本没有第二个灭点，如直面一堵墙，
        // 2、又亦或是第一个灭点或第二个灭点并不符合正交，我们需要剔除掉才行

        // 在满足正交约束下寻找第二候选子集---这里可能还有一个问题，就是万一第一个不满足呢？
        // 第一个如果找错了，会严重影响寻找第二个--引入IMU

        // directly incorporating the orthogonality constrint into model estimation step of the RANSAC procedure

        vector<Line_normal> Line_normalize_2;
        vector<int> Vp2_candadate_index;
        vector<bool> Vp2_outlier;
        for(int j=0; j< num_lines; j++)
        {
            if(Vp1_outlier[j])// Vp1的外点作为Vp2的候选集
            {
                Vp2_candadate_index.push_back(j);
                Vp2_outlier.push_back(false);
            }
        }
        //note: 为抑制误匹配，正交的因素因应射到第二个候选集中
        // 也就是说VP2的候选子需要跟我VP1正交才行
        Eigen::Vector3d best_model_2;
        best_model_2 = VP_estimation(Line_normalize,Vp2_outlier,Vp2_candadate_index,best_model_1,threshold_inlier,num_ransac_iter);
        Draw_vanishing(a,Line_normalize,Vp2_outlier,Vp2_candadate_index,best_model_2);

        for(size_t j=0;j<num_lines;j++)
        {
                int u1 = mvKeylinesUn[j].startPointX,v1=mvKeylinesUn[j].startPointY;
                int u2 = mvKeylinesUn[j].endPointX,v2= mvKeylinesUn[j].endPointY;
                cv::line(a, cv::Point2f(u1,v1),cv::Point2f(u2,v2), cv::Scalar(0, 0, 0),4);
        }
        cv::imshow("image",a);
//        cv::waitKey(10);
        // 从其余的外点中寻找第二个灭点
        // note: 在一些极端环境中可能会出现直面一堵墙，三两条线的情况，这种情况比较极端(暂不考虑 先调通程序)
        //
    }

    void Frame::ExtractORB(int flag, const cv::Mat &im) {
        if (flag == 0)
            (*mpORBextractorLeft)(im, cv::Mat(), mvKeys, mDescriptors);
        else
            (*mpORBextractorRight)(im, cv::Mat(), mvKeysRight, mDescriptorsRight);
    }

    // Optimize Lines
    void Frame::isLineGood(const cv::Mat &imGray, const cv::Mat &imDepth, cv::Mat K) {
        mvDepthLine = vector<float>(mvKeylinesUn.size(), -1.0f);
        // 根据图像上的Keylines，将线从2D映射到3D
        mvLines3D = vector<Vector6d>(mvKeylinesUn.size(), Eigen::Matrix<double, 6, 1>::Zero());

        for (int i = 0; i < mvKeylinesUn.size(); ++i) { // each line
            double len = cv::norm(mvKeylinesUn[i].getStartPoint() - mvKeylinesUn[i].getEndPoint());
            vector<cv::Point3d> pts3d;
            // iterate through a line
            // 根据线段的长度进行采样
            double numSmp = (double) min((int) len, 50); //number of line points sampled
            pts3d.reserve(numSmp);
            for (int j = 0; j <= numSmp; ++j) {
                // use nearest neighbor to querry depth value
                // assuming position (0,0) is the top-left corner of image, then the
                // top-left pixel's center would be (0.5,0.5)
                cv::Point2d pt = mvKeylinesUn[i].getStartPoint() * (1 - j / numSmp) +
                                 mvKeylinesUn[i].getEndPoint() * (j / numSmp);

                if (pt.x < 0 || pt.y < 0 || pt.x >= imDepth.cols || pt.y >= imDepth.rows) continue;
                int row, col; // nearest pixel for pt
                if ((floor(pt.x) == pt.x) && (floor(pt.y) == pt.y)) { // boundary issue
                    col = max(int(pt.x - 1), 0);
                    row = max(int(pt.y - 1), 0);
                } else {
                    col = int(pt.x);
                    row = int(pt.y);
                }

                float d = -1;
                if (imDepth.at<float>(row, col) <= 0.01) { // no depth info
                    continue;
                } else {
                    d = imDepth.at<float>(row, col);
                }
                cv::Point3d p;
                //投射至3D空间[u,v]->[x,y,z]
                p.z = d;
                p.x = (col - cx) * p.z * invfx;
                p.y = (row - cy) * p.z * invfy;

                pts3d.push_back(p);
            }

            if (pts3d.size() < 10.0)  // 线段中间的有效采样点太少 pass
                continue;

            RandomLine3d tmpLine;
            vector<RandomPoint3d> rndpts3d;
            rndpts3d.reserve(pts3d.size());
            // compute uncertainty of 3d points
            for (int j = 0; j < pts3d.size(); ++j) {
                rndpts3d.push_back(compPt3dCov(pts3d[j], K, 1));
            }
            // using ransac to extract a 3d line from 3d pts
            tmpLine = extract3dline_mahdist(rndpts3d);

            if (tmpLine.pts.size() / len > 0.4 && cv::norm(tmpLine.A - tmpLine.B) > 0.02) {
                //this line is reliable
                mvDepthLine[i] = std::min(imDepth.at<float>(mvKeylinesUn[i].endPointY, mvKeylinesUn[i].endPointX),
                                          imDepth.at<float>(mvKeylinesUn[i].startPointY, mvKeylinesUn[i].startPointX));

                FrameLine tempLine;
                tempLine.haveDepth = true;
                tempLine.rndpts3d = tmpLine.pts;
                tempLine.direction = tmpLine.director;
                tempLine.direct1 = tmpLine.direct1;
                tempLine.direct2 = tmpLine.direct2;
                tempLine.p = Point2d(mvKeylinesUn[i].endPointX, mvKeylinesUn[i].endPointY);
                tempLine.q = Point2d(mvKeylinesUn[i].startPointX, mvKeylinesUn[i].startPointY);
                mVF3DLines.push_back(tempLine);

                mvLines3D[i] << tmpLine.A.x, tmpLine.A.y, tmpLine.A.z, tmpLine.B.x, tmpLine.B.y, tmpLine.B.z;
            }

        }

    }

    void Frame::lineDescriptorMAD(vector<vector<DMatch>> line_matches, double &nn_mad, double &nn12_mad) const {
        vector<vector<DMatch>> matches_nn, matches_12;
        matches_nn = line_matches;
        matches_12 = line_matches;

        // estimate the NN's distance standard deviation
        double nn_dist_median;
        sort(matches_nn.begin(), matches_nn.end(), compare_descriptor_by_NN_dist());
        nn_dist_median = matches_nn[int(matches_nn.size() / 2)][0].distance;

        for (unsigned int i = 0; i < matches_nn.size(); i++)
            matches_nn[i][0].distance = fabsf(matches_nn[i][0].distance - nn_dist_median);
        sort(matches_nn.begin(), matches_nn.end(), compare_descriptor_by_NN_dist());
        nn_mad = 1.4826 * matches_nn[int(matches_nn.size() / 2)][0].distance;

        // estimate the NN's 12 distance standard deviation
        double nn12_dist_median;
        sort(matches_12.begin(), matches_12.end(), conpare_descriptor_by_NN12_dist());
        nn12_dist_median =
                matches_12[int(matches_12.size() / 2)][1].distance - matches_12[int(matches_12.size() / 2)][0].distance;
        for (unsigned int j = 0; j < matches_12.size(); j++)
            matches_12[j][0].distance = fabsf(matches_12[j][1].distance - matches_12[j][0].distance - nn12_dist_median);
        sort(matches_12.begin(), matches_12.end(), compare_descriptor_by_NN_dist());
        nn12_mad = 1.4826 * matches_12[int(matches_12.size() / 2)][0].distance;
    }

    void Frame::SetPose(cv::Mat Tcw) {
        mTcw = Tcw.clone();
        UpdatePoseMatrices();
    }

    void Frame::UpdatePoseMatrices() {
        mRcw = mTcw.rowRange(0, 3).colRange(0, 3);
        mRwc = mRcw.t();
        mtcw = mTcw.rowRange(0, 3).col(3);
        mOw = -mRcw.t() * mtcw;

        mTwc = cv::Mat::eye(4, 4, mTcw.type());
        mRwc.copyTo(mTwc.rowRange(0, 3).colRange(0, 3));
        mOw.copyTo(mTwc.rowRange(0, 3).col(3));
    }

    bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit) {
        pMP->mbTrackInView = false;

        // 3D in absolute coordinates
        cv::Mat P = pMP->GetWorldPos();

        // 3D in camera coordinates
        const cv::Mat Pc = mRcw * P + mtcw;
        const float &PcX = Pc.at<float>(0);
        const float &PcY = Pc.at<float>(1);
        const float &PcZ = Pc.at<float>(2);

        // Check positive depth
        if (PcZ < 0.0f)
            return false;

        // Project in image and check it is not outside
        const float invz = 1.0f / PcZ;
        const float u = fx * PcX * invz + cx;
        const float v = fy * PcY * invz + cy;

        if (u < mnMinX || u > mnMaxX)
            return false;
        if (v < mnMinY || v > mnMaxY)
            return false;

        // Check distance is in the scale invariance region of the MapPoint
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const cv::Mat PO = P - mOw;
        const float dist = cv::norm(PO);

        if (dist < minDistance || dist > maxDistance)
            return false;

        // Check viewing angle
        cv::Mat Pn = pMP->GetNormal();

        const float viewCos = PO.dot(Pn) / dist;

        if (viewCos < viewingCosLimit)
            return false;

        // Predict scale in the image
        const int nPredictedLevel = pMP->PredictScale(dist, this);

        // Data used by the tracking
        pMP->mbTrackInView = true;
        pMP->mTrackProjX = u;
        pMP->mTrackProjXR = u - mbf * invz;
        pMP->mTrackProjY = v;
        pMP->mnTrackScaleLevel = nPredictedLevel;
        pMP->mTrackViewCos = viewCos;

        return true;
    }

    bool Frame::isInFrustum(MapLine *pML, float viewingCosLimit) {
        pML->mbTrackInView = false;

        Vector6d P = pML->GetWorldPos();

        cv::Mat SP = (Mat_<float>(3, 1) << P(0), P(1), P(2));
        cv::Mat EP = (Mat_<float>(3, 1) << P(3), P(4), P(5));

        const cv::Mat SPc = mRcw * SP + mtcw;
        const float &SPcX = SPc.at<float>(0);
        const float &SPcY = SPc.at<float>(1);
        const float &SPcZ = SPc.at<float>(2);

        const cv::Mat EPc = mRcw * EP + mtcw;
        const float &EPcX = EPc.at<float>(0);
        const float &EPcY = EPc.at<float>(1);
        const float &EPcZ = EPc.at<float>(2);

        if (SPcZ < 0.0f || EPcZ < 0.0f)
            return false;

        const float invz1 = 1.0f / SPcZ;
        const float u1 = fx * SPcX * invz1 + cx;
        const float v1 = fy * SPcY * invz1 + cy;

        if (u1 < mnMinX || u1 > mnMaxX)
            return false;
        if (v1 < mnMinY || v1 > mnMaxY)
            return false;

        const float invz2 = 1.0f / EPcZ;
        const float u2 = fx * EPcX * invz2 + cx;
        const float v2 = fy * EPcY * invz2 + cy;

        if (u2 < mnMinX || u2 > mnMaxX)
            return false;
        if (v2 < mnMinY || v2 > mnMaxY)
            return false;


        const float maxDistance = pML->GetMaxDistanceInvariance();
        const float minDistance = pML->GetMinDistanceInvariance();

        const cv::Mat OM = 0.5 * (SP + EP) - mOw;
        const float dist = cv::norm(OM);

        if (dist < minDistance || dist > maxDistance)
            return false;


        Vector3d Pn = pML->GetNormal();
        cv::Mat pn = (Mat_<float>(3, 1) << Pn(0), Pn(1), Pn(2));
        const float viewCos = OM.dot(pn) / dist;

        if (viewCos < viewingCosLimit)
            return false;

        const int nPredictedLevel = pML->PredictScale(dist, mfLogScaleFactor);

        pML->mbTrackInView = true;
        pML->mTrackProjX1 = u1;
        pML->mTrackProjY1 = v1;
        pML->mTrackProjX2 = u2;
        pML->mTrackProjY2 = v2;
        pML->mnTrackScaleLevel = nPredictedLevel;
        pML->mTrackViewCos = viewCos;

        return true;
    }


    vector<size_t> Frame::GetFeaturesInArea(const float &x, const float &y, const float &r, const int minLevel,
                                            const int maxLevel) const {
        vector<size_t> vIndices;
        vIndices.reserve(N);

        const int nMinCellX = max(0, (int) floor((x - mnMinX - r) * mfGridElementWidthInv));
        if (nMinCellX >= FRAME_GRID_COLS)
            return vIndices;

        const int nMaxCellX = min((int) FRAME_GRID_COLS - 1, (int) ceil((x - mnMinX + r) * mfGridElementWidthInv));
        if (nMaxCellX < 0)
            return vIndices;

        const int nMinCellY = max(0, (int) floor((y - mnMinY - r) * mfGridElementHeightInv));
        if (nMinCellY >= FRAME_GRID_ROWS)
            return vIndices;

        const int nMaxCellY = min((int) FRAME_GRID_ROWS - 1, (int) ceil((y - mnMinY + r) * mfGridElementHeightInv));
        if (nMaxCellY < 0)
            return vIndices;

        const bool bCheckLevels = (minLevel > 0) || (maxLevel >= 0);

        for (int ix = nMinCellX; ix <= nMaxCellX; ix++) {
            for (int iy = nMinCellY; iy <= nMaxCellY; iy++) {
                const vector<size_t> vCell = mGrid[ix][iy];
                if (vCell.empty())
                    continue;

                for (size_t j = 0, jend = vCell.size(); j < jend; j++) {
                    const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                    if (bCheckLevels) {
                        if (kpUn.octave < minLevel)
                            continue;
                        if (maxLevel >= 0)
                            if (kpUn.octave > maxLevel)
                                continue;
                    }

                    const float distx = kpUn.pt.x - x;
                    const float disty = kpUn.pt.y - y;

                    if (fabs(distx) < r && fabs(disty) < r)
                        vIndices.push_back(vCell[j]);
                }
            }
        }

        return vIndices;
    }

    vector<size_t>  Frame::GetLinesInArea(const float &x1, const float &y1, const float &x2, const float &y2, const float &r,
                          const int minLevel, const int maxLevel) const {
        vector<size_t> vIndices;

        vector<KeyLine> vkl = this->mvKeylinesUn;

        const bool bCheckLevels = (minLevel > 0) || (maxLevel > 0);

        for (size_t i = 0; i < vkl.size(); i++) {
            KeyLine keyline = vkl[i];

            // 1.对比中点距离
            float distance = (0.5 * (x1 + x2) - keyline.pt.x) * (0.5 * (x1 + x2) - keyline.pt.x) +
                             (0.5 * (y1 + y2) - keyline.pt.y) * (0.5 * (y1 + y2) - keyline.pt.y);
            if (distance > r * r)
                continue;

            float slope = (y1 - y2) / (x1 - x2) - keyline.angle;
            if (slope > r * 0.01)
                continue;

            if (bCheckLevels) {
                if (keyline.octave < minLevel)
                    continue;
                if (maxLevel >= 0 && keyline.octave > maxLevel)
                    continue;
            }

            vIndices.push_back(i);
        }

        return vIndices;
    }


    bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY) {
        posX = round((kp.pt.x - mnMinX) * mfGridElementWidthInv);
        posY = round((kp.pt.y - mnMinY) * mfGridElementHeightInv);

        //Keypoint's coordinates are undistorted, which could cause to go out of the image
        if (posX < 0 || posX >= FRAME_GRID_COLS || posY < 0 || posY >= FRAME_GRID_ROWS)
            return false;

        return true;
    }


    void Frame::ComputeBoW() {
        if (mBowVec.empty()) {
            vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
            mpORBvocabulary->transform(vCurrentDesc, mBowVec, mFeatVec, 4);
        }
    }

    void Frame::UndistortKeyPoints() {
        if (mDistCoef.at<float>(0) == 0.0) {
            mvKeysUn = mvKeys;
            return;
        }

        // Fill matrix with points
        cv::Mat mat(N, 2, CV_32F);
        for (int i = 0; i < N; i++) {
            mat.at<float>(i, 0) = mvKeys[i].pt.x;
            mat.at<float>(i, 1) = mvKeys[i].pt.y;
        }

        // Undistort points
        mat = mat.reshape(2);
        cv::undistortPoints(mat, mat, mK, mDistCoef, cv::Mat(), mK);
        mat = mat.reshape(1);

        // Fill undistorted keypoint vector
        mvKeysUn.resize(N);
        for (int i = 0; i < N; i++) {
            cv::KeyPoint kp = mvKeys[i];
            kp.pt.x = mat.at<float>(i, 0);
            kp.pt.y = mat.at<float>(i, 1);
            mvKeysUn[i] = kp;
        }
    }

    void Frame::ComputeImageBounds(const cv::Mat &imLeft) {
        if (mDistCoef.at<float>(0) != 0.0) {
            cv::Mat mat(4, 2, CV_32F);
            mat.at<float>(0, 0) = 0.0;
            mat.at<float>(0, 1) = 0.0;
            mat.at<float>(1, 0) = imLeft.cols;
            mat.at<float>(1, 1) = 0.0;
            mat.at<float>(2, 0) = 0.0;
            mat.at<float>(2, 1) = imLeft.rows;
            mat.at<float>(3, 0) = imLeft.cols;
            mat.at<float>(3, 1) = imLeft.rows;

            // Undistort corners
            mat = mat.reshape(2);
            cv::undistortPoints(mat, mat, mK, mDistCoef, cv::Mat(), mK);
            mat = mat.reshape(1);

            mnMinX = min(mat.at<float>(0, 0), mat.at<float>(2, 0));
            mnMaxX = max(mat.at<float>(1, 0), mat.at<float>(3, 0));
            mnMinY = min(mat.at<float>(0, 1), mat.at<float>(1, 1));
            mnMaxY = max(mat.at<float>(2, 1), mat.at<float>(3, 1));

        } else {
            mnMinX = 0.0f;
            mnMaxX = imLeft.cols;
            mnMinY = 0.0f;
            mnMaxY = imLeft.rows;
        }
    }

    void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth) {
        mvuRight = vector<float>(N, -1);
        mvDepth = vector<float>(N, -1);

        for (int i = 0; i < N; i++) {
            const cv::KeyPoint &kp = mvKeys[i];
            const cv::KeyPoint &kpU = mvKeysUn[i];

            const float &v = kp.pt.y;
            const float &u = kp.pt.x;

            const float d = imDepth.at<float>(v, u);

            if (d > 0) {
                mvDepth[i] = d;
                mvuRight[i] = kpU.pt.x - mbf / d;
            }
        }
    }

    cv::Mat Frame::UnprojectStereo(const int &i) {
        const float z = mvDepth[i];
        if (z > 0) {
            const float u = mvKeysUn[i].pt.x;
            const float v = mvKeysUn[i].pt.y;
            const float x = (u - cx) * z * invfx;
            const float y = (v - cy) * z * invfy;
            cv::Mat x3Dc = (cv::Mat_<float>(3, 1) << x, y, z);
            return mRwc * x3Dc + mOw;
        } else
            return cv::Mat();
    }

    Vector6d Frame::obtain3DLine(const int &i) {
        Vector6d Lines3D = mvLines3D[i];
        cv::Mat Ac = (Mat_<float>(3, 1) << Lines3D(0), Lines3D(1), Lines3D(2));
        cv::Mat A = mRwc * Ac + mOw;
        cv::Mat Bc = (Mat_<float>(3, 1) << Lines3D(3), Lines3D(4), Lines3D(5));
        cv::Mat B = mRwc * Bc + mOw;
        Lines3D << A.at<float>(0, 0), A.at<float>(1, 0), A.at<float>(2, 0),
                B.at<float>(0, 0), B.at<float>(1,0), B.at<float>(2, 0);
        return Lines3D;
    }

    void Frame::ComputePlanes(const cv::Mat &imDepth, const cv::Mat &Depth, const cv::Mat &imRGB, cv::Mat K, const float depthFactor) {
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        planeDetector.readColorImage(imRGB);
        planeDetector.readDepthImage(Depth, K,depthFactor);
        std::chrono::steady_clock::time_point t11 = std::chrono::steady_clock::now();
        planeDetector.runPlaneDetection();
        std::chrono::steady_clock::time_point t12 = std::chrono::steady_clock::now();
        double t121= std::chrono::duration_cast<std::chrono::duration<double> >(t12 - t11).count();
//        std::cout << "Detection Plane Using time: " << t121 << std::endl;
#ifdef DEBUG_INFO
        PointCloud::Ptr viewCloud(new PointCloud());
#endif
        seg_out = planeDetector.seg_output;
        // 遍历提取到的平面
        int fail_planes = 0;
        for (int i = 0; i < planeDetector.plane_num_; i++) {

            auto &indices = planeDetector.plane_vertices_[i];
            // 遍历每平面上的点云
            PointCloud::Ptr inputCloud(new PointCloud());

            for (int j : indices) {
                PointT p;
                p.x = (float) planeDetector.cloud.vertices[j][0];
                p.y = (float) planeDetector.cloud.vertices[j][1];
                p.z = (float) planeDetector.cloud.vertices[j][2];
                if(p.z > mMax_point_dist) continue;//  在这里做了个距离限定
                // 插入点云
                inputCloud->points.push_back(p);
#ifdef DEBUG_INFO
//                viewCloud->points.push_back(p);
#endif
            }

            auto extractedPlane = planeDetector.plane_filter.extractedPlanes[i];
            double nx = extractedPlane->normal[0];
            double ny = extractedPlane->normal[1];
            double nz = extractedPlane->normal[2];
            double cx = extractedPlane->center[0];
            double cy = extractedPlane->center[1];
            double cz = extractedPlane->center[2];

            float d = (float) -(nx * cx + ny * cy + nz * cz);

            pcl::VoxelGrid<PointT> voxel;
            voxel.setLeafSize(0.05, 0.05, 0.05);
//            voxel.setLeafSize(0.01, 0.01, 0.01);
            PointCloud::Ptr coarseCloud(new PointCloud());
            voxel.setInputCloud(inputCloud);
            voxel.filter(*coarseCloud);
//             忽略远处平面
            /*cout << "d "<< d<<"  "<< "size "<<  coarseCloud->size()<<endl;*/
            if(d > mMax_point_dist||coarseCloud->size() < 100)
            {
                fail_planes++;
                continue;
            }

            cv::Mat coef = (cv::Mat_<float>(4, 1) << nx, ny, nz, d);

            bool valid = MaxPointDistanceFromPlane(coef, coarseCloud);

            if (!valid) {
                cout << "################################"<<endl;
                continue;
            }
#ifdef DEBUG_INFO
            std::cout << "extractedPlane normal & center & d" << nx<<" "<<ny<<" "<<nz<<" "<<cx<<" "<<cy<<" "<<cz<<" " << d << std::endl;
            boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
            pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZRGB> fildColor(coarseCloud, "z");
            viewer->addPointCloud<pcl::PointXYZRGB>(coarseCloud, fildColor, "sample cloud");
            viewer->spinOnce(1000);
#endif
            //  将滤波后的点云放置到mvPlanePoints中，在我的理解可能是以此来代表平面的大小
            mvPlanePoints.push_back(*coarseCloud);

            mvPlaneCoefficients.push_back(coef);
        }
#ifdef DEBUG_INFO
        boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer1(new pcl::visualization::PCLVisualizer("3D Viewer"));
        pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZRGB> fildColor(viewCloud, "z");
        viewer1->addPointCloud<pcl::PointXYZRGB>(viewCloud, fildColor, "sample cloud");
        viewer1->spinOnce(1000);
#endif
//        cout <<" planeDetector.plane_num_" << planeDetector.plane_num_<<endl;
        planeDetector.plane_num_-=fail_planes;
//        cout <<" planeDetector.plane_num_ -=fail_planes " << planeDetector.plane_num_<<endl;
        // 这里进行了点云的法向量计算，不清楚是用在何处--- 在FindManhattan中用到
        std::vector<SurfaceNormal> surfaceNormals;

        PointCloud::Ptr inputCloud( new PointCloud() );
        // 这里做了滤波处理
        for (int m=0; m<imDepth.rows; m+=3)
        {
            for (int n=0; n<imDepth.cols; n+=3)
            {
                float d = imDepth.ptr<float>(m)[n];
                PointT p;

                if(d > mMax_point_dist)//对平面点云距离进行限定
                    p.z = 0;
                else
                    p.z = d;
                //cout << "depth:" << d<<endl;
                p.x = ( n - cx) * p.z / fx;
                p.y = ( m - cy) * p.z / fy;

                inputCloud->points.push_back(p);
            }
        }
        inputCloud->height = ceil(imDepth.rows/3.0);
        inputCloud->width = ceil(imDepth.cols/3.0);

            //compute normals
        pcl::IntegralImageNormalEstimation<PointT, pcl::Normal> ne;
        pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
        ne.setNormalEstimationMethod(ne.AVERAGE_3D_GRADIENT);
        //计算边界时深度改变的阈值
        ne.setMaxDepthChangeFactor(0.05f);
        // 设置法向量的平滑patch
        ne.setNormalSmoothingSize(10.0f);

        ne.setInputCloud(inputCloud);
        //计算特征值

        if (inputCloud->size()== 0)
        {
            PCL_ERROR ("Could not estimate a planar model for the given initial plane.\n");
            return;
        }
        ne.compute(*cloud_normals);

        for ( int m=0; m<inputCloud->height; m+=1 ) {
            if(m%2==0) continue;
            for (int n = 0; n < inputCloud->width; n+=1) {
                pcl::Normal normal = cloud_normals->at(n, m);
                SurfaceNormal surfaceNormal;
                if(n%2==0) continue;
                surfaceNormal.normal.x = normal.normal_x;
                surfaceNormal.normal.y = normal.normal_y;
                surfaceNormal.normal.z = normal.normal_z;

                pcl::PointXYZRGB point = inputCloud->at(n, m);
                surfaceNormal.cameraPosition.x = point.x;
                surfaceNormal.cameraPosition.y = point.y;
                surfaceNormal.cameraPosition.z = point.z;
                surfaceNormal.FramePosition.x = n*3;
                surfaceNormal.FramePosition.y = m*3;

                surfaceNormals.push_back(surfaceNormal);
            }
        }

        vSurfaceNormal = surfaceNormals;
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        double t21= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
//        std::cout << "Extract Plane Using time: " << t21 << std::endl;
    }

    void Frame::ComputePlanes_CAPE(const cv::Mat &depth, const cv::Mat &imGrey, cv::Mat K)
    {
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        planeDetectionCape.readColorImage(imGrey);
        planeDetectionCape.readDepthImage(depth, K);
        std::chrono::steady_clock::time_point t11 = std::chrono::steady_clock::now();
        planeDetectionCape.runPlaneDetection();
        std::chrono::steady_clock::time_point t12 = std::chrono::steady_clock::now();
        double t121= std::chrono::duration_cast<std::chrono::duration<double> >(t12 - t11).count();
//        std::cout << "Detection Plane Using time: " << t121 << std::endl;
        // 遍历提取到的平面
        int fail_planes = 0;
//        cout <<"planeDetectionCape.nr_planes " << planeDetectionCape.nr_planes << endl;
//        cout <<"plane cloud " << planeDetectionCape.plane_cloud.size()<< endl;
        seg_out = planeDetectionCape.seg_output;
        for(int i=0; i< planeDetectionCape.nr_planes; i++)
        {
            PointCloud::Ptr inputCloud(new PointCloud());
            inputCloud = planeDetectionCape.plane_cloud[i];

            double nx = planeDetectionCape.plane_params[i].normal[0];
            double ny = planeDetectionCape.plane_params[i].normal[1];
            double nz = planeDetectionCape.plane_params[i].normal[2];
            double d =  planeDetectionCape.plane_params[i].d;

            pcl::VoxelGrid<PointT> voxel;
            voxel.setLeafSize(0.05, 0.05, 0.05);
            PointCloud::Ptr coarseCloud(new PointCloud());
            voxel.setInputCloud(inputCloud);
            voxel.filter(*coarseCloud);
            if(d > mMax_point_dist||coarseCloud->size() < 100)
            {
                fail_planes++;
                continue;
            }
            cv::Mat coef = (cv::Mat_<float>(4, 1) << nx, ny, nz, d);
            bool valid = MaxPointDistanceFromPlane(coef, coarseCloud);
            if (!valid) {
                cout << "################################"<<endl;
                fail_planes++;
                continue;
            }
//            cout <<"coarseCloud->points.size() :" <<coarseCloud->points.size()<< endl;
            mvPlanePoints.push_back(*coarseCloud);
            mvPlaneCoefficients.push_back(coef);
        }
//        cout <<" planeDetector.plane_num_" << planeDetectionCape.nr_planes<<endl;
        planeDetectionCape.nr_planes-=fail_planes;
//        cout <<" planeDetector.plane_num_ -=fail_planes " << planeDetectionCape.nr_planes<<endl;
        if(planeDetectionCape.nr_planes == 0)
            cout << "*************************************************************************"<< endl;

        std::vector<SurfaceNormal> surfaceNormals;
        PointCloud::Ptr inputCloud( new PointCloud() );

        // 这里做了滤波处理
        // note: 不要将深度图的点云都投过去，可以单独投射平面上的点云
        for (int m=0; m<depth.rows; m+=3)
        {
            for (int n=0; n<depth.cols; n+=3)
            {
                float d = depth.ptr<float>(m)[n];
                PointT p;

                if(d > mMax_point_dist)//对平面点云距离进行限定
                    p.z = 0;
                else
                    p.z = d;
                //cout << "depth:" << d<<endl;
                p.x = ( n - cx) * p.z / fx;
                p.y = ( m - cy) * p.z / fy;

                inputCloud->points.push_back(p);
            }
        }
        inputCloud->height = ceil(depth.rows/3.0);
        inputCloud->width = ceil(depth.cols/3.0);

        //compute normals
        // 通过pcl 估计点云的法向量
        pcl::IntegralImageNormalEstimation<PointT, pcl::Normal> ne;
        pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
        ne.setNormalEstimationMethod(ne.AVERAGE_3D_GRADIENT);

        //计算边界时深度改变的阈值
        ne.setMaxDepthChangeFactor(0.05f);
        // 设置法向量的平滑patch
        ne.setNormalSmoothingSize(10.0f);

        ne.setInputCloud(inputCloud);
        //计算特征值

        if (inputCloud->size()== 0)
        {
            PCL_ERROR ("Could not estimate a planar model for the given initial plane.\n");
            return;
        }
        ne.compute(*cloud_normals);

        for ( int m=0; m<inputCloud->height; m+=1 ) {
            if(m%2==0) continue;
            for (int n = 0; n < inputCloud->width; n+=1) {
                pcl::Normal normal = cloud_normals->at(n, m);
                SurfaceNormal surfaceNormal;
                if(n%2==0) continue;
                surfaceNormal.normal.x = normal.normal_x;
                surfaceNormal.normal.y = normal.normal_y;
                surfaceNormal.normal.z = normal.normal_z;

                pcl::PointXYZRGB point = inputCloud->at(n, m);
                surfaceNormal.cameraPosition.x = point.x;
                surfaceNormal.cameraPosition.y = point.y;
                surfaceNormal.cameraPosition.z = point.z;
                surfaceNormal.FramePosition.x = n*3;
                surfaceNormal.FramePosition.y = m*3;

                surfaceNormals.push_back(surfaceNormal);
            }
        }

        vSurfaceNormal = surfaceNormals;
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        double t21= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();
//        std::cout << "Extract Plane Using time: " << t21 << std::endl;
    }

    bool Frame::MaxPointDistanceFromPlane(cv::Mat &plane, PointCloud::Ptr pointCloud) {
        auto disTh = Config::Get<double>("Plane.DistanceThreshold");
        bool erased = false;
//        double max = -1;
        int i = 0;
        auto &points = pointCloud->points;
//        std::cout << "points before: " << points.size() << std::endl;
// 我要做以下更改： 不能以单一点来否定这个平面 要按照一定的比例来处理
        int point_num = points.size();
        int point_bad = 0;
        for (auto &p : points) {
            double absDis = abs(plane.at<float>(0) * p.x +
                                plane.at<float>(1) * p.y +
                                plane.at<float>(2) * p.z +
                                plane.at<float>(3));

            if (absDis > disTh)
            {
                cout << "absDis "<<absDis << endl;
                return false;
            }
//            if(absDis > (disTh-0.02))
//            {
//                auto it = pointCloud->points.begin()+i;
//                pointCloud->points.erase(it);
//                i++;
//                point_bad++;
//                continue;
//            }
//                auto it = pointCloud->points.begin()+i;
//                pointCloud->points.erase(it);
//                i++;
//                point_bad++;
//                continue;
//                return false;
//            }

            i++;
        }
//        std::cout << "points after: " << points.size() << std::endl;
//        float bad_rate = float(point_bad)/point_num;
//        cout << "bad_rate " <<bad_rate<<endl;
//        if(bad_rate > 0.15)
//            return false;

             // 根据点云重新更新平面参数
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        // Create the segmentation object
        pcl::SACSegmentation<PointT> seg;
        // Optional
        seg.setOptimizeCoefficients(true);
        // Mandatory
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(disTh);

        seg.setInputCloud(pointCloud);
        seg.segment(*inliers, *coefficients);
        if (inliers->indices.size () == 0)
        {
            PCL_ERROR ("Could not estimate a planar model for the given initial plane.\n");
            return false;
        }

        float oldVal = plane.at<float>(3);
        float newVal = coefficients->values[3];

        cv::Mat oldPlane = plane.clone();


        plane.at<float>(0) = coefficients->values[0];
        plane.at<float>(1) = coefficients->values[1];
        plane.at<float>(2) = coefficients->values[2];
        plane.at<float>(3) = coefficients->values[3];

        if ((newVal < 0 && oldVal > 0) || (newVal > 0 && oldVal < 0)) {
            plane = -plane;
//                double dotProduct = plane.dot(oldPlane) / sqrt(plane.dot(plane) * oldPlane.dot(oldPlane));
//                std::cout << "Flipped plane: " << plane.t() << std::endl;
//                std::cout << "Flip plane: " << dotProduct << std::endl;
        }
//        }

        return true;
    }

    // 这里的话由于我们对平面设定了相对宽松的阈值，即使根据相机位姿未进行优化的位姿将平面投影到世界中，也可以进行匹配
    // 这里的匹配跟我的点 线 进行的描述子的匹配不同
    cv::Mat Frame::ComputePlaneWorldCoeff(const int &idx) {
        cv::Mat temp;
        cv::transpose(mTcw, temp);
        cv::Mat b = -mOw.t();
        // 将其转化为世界坐标下的面参数
        return temp * mvPlaneCoefficients[idx];
    }
    cv::Mat Frame::ComputePlaneWorldCoeff_MF(const int &idx, const cv::Mat Rwc_MF)
    {
        cv::Mat temp;
        cv::transpose(mTcw, temp);

        Rwc_MF.copyTo(temp.rowRange(0,3).colRange(0,3));

        cv::Mat b = -mOw.t();
        // 将其转化为世界坐标下的面参数
        return temp * mvPlaneCoefficients[idx];
    }

    void Frame::ExtractObject(const cv::Mat &imRGB)
    {
        m_Semanticer->Detect(imRGB);
        current_objects = m_Semanticer->GetResult();
    }
} //namespace Planar_SLAM


