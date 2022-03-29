#include "Tracking.h"
#include "ORBmatcher.h"
#include "FrameDrawer.h"
#include "Converter.h"
#include "Map.h"
#include "Initializer.h"
#include "Optimizer.h"
#include "PnPsolver.h"
#include <iostream>
#include <mutex>
#include "PlaneExtractor.h"
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/core.hpp>

#include <pcl/common/common_headers.h>
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/io/io.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>

#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>

#include "sub.h"
using namespace std;
using namespace cv;
using namespace cv::line_descriptor;
using namespace Eigen;

namespace Planar_SLAM {

    PlaneDetection plane_detection;

    Tracking::Tracking(System *pSys, ORBVocabulary *pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Map *pMap,
                       KeyFrameDatabase *pKFDB, const string &strSettingPath, const int sensor, ros::NodeHandle& nh) :
            mState(NO_IMAGES_YET), mSensor(sensor), mbOnlyTracking(false), mbVO(false), mpORBVocabulary(pVoc),
            mpKeyFrameDB(pKFDB), mpInitializer(static_cast<Initializer *>(NULL)), mpSystem(pSys),
            mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpMap(pMap), mnLastRelocFrameId(0) {
        // Load camera parameters from settings file
        depth_pub = nh.advertise<sensor_msgs::Image>("/camera/aligned_depth_to_color/image_raw", 100);
        odometry_pub = nh.advertise<nav_msgs::Odometry>("/vins_estimator/odometry", 100);


        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
        float fx = fSettings["Camera.fx"];
        float fy = fSettings["Camera.fy"];
        float cx = fSettings["Camera.cx"];
        float cy = fSettings["Camera.cy"];

        cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
        K.at<float>(0, 0) = fx;
        K.at<float>(1, 1) = fy;
        K.at<float>(0, 2) = cx;
        K.at<float>(1, 2) = cy;
        K.copyTo(mK);

        cv::Mat DistCoef(4, 1, CV_32F);
        DistCoef.at<float>(0) = fSettings["Camera.k1"];
        DistCoef.at<float>(1) = fSettings["Camera.k2"];
        DistCoef.at<float>(2) = fSettings["Camera.p1"];
        DistCoef.at<float>(3) = fSettings["Camera.p2"];
        const float k3 = fSettings["Camera.k3"];
        if (k3 != 0) {
            DistCoef.resize(5);
            DistCoef.at<float>(4) = k3;
        }
        DistCoef.copyTo(mDistCoef);

        //
        int img_width = fSettings["Camera.width"];
        int img_height = fSettings["Camera.height"];

        cout << "img_width = " << img_width << endl<< "img_height = " << img_height << endl;

        initUndistortRectifyMap(mK, mDistCoef, Mat_<double>::eye(3, 3), mK, Size(img_width, img_height), CV_32F,
                                mUndistX, mUndistY);

        cout << "mUndistX size = " << mUndistX.size << "mUndistY size = " << mUndistY.size << endl;

        mbf = fSettings["Camera.bf"];

        float fps = fSettings["Camera.fps"];
        if (fps == 0)
            fps = 30;

        // Max/Min Frames to insert keyframes and to check relocalisation
        mMinFrames = 10;
        mMaxFrames = fps;

        cout << endl << "Camera Parameters: " << endl;
        cout << "- fx: " << fx << endl;
        cout << "- fy: " << fy << endl;
        cout << "- cx: " << cx << endl;
        cout << "- cy: " << cy << endl;
        cout << "- k1: " << DistCoef.at<float>(0) << endl;
        cout << "- k2: " << DistCoef.at<float>(1) << endl;
        if (DistCoef.rows == 5)
            cout << "- k3: " << DistCoef.at<float>(4) << endl;
        cout << "- p1: " << DistCoef.at<float>(2) << endl;
        cout << "- p2: " << DistCoef.at<float>(3) << endl;
        cout << "- fps: " << fps << endl;


        int nRGB = fSettings["Camera.RGB"];
        mbRGB = nRGB;

        if (mbRGB)
            cout << "- color order: RGB (ignored if grayscale)" << endl;
        else
            cout << "- color order: BGR (ignored if grayscale)" << endl;

        // Load ORB parameters

        int nFeatures = fSettings["ORBextractor.nFeatures"];
        float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
        int nLevels = fSettings["ORBextractor.nLevels"];
        int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
        int fMinThFAST = fSettings["ORBextractor.minThFAST"];

        mpORBextractorLeft = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

        if (sensor == System::STEREO)
            mpORBextractorRight = new ORBextractor(nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

        cout << endl << "ORB Extractor Parameters: " << endl;
        cout << "- Number of Features: " << nFeatures << endl;
        cout << "- Scale Levels: " << nLevels << endl;
        cout << "- Scale Factor: " << fScaleFactor << endl;
        cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
        cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

        if (sensor == System::STEREO || sensor == System::RGBD) {
            mThDepth = mbf * (float) fSettings["ThDepth"] / fx;
            cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
        }

        if (sensor == System::RGBD) {
            mDepthMapFactor = fSettings["DepthMapFactor"];
            if (fabs(mDepthMapFactor) < 1e-5)
                mDepthMapFactor = 1;
            else
                mDepthMapFactor = 1.0f / mDepthMapFactor;
        }

        mfDThRef = fSettings["Plane.AssociationDisRef"];
        mfDThMon = fSettings["Plane.AssociationDisMon"];
        mfAThRef = fSettings["Plane.AssociationAngRef"];
        mfAThMon = fSettings["Plane.AssociationAngMon"];

        mfVerTh = fSettings["Plane.VerticalThreshold"];
        mfParTh = fSettings["Plane.ParallelThreshold"];

        mMax_merge_dist = fSettings["Plane.MAX_MERGE_DIST"];
        mPatch_size = fSettings["Plane.PATCH_SIZE"];
        mMax_point_dist = fSettings["Point.MaxDistance"];


        manhattanCount = 0;
        fullManhattanCount = 0;

//        mpPointCloudMapping = make_shared<MeshViewer>(mpMap);
    }


    void Tracking::SetLocalMapper(LocalMapping *pLocalMapper) {
        mpLocalMapper = pLocalMapper;
    }

    void Tracking::SetLoopClosing(LoopClosing *pLoopClosing) {
        mpLoopClosing = pLoopClosing;
    }

    void Tracking::SetViewer(Viewer *pViewer) {
        mpViewer = pViewer;
    }
    void Tracking::SetPangolinViewer(PangolinViewer *pViewer)
    {
        mpPangolinViewer=pViewer;
    }

// 输入左目RGB或RGBA图像和深度图
// 1、将图像转为mImGray和imDepth并初始化mCurrentFrame
// 2、进行tracking过程
// 输出世界坐标系到该帧相机坐标系的变换矩阵
    cv::Mat Tracking::GrabImageRGBD(const cv::Mat &imRGB, const cv::Mat &imD, const double &timestamp,Eigen::Quaterniond &GroundTruth_R) {
        mImRGB = imRGB; //彩色图像
        mImGray = imRGB;//灰度图像
        mImDepth = imD; //深度图像

        // step 1：将RGB或RGBA图像转为灰度图像
        if (mImGray.channels() == 3) {
            if (mbRGB)
                cvtColor(mImGray, mImGray, CV_RGB2GRAY);
            else
                cvtColor(mImGray, mImGray, CV_BGR2GRAY);
        } else if (mImGray.channels() == 4) {
            if (mbRGB)
                cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
            else
                cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
        }

#ifdef DEBUG_INFO
        cv::imshow("mImGray",mImGray);
        cv::imshow("mImDepth",mImDepth);
        cv::waitKey(1);
#endif
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        // 构造Frame  RGB图 灰度图 深度图 时间戳 ORB特征提取器 词典 相机内参矩阵 相机的去畸变参数 相机基线*相机焦距 内外点区分深度阈值 深度转换因子 以及YOLOX引擎
        mCurrentFrame = Frame(mImRGB, mImGray, mImDepth, timestamp, mpORBextractorLeft, mpORBVocabulary, mK,
                              mDistCoef, mbf, mThDepth, mDepthMapFactor,mMax_merge_dist,mPatch_size,mMax_point_dist,Semanticer);

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
        double t12= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        Track(GroundTruth_R);
        std::chrono::steady_clock::time_point t3 = std::chrono::steady_clock::now();
        double t32= std::chrono::duration_cast<std::chrono::duration<double> >(t3 - t2).count();
#ifdef DEBUG_INFO
        std::cout << "creat Frame cost time: " << t12 <<  endl;
        std::cout << "Tracking cost time: " << t32 <<  endl;
#endif

        return mCurrentFrame.mTcw.clone();
    }

/*
 * @brief Main tracking function. It is independent of the input sensor.
 *
 * track包含两部分：估计运动、跟踪局部地图
 *
 * Step 1：初始化
 * Step 2：跟踪
 * Step 3：记录位姿信息，用于轨迹复现
 */
    void Tracking::Track(Eigen::Quaterniond &GroundTruth_R) {
        // mState为tracking的状态，包括 SYSTME_NOT_READY, NO_IMAGE_YET, NOT_INITIALIZED, OK, LOST
        // 如果图像复位过、或者第一次运行，则为NO_IMAGE_YET状态
        if (mState == NO_IMAGES_YET) {
            mState = NOT_INITIALIZED;
        }
        // mLastProcessedState 存储了Tracking最新的状态，用于FrameDrawer中的绘制
        mLastProcessedState = mState;

        // Get Map Mutex -> Map cannot be changed
        // 地图更新时加锁。保证地图不会发生变化
        // 疑问:这样子会不会影响地图的实时更新?
        // 回答：主要耗时在构造帧中特征点的提取和匹配部分,在那个时候地图是没有被上锁的,有足够的时间更新地图
        unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

        mvpSurfaceNormals_M.mvpsurfacenormal = mCurrentFrame.vSurfaceNormal;
        mvpSurfaceNormals_M.bsurfacenormal_inline =vector<bool>(mCurrentFrame.vSurfaceNormal.size(), false);

        if (mState == NOT_INITIALIZED &&  mpMap->GetMaxKFid() == 0) {
            if(mCurrentFrame.mnPlaneNum < 2) return;
            if (mSensor == System::STEREO || mSensor == System::RGBD) {
                cout << "mCurrentFrame.mnPlaneNum " <<mCurrentFrame.mnPlaneNum << endl;
                Rotation_cm = cv::Mat::zeros(cv::Size(3, 3), CV_32F);
                // 平面上的点云是用来求取曼哈顿到相机的变换
                Rotation_cm = mpMap->FindManhattan(mCurrentFrame, mfVerTh, true);
                //Rotation_cm=SeekManhattanFrame(mCurrentFrame.vSurfaceNormal,mCurrentFrame.mVF3DLines).clone();
                // 根据上一帧的主轴确定当前帧的主轴
                Rotation_cm = TrackManhattanFrame(Rotation_cm, mCurrentFrame.vSurfaceNormal, mCurrentFrame.mVF3DLines).clone();
                Rotation_cm = TrackManhattanFrame(Rotation_cm, mCurrentFrame.vSurfaceNormal, mCurrentFrame.mVF3DLines).clone();
                Rotation_cm = TrackManhattanFrame(Rotation_cm, mCurrentFrame.vSurfaceNormal, mCurrentFrame.mVF3DLines).clone();

                //双目RGBD相机的初始化共用一个函数
                StereoInitialization();
                Rotation_wm = cv::Mat::eye(3, 3, CV_32F);
                /*利用反投影对面参数进行修正
                 * */
//                UpdatePlane(mLastRcm);
            } else
                MonocularInitialization();
            mLastRcm = Rotation_cm.clone();
            mpMap->SetRmc(Rotation_cm);
            mpFrameDrawer->Update(this);

            if (mState != OK)
                return;
        }
        else if(mState == NOT_INITIALIZED &&  mpMap->GetMaxKFid() > 0){
            if(mSensor==System::STEREO || mSensor==System::RGBD){
                cout << "Tracking.cc :: Stereo Initializing with map......" << endl;
                cout << "mCurrentFrame.mnPlaneNum " <<mCurrentFrame.mnPlaneNum << endl;
                Rotation_cm = cv::Mat::zeros(cv::Size(3, 3), CV_32F);
                StereoInitializationWithMap();
                cout << "Map initialized for Stereo camera. Tracking thread keep working." << endl;

                Rotation_cm.at<float>(0,0) = 1;Rotation_cm.at<float>(1,1) = 1;Rotation_cm.at<float>(2,2) = 1;

//                Rotation_cm = mpMap->FindManhattan(mCurrentFrame, mfVerTh, true);
//                Rotation_cm = TrackManhattanFrame(Rotation_cm, mCurrentFrame.vSurfaceNormal, mCurrentFrame.mVF3DLines).clone();
                mLastRcm = Rotation_cm.clone();
                mpMap->SetRmc(mLastRcm);
                mpFrameDrawer->Update(this);
            }

            mpFrameDrawer->Update(this);

            if(mState!=OK){
                cout << "mState!=OK, return."<< endl;
                mState = NOT_INITIALIZED;
                return;
            }
        }
        else {
            //Tracking: system is initialized
            // bOK为临时变量，用于表示每个函数是否执行成功
            bool bOK = false;
            bool bManhattan = false;
//            cout << "INITIALIZED"<< endl;
            floor_normal.copyTo(mCurrentFrame.Floor_Normal);
            // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
            // mbOnlyTracking等于false表示正常SLAM模式（定位+地图更新），mbOnlyTracking等于true表示仅定位模式
            // tracking 类构造时默认为false
            if (!mbOnlyTracking) {
                mUpdateMF = true;
                cv::Mat MF_can = cv::Mat::zeros(cv::Size(3, 3), CV_32F);
                cv::Mat MF_can_T = cv::Mat::zeros(cv::Size(3, 3), CV_32F);

                MF_can = TrackManhattanFrame(mLastRcm, mCurrentFrame.vSurfaceNormal, mCurrentFrame.mVF3DLines).clone();// MF to camera
                MF_can.copyTo(mLastRcm);//.clone();
                MF_can = TrackManhattanFrame(mLastRcm, mCurrentFrame.vSurfaceNormal, mCurrentFrame.mVF3DLines).clone();// MF to camera
                MF_can.copyTo(mLastRcm);//.clone();
                MF_can = TrackManhattanFrame(mLastRcm, mCurrentFrame.vSurfaceNormal, mCurrentFrame.mVF3DLines).clone();// MF to camera

                /*利用反投影对面参数进行修正
                 * */
//                UpdatePlane(MF_can);

                /*根据更新的面参数重新确定Rcm
                 * 这里必须需要至少需要两个相互正交的面才行。如果有IMU存在的话，那必须有一个与其正交才行。
                 *
                 *
                 * MF_can = UpadateManhattanFram(MF_can,mCurrentFrame);
                 *
                 * */

                MF_can.copyTo(mLastRcm);//.clone();
                MF_can_T = MF_can.t();//MF_can_T: camera to MF
                // Rotation_cm -> Rotation_mw
                mRotation_wc = Rotation_wm * MF_can_T;//Rotation_cm 不仅是MF to camera 更由于第一帧的时候将相机坐标系作为世界坐标系 所以他也是MF to World
                mRotation_wc = mRotation_wc.t(); // 实际是w到c
                // Step 2.1 检查并更新上一帧被替换的MapPoints
                // 局部建图线程则可能会对原有的地图点进行替换.在这里进行检查
                CheckReplacedInLastFrame();

                if (mVelocity.empty() || mCurrentFrame.mnId < mnLastRelocFrameId + 2) {
//                    bOK = TranslationEstimation(false);
                    bOK = TrackReferenceKeyFrame(false);

                } else {
                    // 用最近的普通帧来跟踪当前的普通帧
                    // 根据恒速模型设定当前帧的初始位姿
                    // 通过投影的方式在参考帧中找当前帧特征点的匹配点
                    // 优化每个特征点所对应3D点的投影误差即可得到位姿
//                    bOK = TranslationWithMotionModel(GroundTruth_R,false);
                    bOK = TrackWithMotionModel(false);
//                    if(!bOK){
//                        bOK = TranslationWithMotionModel(GroundTruth_R ,false);
//                    }

                    if (!bOK) {
                        //根据恒速模型失败了，只能根据参考关键帧来跟踪
                        cout << "根据恒速模型失败了，只能根据参考关键帧来跟踪" << endl;
//                        bOK = TranslationEstimation(false);
                        bOK = TrackReferenceKeyFrame(false);
                    }
                }
            }
            else{
                // Localization Mode: Local Mapping is deactivated
                cout << endl << "Tracking.cc :: Pure localization mode." << endl;

                if(mState==LOST)
                {
                    cout << "Tracking.cc :: mState==LOST, Relocalization();" << endl;
                    bOK = Relocalization(false);
                }
                else
                {
                    if(!mbVO)
                    {
                        // In last frame we tracked enough MapPoints in the map
                        cout << "Tracking.cc :: Now mbVO is false, means in the last frame we tracked enough MapPoints in the map(nmatchesMap>20);" << endl;
                        if(!mVelocity.empty())
                        {
                            cout << "Tracking.cc :: mVelocity not empty, TrackWithMotionModel();" << endl;
                            bOK = TrackWithMotionModel(false);
                        }
                        else
                        {
                            cout << "Tracking.cc :: mVelocity is empty, TrackReferenceKeyFrame();" << endl;
                            bOK = TrackReferenceKeyFrame(false);
                        }
                    }
                    else
                    {
                        // In last frame we tracked mainly "visual odometry" points.
                        cout << "Tracking.cc :: Now mbVO is true, means that when we do relocalization, nmatchesMap<10; mbVO will be false if nmatchesMap>20;" << endl;

                        // We compute two camera poses, one from motion model and one doing relocalization.
                        // If relocalization is sucessfull we choose that solution, otherwise we retain
                        // the "visual odometry" solution.

                        bool bOKMM = false;
                        bool bOKReloc = false;
                        vector<MapPoint*> vpMPsMM;
                        vector<bool> vbOutMM;
                        cv::Mat TcwMM;
                        if(!mVelocity.empty())
                        {
                            bOKMM = TrackWithMotionModel(false);
                            vpMPsMM = mCurrentFrame.mvpMapPoints;
                            vbOutMM = mCurrentFrame.mvbOutlier;
                            TcwMM = mCurrentFrame.mTcw.clone();
                        }
                        bOKReloc = Relocalization(false);

                        if(bOKMM && !bOKReloc)
                        {
                            cout << "Tracking.cc :: bOKMM && !bOKReloc; means TrackWithMotionModel() succeed, Relocalization() failed." << endl;
                            mCurrentFrame.SetPose(TcwMM);
                            mCurrentFrame.mvpMapPoints = vpMPsMM;
                            mCurrentFrame.mvbOutlier = vbOutMM;

                            if(mbVO)
                            {
                                cout << "Tracking.cc :: mbVO; Add current keypoints to mvpMapPoints." << endl;
                                for(int i =0; i<mCurrentFrame.N; i++)
                                {
                                    if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                                    {
                                        mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                                    }
                                }
                            }
                        }
                        else if(bOKReloc)
                        {
                            mbVO = false;
                            cout << "Tracking.cc :: !bOKMM && bOKReloc; means TrackWithMotionModel() failed, Relocalization() succeed." << endl;
                        }

                        bOK = bOKReloc || bOKMM;
                        if(!bOK){ // for debug use
                            cout << "Tracking.cc :: !bOK, because bOKReloc = " << bOKReloc << ", bOKMM = " << bOKMM << endl;
                        }
                    }
                }
            }

            // 计算曼哈顿和追踪之间的旋转的差异
            cv::Mat GT_rotation;
            GT_rotation = Converter::toCvMat(GroundTruth_R.toRotationMatrix());

//            double alpha = MatrixResidual(mRotation_wc,GT_rotation);
//            std::cout << "***********Manhataan : "<< alpha << std::endl;
//            double blpha = MatrixResidual(mCurrentFrame.mRcw,GT_rotation);
//            std::cout << "***********Estimation: "<< blpha << std::endl;

            mpMap->SetSurfaceNormal_M(mvpSurfaceNormals_M);
            mpMap->SetRmc(mLastRcm);
            mCurrentFrame.mpReferenceKF = mpReferenceKF;


            //Pm  Rmw * Pw   令y=0  反投影回去
            //Rotation_cm  Mf to world
//            cout<< "track local map"<< endl;
            // If we have an initial estimation of the camera pose and matching. Track the local map.
            if (!mbOnlyTracking) {
                if (bOK) {
                    bOK = TrackLocalMap(true);
                } else {
                    bOK = Relocalization(true);
                }
            }
            else{
                if(bOK && !mbVO){
                    cout << "Tracking.cc :: bOK && !mbVO; means that there are many matches in the map, so we can TrackLocalMap()." << endl;
                    bOK = TrackLocalMap(false);
                }
            }
//            cout<< "track local map over"<< endl;
            //Pm  Rmw * Pw   令y=0  反投影回去
            //Rotation_cm  Mf to world
            // fix camera on ground
            if(1==0 && !mbOnlyTracking )
            {
                cv::Mat Rwm = Rotation_wm;
                cv::Mat Pm = Rwm.t()*mCurrentFrame.mOw;
                Eigen::Vector3f P_m;
                cv2eigen(Pm,P_m);
                P_m(1)=0.0;
                eigen2cv(P_m,Pm);
                cv::Mat Pow = Rwm*Pm;
                cv::Mat mtcw = -mCurrentFrame.mRcw*Pow;
                mtcw.copyTo(mCurrentFrame.mTcw.rowRange(0,3).col(3));
                mCurrentFrame.UpdatePoseMatrices();
                // 不仅是P，旋转同样也要处理   [ *, 0, *]
                /*  相机到曼哈顿之间的变换：   [ 0, 1, 0] =R_dis*Rmc*I=Rmw*Rwc*I
                 *                         [ *, 0, *]
                 *  对Rmc*I进行处理
                 *  Rmc
                 * */
            }
    //            double clpha = MatrixResidual(mCurrentFrame.mRcw,GT_rotation);
    //            std::cout << "***********     : "<< clpha << std::endl;
            // update rotation from manhattan
            cv::Mat new_Rotation_wc = mCurrentFrame.mTcw.rowRange(0, 3).colRange(0, 3).t();
            cv::Mat Rotation_mc = Rotation_cm.t();
            cv::Mat MF_can_T;
            MF_can_T = Rotation_mc * new_Rotation_wc;
            mLastRcm = MF_can_T.t();

            if (bOK)
                mState = OK;
            else
                mState = LOST;
            // Update drawer
            // 更新线程中的图像、特征点、线等信息
//            cout << "mpFrameDrawer->Update"<< endl;
            mpFrameDrawer->Update(this);
//            cout << "finish"<< endl;
            mpMap->FlagMatchedPlanePoints(mCurrentFrame, mfDThRef);

            //Update Planes
            // 更新平面
//            cout << "Update Planes" << endl;
            for (int i = 0; i < mCurrentFrame.mnPlaneNum; ++i) {
                MapPlane *pMP = mCurrentFrame.mvpMapPlanes[i];// 匹配成功的面
                if (pMP) {
                    pMP->UpdateCoefficientsAndPoints(mCurrentFrame, i);
                } else if (!mCurrentFrame.mvbPlaneOutlier[i])  {
                    //这个面没有匹配上，同样也不是外点
                    //  todo 在这里我想加入一个关系就是，他不从属于anyone, 但是他要在约束的环境中才行
                    MapPlane *pMPv =mCurrentFrame.mvpVerticalPlanes[i];
                    MapPlane *pMPp =mCurrentFrame.mvpParallelPlanes[i];
    //                    cout << "insertImage"
    //                         << mCurrentFrame.mvpVerticalPlanes[i]->mnId<< "  "
    //                         <<mCurrentFrame.mvpParallelPlanes[i]->mnId<< endl;
                    if(pMPv||pMPp)
                    {
                        mCurrentFrame.mbNewPlane = true;
    //                        cout << " mCurrentFrame.mbNewPlane = true;"<< endl;
                    }
                }
            }

    //            mpPointCloudMapping->print();

            // If tracking were good, check if we insert a keyframe
            //只有在成功追踪时才考虑生成关键帧的问题
            if (bOK) {
                // Update motion model
                // Step 5：跟踪成功，更新恒速运动模型
//                cout << "Update motion model"<< endl;
                if (!mLastFrame.mTcw.empty()) {
                    cv::Mat LastTwc = cv::Mat::eye(4, 4, CV_32F);
                    mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0, 3).colRange(0, 3));
                    mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0, 3).col(3));
                    // mVelocity = Tcl = Tcw * Twl,表示上一帧到当前帧的变换， 其中 Twl = LastTwc
                    mVelocity = mCurrentFrame.mTcw * LastTwc;
                } else
                    //否则速度为空
                    mVelocity = cv::Mat();
                //更新显示中的位姿
                mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);
                // Clean VO matches
                // Step 6：清除观测不到的地图点
                for (int i = 0; i < mCurrentFrame.N; i++) {
                    MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                    if (pMP)
                        if (pMP->Observations() < 1) {
                            mCurrentFrame.mvbOutlier[i] = false;
                            mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                        }
                }
                for (int i = 0; i < mCurrentFrame.NL; i++) {
                    MapLine *pML = mCurrentFrame.mvpMapLines[i];
                    if (pML)
                        if (pML->Observations() < 1) {
                            mCurrentFrame.mvbLineOutlier[i] = false;
                            mCurrentFrame.mvpMapLines[i] = static_cast<MapLine *>(NULL);
                        }
                }
                for (int i = 0; i < mCurrentFrame.mnPlaneNum; i++) {
                    MapPlane *pMP = mCurrentFrame.mvpMapPlanes[i];//这个是地图中的平面，但凡是能匹配的上
                    if (pMP)
                        if (pMP->Observations() < 1) {
                            mCurrentFrame.mvbPlaneOutlier[i] = false;
                            mCurrentFrame.mvpMapPlanes[i] = static_cast<MapPlane *>(NULL);
                        }

                    MapPlane *pVMP = mCurrentFrame.mvpVerticalPlanes[i];
                    if (pVMP)
                        if (pVMP->Observations() < 1) {
                            mCurrentFrame.mvbVerPlaneOutlier[i] = false;
                            mCurrentFrame.mvpVerticalPlanes[i] = static_cast<MapPlane *>(NULL);
                        }

                    MapPlane *pPMP = mCurrentFrame.mvpParallelPlanes[i];
                    if (pVMP)
                        if (pVMP->Observations() < 1) {
                            mCurrentFrame.mvbParPlaneOutlier[i] = false;
                            mCurrentFrame.mvpParallelPlanes[i] = static_cast<MapPlane *>(NULL);
                        }
                }

                // Delete temporal MapPoints
                // Delete temporal MapPoints
                // Step 7：清除恒速模型跟踪中 UpdateLastFrame中为当前帧临时添加的MapPoints（仅双目和rgbd）
                // 步骤6中只是在当前帧中将这些MapPoints剔除，这里从MapPoints数据库中删除
                // 临时地图点仅仅是为了提高双目或rgbd摄像头的帧间跟踪效果，用完以后就扔了，没有添加到地图中
                for (list<MapPoint *>::iterator lit = mlpTemporalPoints.begin(), lend = mlpTemporalPoints.end();
                     lit != lend; lit++) {
                    MapPoint *pMP = *lit;
                    delete pMP;
                }
                for (list<MapLine *>::iterator lit = mlpTemporalLines.begin(), lend = mlpTemporalLines.end();
                     lit != lend; lit++) {
                    MapLine *pML = *lit;
                    delete pML;
                }
                mlpTemporalPoints.clear();
                mlpTemporalLines.clear();
                // Check if we need to insert a new keyframe
                // Step 8：检测并插入关键帧，对于双目或RGB-D会产生新的地图点
                if (NeedNewKeyFrame()) {
                    CreateNewKeyFrame();
                }
                //
                // wangwen
                // publish depth and odometry // camera to world
                static int cnt = 0;
                if(mbOnlyTracking && ++cnt >5){
                    cnt =0;
                    cv::Mat Rwc = mCurrentFrame.GetRotationInverse();
                    cv::Mat twc = mCurrentFrame.GetCameraCenter();

                    cv::Mat R = cv::Mat::zeros(3,3,CV_32FC1);
                    R.at<float>(0,0) = 1;
                    R.at<float>(1,2) = 1;
                    R.at<float>(2,1) = -1;

                    vector<float> q_ = Planar_SLAM::Converter::toQuaternion(R*Rwc);
                    Eigen::Matrix<double,3,1> t_ = Planar_SLAM::Converter::toVector3d(R*twc);

                    nav_msgs::Odometry odometry;
                    odometry.header.frame_id = "world";
                    odometry.pose.pose.orientation.x = q_[0];
                    odometry.pose.pose.orientation.y = q_[1];
                    odometry.pose.pose.orientation.z = q_[2];
                    odometry.pose.pose.orientation.w = q_[3];

                    odometry.pose.pose.position.x = t_.x();
                    odometry.pose.pose.position.y = t_.y();
                    odometry.pose.pose.position.z = t_.z();

                    cout << "###########################"<< t_.transpose() << endl;

//                    // image
//                    cv_bridge::CvImage out_msg;
//                    cv::Mat depth = mCurrentFrame.depth;
//                    out_msg.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
//                    out_msg.header.stamp = ros::Time::now();
//                    std_msgs::Header img_heager;
//                    img_heager.stamp = ros::Time::now();
//                    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(img_heager,sensor_msgs::image_encodings::TYPE_32FC1,depth).toImageMsg();
//
//                    cout << odometry.header.stamp<< "\t"<< out_msg.header.stamp<< endl;
//                    depth_pub.publish(msg);
                    odometry_pub.publish(odometry);
                }
                // We allow points with high innovation (considererd outliers by the Huber Function)
                // pass to the new keyframe, so that bundle adjustment will finally decide
                // if they are outliers or not. We don't want next frame to estimate its position
                // with those points so we discard them in the frame.
                for (int i = 0; i < mCurrentFrame.N; i++) {
                    if (mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                }

                for (int i = 0; i < mCurrentFrame.NL; i++) {
                    if (mCurrentFrame.mvpMapLines[i] && mCurrentFrame.mvbLineOutlier[i])
                        mCurrentFrame.mvpMapLines[i] = static_cast<MapLine *>(NULL);
                }
            }

            // Reset if the camera get lost soon after initialization
            if (mState == LOST) {
                if (mpMap->KeyFramesInMap() <= 5) {
                    mpSystem->Reset();
                    return;
                }
            }

            if (!mCurrentFrame.mpReferenceKF)
                mCurrentFrame.mpReferenceKF = mpReferenceKF;

            mLastFrame = Frame(mCurrentFrame);
        }
//        cout << "Store frame pose "<<endl;
        // Store frame pose information to retrieve the complete camera trajectory afterwards.
        if (!mCurrentFrame.mTcw.empty()) {
            cv::Mat Tcr = mCurrentFrame.mTcw * mCurrentFrame.mpReferenceKF->GetPoseInverse();
            mlRelativeFramePoses.push_back(Tcr);
            mlpReferences.push_back(mpReferenceKF);
            mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
            mlbLost.push_back(mState == LOST);
        } else {
            // This can happen if tracking is lost
            mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
            mlpReferences.push_back(mlpReferences.back());
            mlFrameTimes.push_back(mlFrameTimes.back());
            mlbLost.push_back(mState == LOST);
        }

    }

    cv::Mat
    Tracking::SeekManhattanFrame(vector<SurfaceNormal> &vTempSurfaceNormal, vector<FrameLine> &vVanishingDirection) {


        vector<cv::Mat> vRotaionMatrix;
        vector<cv::Mat> vRotaionMatrix_good;
        cv::RNG rnger(cv::getTickCount());
        vector<cv::Mat> vSN_good;
        vector<double> lambda_good;
        vector<cv::Point2d> m_j_selected;
        // R_cm_update matrix
        cv::Mat R_cm_update=cv::Mat::eye(cv::Size(3,3),CV_32F);
        cv::Mat R_cm_new=cv::Mat::eye(cv::Size(3,3),CV_32F);
// initialization with random matrix
#if 1
        cv::Mat qu = cv::Mat::zeros(cv::Size(4,1),CV_32F);
        rnger.fill(qu, cv::RNG::UNIFORM, cv::Scalar::all(0.01), cv::Scalar::all(1));
        Eigen::Quaterniond qnorm;
        Eigen::Quaterniond q(qu.at<float>(0,0),qu.at<float>(1,0),qu.at<float>(2,0),qu.at<float>(3,0));//=Eigen::MatrixXd::Random(1, 4);
        qnorm.x()=q.x()/q.norm();qnorm.y()=q.y()/q.norm();
        qnorm.z()=q.z()/q.norm();qnorm.w()=q.w()/q.norm();
        cv::eigen2cv(qnorm.matrix(),R_cm_update);//	eigen2cv(m, img);;*/
        //cout<<R_cm_update<<endl;
        cv::SVD svd; cv::Mat U,W,VT;
        svd.compute(R_cm_update,W,U,VT);
        R_cm_update=U*VT;
        //cout<<000<<R_cm_update<<endl;
        // R_cm_Rec matrix
        cv::Mat R_cm_Rec=cv::Mat::zeros(cv::Size(3,3),CV_32F);

        cv::Mat R_cm_initial;
        int  validMF=0;
        //cout<<R_cm_update<<endl;
        R_cm_new.at<float>(0,0) = R_cm_update.at<double>(0,0);
        R_cm_new.at<float>(0,1) = R_cm_update.at<double>(0,1);
        R_cm_new.at<float>(0,2) = R_cm_update.at<double>(0,2);
        R_cm_new.at<float>(1,0) = R_cm_update.at<double>(1,0);
        R_cm_new.at<float>(1,1) = R_cm_update.at<double>(1,1);
        R_cm_new.at<float>(1,2) = R_cm_update.at<double>(1,2);
        R_cm_new.at<float>(2,0) = R_cm_update.at<double>(2,0);
        R_cm_new.at<float>(2,1) = R_cm_update.at<double>(2,1);
        R_cm_new.at<float>(2,2) = R_cm_update.at<double>(2,2);
        //cout<<R_cm_new<<endl;
        //matTemp.convertTo(MatTemp2, CV_8U)
#endif

        R_cm_new = TrackManhattanFrame(R_cm_new, vTempSurfaceNormal,vVanishingDirection);
        return R_cm_new;//vRotaionMatrix_good[0];
    }


    cv::Mat Tracking::ClusterMultiManhattanFrame(vector<cv::Mat> &vRotationCandidate, double &clusterRatio) {
        //MF_nonRd = [];
        vector<vector<int>> bin;
        //succ_rate = [];
        cv::Mat a;
        vector<cv::Mat> MF_nonRd;
        int histStart = 0;
        float histStep = 0.1;
        int histEnd = 2;
        int HasPeak = 1;
        int numMF_can = vRotationCandidate.size();
        int numMF = numMF_can;
        //rng(0,'twister');
        int numMF_nonRd = 0;

        while (HasPeak == 1) {
            //随机的一个Rotation
            cv::Mat R = vRotationCandidate[rand() % (numMF_can - 1) + 1];
            cv::Mat tempAA;
            vector<cv::Point3f> Paa;
            //
            vector<float> vDistanceOfRotation;
            cv::Mat Rvec = R.t() * R;
            float theta = acos((trace(Rvec)[0] - 1) / 2);
            cv::Point3f w;
            w.x = theta * 1 / 2 * sin(theta) * (Rvec.at<float>(2, 1) - Rvec.at<float>(1, 2));
            w.y = theta * 1 / 2 * sin(theta) * (Rvec.at<float>(0, 2) - Rvec.at<float>(2, 0));
            w.z = theta * 1 / 2 * sin(theta) * (Rvec.at<float>(1, 0) - Rvec.at<float>(0, 1));

            for (int i = 0; i < vRotationCandidate.size(); i++) {
                cv::Mat RvecBetween = R.t() * vRotationCandidate[i];
                float theta = acos((trace(RvecBetween)[0] - 1) / 2);
                cv::Point3f wb;
                wb.x = theta * 1 / 2 * sin(theta) * (RvecBetween.at<float>(2, 1) - RvecBetween.at<float>(1, 2));
                wb.y = theta * 1 / 2 * sin(theta) * (RvecBetween.at<float>(0, 2) - RvecBetween.at<float>(2, 0));
                wb.z = theta * 1 / 2 * sin(theta) * (RvecBetween.at<float>(1, 0) - RvecBetween.at<float>(0, 1));
                Paa.push_back(wb);
                vDistanceOfRotation.push_back(norm(w - wb));
            }

            //
            bin = EasyHist(vDistanceOfRotation, histStart, histStep, histEnd);

            HasPeak = 0;
            for (int k = 0; k < bin.size(); k++) {
                int binSize = 0;
                for (int n = 0; n < bin[k].size(); n++) { if (bin[k][n] > 0)binSize++; }
                if (binSize / numMF_can > clusterRatio) {
                    HasPeak = 1;
                    break;
                }
            }
            //if(HasPeak == 0) return;

            int binSize1 = 1;
            for (int n = 0; n < bin[0].size(); n++) { if (bin[0][n] > 0)binSize1++; }
            // check whether the dominant bin happens at zero
            if (binSize1 / numMF_can > clusterRatio) {
                cv::Point3f meanPaaTem(0, 0, 0);
                for (int n = 0; n < bin[0].size(); n++) {
                    if (bin[0][n] > 0) {
                        meanPaaTem += Paa[bin[0][n]];
                        meanPaaTem = meanPaaTem / binSize1;
                        meanPaaTem = meanPaaTem / norm(meanPaaTem);
                    }

                }
                //calculate the mean
                float s = sin(norm(meanPaaTem));
                float c = cos(norm(meanPaaTem));
                float t = 1 - c;
                cv::Point3f vec_n(0, 0, 0);
                if (norm(meanPaaTem) <= 0.0001) {}

                else
                    vec_n = meanPaaTem;


                float x = vec_n.x;
                float y = vec_n.y;
                float z = vec_n.z;
                cv::Mat mm = cv::Mat::zeros(cv::Size(3, 3), CV_32F);
                mm.at<float>(0, 0) = t * x * x + c;
                mm.at<float>(0, 1) = t * x * y - s * z;
                mm.at<float>(0, 2) = t * x * z + s * y;
                mm.at<float>(1, 0) = t * x * y + s * z;
                mm.at<float>(1, 1) = t * y * y + c;
                mm.at<float>(1, 2) = t * y * z - s * x;
                mm.at<float>(2, 0) = t * x * z - s * y;
                mm.at<float>(2, 1) = t * y * z + s * x;
                mm.at<float>(2, 2) = t * z * z + c;

                if (isnan(sum(mm)[0]) && (norm(meanPaaTem) == 0)) {
                    numMF_nonRd += 1;
                    MF_nonRd.push_back(R);
                } else {
                    numMF_nonRd = numMF_nonRd + 1;
                    MF_nonRd.push_back(R * mm);
                    //succ_rate{numMF_nonRd} = numel(bin{1})/numMF_can;
                }

                /*for(int j = 0;j<bin[0].size();j++)
            {
                if(bin[0][j]>0)
                {
                    vRotationCandidate[];
                }

            }*/
                //MF_can{bin{1}(j)} = [];

            }

        }
        return a;
    }

    vector<vector<int>> Tracking::EasyHist(vector<float> &vDistance, int &histStart, float &histStep, int &histEnd) {
        int numData = vDistance.size();
        int numBin = (histEnd - histEnd) / histStep;
        vector<vector<int>> bin(numBin, vector<int>(numBin, 0));//bin(numBin,0);
        for (int i = 0; i < numBin; i++) {
            float down = (i - 1) * histStep + histStart;
            float up = down + histStep;
            for (int j = 1; j < numData; j++) {
                if (vDistance[j] >= down && vDistance[j] < up)
                    bin[i].push_back(j);//=bin[i]+1;
            }

        }
        return bin;

    }

    cv::Mat Tracking::ProjectSN2MF(int a, const cv::Mat &R_cm, const vector<SurfaceNormal> &vTempSurfaceNormal,
                                   vector<FrameLine> &vVanishingDirection) {
        vector<cv::Point2d> m_j_selected;
        cv::Mat R_cm_Rec = cv::Mat::zeros(cv::Size(1, 3), CV_32F);
        cv::Mat R_cm_NULL = cv::Mat::zeros(cv::Size(1, 3), CV_32F);
        cv::Mat R_mc = cv::Mat::zeros(cv::Size(3, 3), CV_32F);
        //R_Mc = [R_cM(:,mod(a+3,3)+1), R_cM(:,mod(a+4,3)+1), R_cM(:,mod(a+5,3)+1)].';

        int c1 = (a + 3) % 3;
        int c2 = (a + 4) % 3;
        int c3 = (a + 5) % 3;
        R_mc.at<float>(0, 0) = R_cm.at<float>(0, c1);
        R_mc.at<float>(0, 1) = R_cm.at<float>(0, c2);
        R_mc.at<float>(0, 2) = R_cm.at<float>(0, c3);
        R_mc.at<float>(1, 0) = R_cm.at<float>(1, c1);
        R_mc.at<float>(1, 1) = R_cm.at<float>(1, c2);
        R_mc.at<float>(1, 2) = R_cm.at<float>(1, c3);
        R_mc.at<float>(2, 0) = R_cm.at<float>(2, c1);
        R_mc.at<float>(2, 1) = R_cm.at<float>(2, c2);
        R_mc.at<float>(2, 2) = R_cm.at<float>(2, c3);
        R_mc = R_mc.t();




        //cout<<"R_cm"<<R_cm<<endl;

        /*cout<<"RCM"<<R_cm.at<float>(0,c1)<<", "<<R_cm.at<float>(1,c1)<<","<<R_cm.at<float>(2,c1)<<","<<
            R_cm.at<float>(0,c2)<<","<<R_cm.at<float>(1,c2)<<","<<R_cm.at<float>(2,c2)<<","<<
            R_cm.at<float>(0,c3)<<","<<R_cm.at<float>(1,c3)<<","<<R_cm.at<float>(2,c3)<<endl;*/
        size_t sizeOfSurfaceNormal = vTempSurfaceNormal.size() + vVanishingDirection.size();
        //cout<<"size of SN"<<sizeOfSurfaceNormal<<endl;
        for (size_t i = 0; i < sizeOfSurfaceNormal; i++) {
            cv::Mat temp = cv::Mat::zeros(cv::Size(1, 3), CV_32F);

            if (i >= vTempSurfaceNormal.size()) {
                temp.at<float>(0, 0) = vVanishingDirection[i].direction.x;
                temp.at<float>(1, 0) = vVanishingDirection[i].direction.y;
                temp.at<float>(2, 0) = vVanishingDirection[i].direction.z;
            } else {
                temp.at<float>(0, 0) = vTempSurfaceNormal[i].normal.x;
                temp.at<float>(1, 0) = vTempSurfaceNormal[i].normal.y;
                temp.at<float>(2, 0) = vTempSurfaceNormal[i].normal.z;
            }
            //cout<<temp<<endl;
            //cout<<" TEMP"<<vTempSurfaceNormal[i].x<<","<<vTempSurfaceNormal[i].y<<","<<vTempSurfaceNormal[i].z<<endl;

            cv::Point3f n_ini;
            cv::Mat m_ini;
            m_ini = R_mc * temp;
            n_ini.x = m_ini.at<float>(0, 0);
            n_ini.y = m_ini.at<float>(1, 0);
            n_ini.z = m_ini.at<float>(2, 0);
            /*n_ini.x=R_mc.at<float>(0,0)*temp.at<float>(0,0)+R_mc.at<float>(0,1)*temp.at<float>(1,0)+R_mc.at<float>(0,2)*temp.at<float>(2,0);
        n_ini.y=R_mc.at<float>(1,0)*temp.at<float>(0,0)+R_mc.at<float>(1,1)*temp.at<float>(1,0)+R_mc.at<float>(1,2)*temp.at<float>(2,0);
        n_ini.z=R_mc.at<float>(2,0)*temp.at<float>(0,0)+R_mc.at<float>(2,1)*temp.at<float>(1,0)+R_mc.at<float>(2,2)*temp.at<float>(2,0);
        //cout<<"R_mc"<<R_mc<<endl;*/


            double lambda = sqrt(n_ini.x * n_ini.x + n_ini.y * n_ini.y);//at(k).y*n_a.at(k).y);
            //cout<<lambda<<endl;
            if (lambda < sin(0.2618)) //0.25
            {
                double tan_alfa = lambda / std::abs(n_ini.z);
                double alfa = asin(lambda);
                double m_j_x = alfa / tan_alfa * n_ini.x / n_ini.z;
                double m_j_y = alfa / tan_alfa * n_ini.y / n_ini.z;
                if (!std::isnan(m_j_x) && !std::isnan(m_j_y))
                    m_j_selected.push_back(cv::Point2d(m_j_x, m_j_y));
                if (i < vTempSurfaceNormal.size()) {
                    if (a == 1)mCurrentFrame.vSurfaceNormalx.push_back(vTempSurfaceNormal[i].FramePosition);
                    else if (a == 2)mCurrentFrame.vSurfaceNormaly.push_back(vTempSurfaceNormal[i].FramePosition);
                    else if (a == 3)mCurrentFrame.vSurfaceNormalz.push_back(vTempSurfaceNormal[i].FramePosition);
                } else {
                    if (a == 1) {
                        cv::Point2d endPoint = vVanishingDirection[i].p;
                        cv::Point2d startPoint = vVanishingDirection[i].q;
                        vector<cv::Point2d> pointPair(2);
                        pointPair.push_back(endPoint);
                        pointPair.push_back(startPoint);
                        mCurrentFrame.vVanishingLinex.push_back(pointPair);
                        for (int k = 0; k < vVanishingDirection[i].rndpts3d.size(); k++)
                            mCurrentFrame.vVaishingLinePCx.push_back(vVanishingDirection[i].rndpts3d[k]);
                    } else if (a == 2) {
                        cv::Point2d endPoint = vVanishingDirection[i].p;
                        cv::Point2d startPoint = vVanishingDirection[i].q;
                        vector<cv::Point2d> pointPair(2);
                        pointPair.push_back(endPoint);
                        pointPair.push_back(startPoint);
                        mCurrentFrame.vVanishingLiney.push_back(pointPair);
                        for (int k = 0; k < vVanishingDirection[i].rndpts3d.size(); k++)
                            mCurrentFrame.vVaishingLinePCy.push_back(vVanishingDirection[i].rndpts3d[k]);
                    } else if (a == 3) {
                        cv::Point2d endPoint = vVanishingDirection[i].p;
                        cv::Point2d startPoint = vVanishingDirection[i].q;
                        vector<cv::Point2d> pointPair(2);
                        pointPair.push_back(endPoint);
                        pointPair.push_back(startPoint);
                        mCurrentFrame.vVanishingLinez.push_back(pointPair);
                        for (int k = 0; k < vVanishingDirection[i].rndpts3d.size(); k++)
                            mCurrentFrame.vVaishingLinePCz.push_back(vVanishingDirection[i].rndpts3d[k]);
                    }

                }
                //lambda_good.push_back(lambda);
                //找到一个面
            }
        }
        //cout<<"m_j_selected.push_back(temp)"<<m_j_selected.size()<<endl;

        if (m_j_selected.size() > sizeOfSurfaceNormal / 20) {
            //cv::Point2d s_j = MeanShift(m_j_selected);
            sMS tempMeanShift = MeanShift(m_j_selected);
            cv::Point2d s_j = tempMeanShift.centerOfShift;// MeanShift(m_j_selected);
            float s_j_density = tempMeanShift.density;
            //cout<<"tracking:s_j"<<s_j.x<<","<<s_j.y<<endl;
            float alfa = norm(s_j);
            float ma_x = tan(alfa) / alfa * s_j.x;
            float ma_y = tan(alfa) / alfa * s_j.y;
            cv::Mat temp1 = cv::Mat::zeros(cv::Size(1, 3), CV_32F);
            temp1.at<float>(0, 0) = ma_x;
            temp1.at<float>(1, 0) = ma_y;
            temp1.at<float>(2, 0) = 1;

            R_cm_Rec = R_mc.t() * temp1;
            R_cm_Rec = R_cm_Rec / norm(R_cm_Rec); //列向量
            return R_cm_Rec;
        }

        return R_cm_NULL;

    }

    /*refer Paper: Divide and Conquer Efficient  ensity-based Tracking of 3D Sensors in Manhattan Worlds
     * 将点云的法向量或线的灭点投影到指定轴的圆锥内
     * a : 指定的轴
     * R_mc: Camera to Mahattam
     * vTempSurfaceNormal: 当前帧提取到平面点云的法向量
     * vVanishingDirection： 线的灭点
     * numOfSN： 前两个轴的最小法向量数量
     * */
    // 对点云的法向量进行滤波处理
    ResultOfMS Tracking::ProjectSN2MF(int a, const cv::Mat &R_mc, const vector<SurfaceNormal> &vTempSurfaceNormal,
                                      vector<FrameLine> &vVanishingDirection, const int numOfSN) {
        vector<cv::Point2d> m_j_selected;
        cv::Mat R_cm_Rec = cv::Mat::zeros(cv::Size(1, 3), CV_32F);
        cv::Mat R_cm_NULL = cv::Mat::zeros(cv::Size(1, 3), CV_32F);
        ResultOfMS RandDen;
        RandDen.axis = a;

        size_t sizeOfSurfaceNormal = vTempSurfaceNormal.size() + vVanishingDirection.size();
        m_j_selected.reserve(sizeOfSurfaceNormal);

        for (size_t i = 0; i < sizeOfSurfaceNormal; i++) {
            //cv::Mat temp=cv::Mat::zeros(cv::Size(1,3),CV_32F);

            cv::Point3f n_ini;
            int tepSize = i - vTempSurfaceNormal.size();
            if (i >= vTempSurfaceNormal.size()) {

                n_ini.x = R_mc.at<float>(0, 0) * vVanishingDirection[tepSize].direction.x +
                          R_mc.at<float>(0, 1) * vVanishingDirection[tepSize].direction.y +
                          R_mc.at<float>(0, 2) * vVanishingDirection[tepSize].direction.z;
                n_ini.y = R_mc.at<float>(1, 0) * vVanishingDirection[tepSize].direction.x +
                          R_mc.at<float>(1, 1) * vVanishingDirection[tepSize].direction.y +
                          R_mc.at<float>(1, 2) * vVanishingDirection[tepSize].direction.z;
                n_ini.z = R_mc.at<float>(2, 0) * vVanishingDirection[tepSize].direction.x +
                          R_mc.at<float>(2, 1) * vVanishingDirection[tepSize].direction.y +
                          R_mc.at<float>(2, 2) * vVanishingDirection[tepSize].direction.z;
            } else {

                n_ini.x = R_mc.at<float>(0, 0) * vTempSurfaceNormal[i].normal.x +
                          R_mc.at<float>(0, 1) * vTempSurfaceNormal[i].normal.y +
                          R_mc.at<float>(0, 2) * vTempSurfaceNormal[i].normal.z;
                n_ini.y = R_mc.at<float>(1, 0) * vTempSurfaceNormal[i].normal.x +
                          R_mc.at<float>(1, 1) * vTempSurfaceNormal[i].normal.y +
                          R_mc.at<float>(1, 2) * vTempSurfaceNormal[i].normal.z;
                n_ini.z = R_mc.at<float>(2, 0) * vTempSurfaceNormal[i].normal.x +
                          R_mc.at<float>(2, 1) * vTempSurfaceNormal[i].normal.y +
                          R_mc.at<float>(2, 2) * vTempSurfaceNormal[i].normal.z;
            }


            double lambda = sqrt(n_ini.x * n_ini.x + n_ini.y * n_ini.y);//at(k).y*n_a.at(k).y);
            //cout<<lambda<<endl;
            //inside the cone
            //refer Paper: Divide and Conquer Efficient  ensity-based Tracking of 3D Sensors in Manhattan Worlds
            // 这里多少有点复杂，时间关系我先pass
            if (lambda < sin(0.2518)) //0.25
            {
                double tan_alfa = lambda / std::abs(n_ini.z);
                double alfa = asin(lambda); //反正弦

                double m_j_x = alfa / tan_alfa * n_ini.x / n_ini.z;
                double m_j_y = alfa / tan_alfa * n_ini.y / n_ini.z;

                if (!std::isnan(m_j_x) && !std::isnan(m_j_y))
                    m_j_selected.push_back(cv::Point2d(m_j_x, m_j_y));
                if (i < vTempSurfaceNormal.size()) {
                    if (a == 1) {
                        mCurrentFrame.vSurfaceNormalx.push_back(vTempSurfaceNormal[i].FramePosition);
                        mCurrentFrame.vSurfacePointx.push_back(vTempSurfaceNormal[i].cameraPosition);
                    } else if (a == 2) {
                        mCurrentFrame.vSurfaceNormaly.push_back(vTempSurfaceNormal[i].FramePosition);
                        mCurrentFrame.vSurfacePointy.push_back(vTempSurfaceNormal[i].cameraPosition);
                    } else if (a == 3) {
                        mCurrentFrame.vSurfaceNormalz.push_back(vTempSurfaceNormal[i].FramePosition);
                        mCurrentFrame.vSurfacePointz.push_back(vTempSurfaceNormal[i].cameraPosition);
                    }
                } else {
                    if (a == 1) {
                        cv::Point2d endPoint = vVanishingDirection[tepSize].p;
                        cv::Point2d startPoint = vVanishingDirection[tepSize].q;
                        vector<cv::Point2d> pointPair(2);
                        pointPair.push_back(endPoint);
                        pointPair.push_back(startPoint);
                        mCurrentFrame.vVanishingLinex.push_back(pointPair);
                        for (int k = 0; k < vVanishingDirection[tepSize].rndpts3d.size(); k++)
                            mCurrentFrame.vVaishingLinePCx.push_back(vVanishingDirection[tepSize].rndpts3d[k]);
                    } else if (a == 2) {
                        cv::Point2d endPoint = vVanishingDirection[tepSize].p;
                        cv::Point2d startPoint = vVanishingDirection[tepSize].q;
                        vector<cv::Point2d> pointPair(2);
                        pointPair.push_back(endPoint);
                        pointPair.push_back(startPoint);
                        mCurrentFrame.vVanishingLiney.push_back(pointPair);
                        for (int k = 0; k < vVanishingDirection[tepSize].rndpts3d.size(); k++)
                            mCurrentFrame.vVaishingLinePCy.push_back(vVanishingDirection[tepSize].rndpts3d[k]);
                    } else if (a == 3) {
                        cv::Point2d endPoint = vVanishingDirection[tepSize].p;
                        cv::Point2d startPoint = vVanishingDirection[tepSize].q;
                        vector<cv::Point2d> pointPair(2);
                        pointPair.push_back(endPoint);
                        pointPair.push_back(startPoint);
                        mCurrentFrame.vVanishingLinez.push_back(pointPair);
                        for (int k = 0; k < vVanishingDirection[tepSize].rndpts3d.size(); k++)
                            mCurrentFrame.vVaishingLinePCz.push_back(vVanishingDirection[tepSize].rndpts3d[k]);
                    }
                }


            }
        }
        //cout<<"a=1:"<<mCurrentFrame.vSurfaceNormalx.size()<<",a =2:"<<mCurrentFrame.vSurfaceNormaly.size()<<", a=3:"<<mCurrentFrame.vSurfaceNormalz.size()<<endl;
        //cout<<"m_j_selected.push_back(temp)"<<m_j_selected.size()<<endl;

        if (m_j_selected.size() > numOfSN) {
            // note: 在进行mean shit 之前需要对离群点进行剔除
            // reject outlier: 统计滤波
            /*
             *Statistical filtering  || Radius filtering
             *
             * */
//            vector<cv::Point2d> mm_j_selected = Outlier_Filter(m_j_selected);
            sMS tempMeanShift = MeanShift(m_j_selected);
            cv::Point2d s_j = tempMeanShift.centerOfShift;// MeanShift(m_j_selected);
            float s_j_density = tempMeanShift.density;
            //cout<<"tracking:s_j"<<s_j.x<<","<<s_j.y<<endl;

            float alfa = norm(s_j);
            float ma_x = tan(alfa) / alfa * s_j.x;
            float ma_y = tan(alfa) / alfa * s_j.y;
            cv::Mat temp1 = cv::Mat::zeros(cv::Size(1, 3), CV_32F);
            temp1.at<float>(0, 0) = ma_x;
            temp1.at<float>(1, 0) = ma_y;
            temp1.at<float>(2, 0) = 1;
            cv::Mat rtemp = R_mc.t();
            R_cm_Rec = rtemp * temp1;
            R_cm_Rec = R_cm_Rec / norm(R_cm_Rec); //列向量
            RandDen.R_cm_Rec = R_cm_Rec;
            RandDen.s_j_density = s_j_density;

            return RandDen;
        }
        RandDen.R_cm_Rec = R_cm_NULL;
        return RandDen;
    }

    /*refer Paper: Divide and Conquer Efficient  ensity-based Tracking of 3D Sensors in Manhattan Worlds
     * 将点云的法向量或线的灭点投影到指定轴的圆锥内
     * a : 指定的轴
     * R_mc: Camera to Mahattam
     * vTempSurfaceNormal: 当前帧提取到平面点云的法向量
     * vVanishingDirection： 线的灭点
     * */
    axiSNV Tracking::ProjectSN2Conic(int a, const cv::Mat &R_mc, const vector<SurfaceNormal> &vTempSurfaceNormal,
                                     vector<FrameLine> &vVanishingDirection) {
        int numInConic = 0;
        vector<cv::Point2d> m_j_selected;
        cv::Mat R_cm_Rec = cv::Mat::zeros(cv::Size(3, 3), CV_32F);
        cv::Mat R_cm_NULL = cv::Mat::zeros(cv::Size(1, 3), CV_32F);
        //cv::Mat R_mc=cv::Mat::zeros(cv::Size(3,3),CV_32F);
        vector<SurfaceNormal> vSNCadidate;
        axiSNV tempaxiSNV;
        tempaxiSNV.axis = a;


        size_t sizeOfSurfaceNormal = vTempSurfaceNormal.size() + vVanishingDirection.size();
        tempaxiSNV.SNVector.reserve(sizeOfSurfaceNormal);
//        cout << "size of SN" << sizeOfSurfaceNormal << endl;
        for (size_t i = 0; i < sizeOfSurfaceNormal; i++) {

            cv::Point3f n_ini;
            // 先拿面开刀
            if (i < vTempSurfaceNormal.size()) {
                // Rmc * [x,y,z] = 为投影在Manhattan各个轴上的份量 x y z
                n_ini.x = R_mc.at<float>(0, 0) * vTempSurfaceNormal[i].normal.x +
                          R_mc.at<float>(0, 1) * vTempSurfaceNormal[i].normal.y +
                          R_mc.at<float>(0, 2) * vTempSurfaceNormal[i].normal.z;
                n_ini.y = R_mc.at<float>(1, 0) * vTempSurfaceNormal[i].normal.x +
                          R_mc.at<float>(1, 1) * vTempSurfaceNormal[i].normal.y +
                          R_mc.at<float>(1, 2) * vTempSurfaceNormal[i].normal.z;
                n_ini.z = R_mc.at<float>(2, 0) * vTempSurfaceNormal[i].normal.x +
                          R_mc.at<float>(2, 1) * vTempSurfaceNormal[i].normal.y +
                          R_mc.at<float>(2, 2) * vTempSurfaceNormal[i].normal.z;

                // 计算x y 轴上的合力， 要是完全符合 那他应该为0 ， 整体的向量应该为[0,0,1]
                // 这里给他画了一个圆，半径为0.25，只要x,y的合力在这个圆内，那我就认为他的主力军是指向我z轴的
                double lambda = sqrt(n_ini.x * n_ini.x + n_ini.y * n_ini.y);//at(k).y*n_a.at(k).y);
                //cout<<lambda<<endl;
                // note: 我认为对灭点应该增加一些权重，因为本身也没有几根线特征，不加权重的话灭点多起的作用微乎其微
                if (lambda < sin(0.2018)) //0.25
                {
                    //vSNCadidate.push_back(vTempSurfaceNormal[i]);
                    //numInConic++;
                    // 将向量在这个轴a的圆锥内的向量保存起来
                    tempaxiSNV.SNVector.push_back(vTempSurfaceNormal[i]);
                    mvpSurfaceNormals_M.bsurfacenormal_inline[i] = true;
                }
            } else {
                // 将线的灭点同样进行投影，放进来一块清算，整体流程跟面是一样的，只不过他的圆锥开的口小一点
                //cout<<"vanishing"<<endl;
                int tepSize = i - vTempSurfaceNormal.size();
                //cout<<vVanishingDirection[tepSize].direction.x<<"vanishing"<<endl;

                n_ini.x = R_mc.at<float>(0, 0) * vVanishingDirection[tepSize].direction.x +
                          R_mc.at<float>(0, 1) * vVanishingDirection[tepSize].direction.y +
                          R_mc.at<float>(0, 2) * vVanishingDirection[tepSize].direction.z;
                n_ini.y = R_mc.at<float>(1, 0) * vVanishingDirection[tepSize].direction.x +
                          R_mc.at<float>(1, 1) * vVanishingDirection[tepSize].direction.y +
                          R_mc.at<float>(1, 2) * vVanishingDirection[tepSize].direction.z;
                n_ini.z = R_mc.at<float>(2, 0) * vVanishingDirection[tepSize].direction.x +
                          R_mc.at<float>(2, 1) * vVanishingDirection[tepSize].direction.y +
                          R_mc.at<float>(2, 2) * vVanishingDirection[tepSize].direction.z;

                double lambda = sqrt(n_ini.x * n_ini.x + n_ini.y * n_ini.y);//at(k).y*n_a.at(k).y);
                //cout<<lambda<<endl;
                if (lambda < sin(0.1018)) //0.25
                {
                    //vSNCadidate.push_back(vTempSurfaceNormal[i]);
                    //numInConic++;
                    tempaxiSNV.Linesvector.push_back(vVanishingDirection[tepSize]);
                }
            }
        }

        return tempaxiSNV;//numInConic;

    }

    /* @brief 对投影的法向量进行滤波处理，剔除离群点。
     * @brief 统计滤波
     * @param[in] origin_point 原始投影法向量
     * wangwen
     * */
    vector<cv::Point2d> Tracking::Outlier_Filter(vector<cv::Point2d> &origin_point)
    {
        vector<cv::Point2d> final_points;

        // 临近点的点数根据总体法向量的个数来确定比较稳妥

        // 将数据保存起来 txt  单独分析



        ofstream outfile;
        outfile.open("/home/wangwen/catkin_ws/test.txt");

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>); // 用完要释放掉
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>); // 用完要释放掉
        cout << "origin_point.size()"<< origin_point.size() << endl;
        outfile << origin_point.size() << endl;
        for(size_t i=0 ;i < origin_point.size();i++)
        {
            pcl::PointXYZ p;
            p.x = origin_point[i].x;
            p.y = origin_point[i].y;
            p.z = 1;
            outfile << p.x << " " << p.y<< endl;
            cloud->push_back(p);
        }
        outfile.close();
        cout << "wait***********"<< endl;
//        cv::waitKey(500);

//        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> Static;   //创建滤波器对象
//        Static.setInputCloud (cloud);                           //设置待滤波的点云
//        Static.setMeanK(500);                               //设置在进行统计时考虑查询点临近点数
//        Static.setStddevMulThresh (0.05);                      //设置判断是否为离群点的阀值
//        Static.filter (*cloud_filtered);                    //存储
//
//        for(size_t i=0 ;i < cloud_filtered->size();i++)
//        {
//            cv::Point2d p;
//            p.x = cloud_filtered->points[i].x;
//            p.y = cloud_filtered->points[i].y;
//            final_points.push_back(p);
//        }
//        cout << "final_points.size()"<< final_points.size() << endl;
        /* 可视化 */

        return origin_point;
    }

    // 重点！！！
    // 利用线面特征来实现追踪
    /**
    * @brief 获取曼哈顿到相机的位姿变换
    *
    * @param[in] mLastRcm                it is used as initialization point to find current unknown orientation of MF Rck_M
    * @param[in] vSurfaceNormal          深度图点云表面法向量
    * @param[in] vVanishingDirection      线特征
    */
    cv::Mat Tracking::TrackManhattanFrame(cv::Mat &mLastRcm, vector<SurfaceNormal> &vSurfaceNormal,
                                          vector<FrameLine> &vVanishingDirection) {
//        cout << "begin Tracking" << endl;
        cv::Mat R_cm_update = mLastRcm.clone();
        int isTracked = 0;
        vector<double> denTemp(3, 0.00001);

        for (int i = 0; i <1; i++) {

            cv::Mat R_cm = R_cm_update;//cv::Mat::eye(cv::Size(3,3),CV_32FC1);  // 对角线为1的对角矩阵(3, 3, CV_32FC1);
            int directionFound1 = 0;
            int directionFound2 = 0;
            int directionFound3 = 0; //三个方向
            int numDirectionFound = 0;

            vector<axiSNVector> vaxiSNV(4);// surface normal vector
            vector<int> numInCone = vector<int>(3, 0);
            vector<cv::Point2f> vDensity;
            //chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
            // 三个主轴
            // Transform the current surface normal vector nk expressed in the camera frame into n_k' expressed in MF
            // 法向量各找各妈
            for (int a = 1; a < 4; a++) {
                // 遍历每个主轴方向上的圆锥
                //在每个conic有多少 点
                cv::Mat R_mc = cv::Mat::zeros(cv::Size(3, 3), CV_32F);
                int c1 = (a + 3) % 3;
                int c2 = (a + 4) % 3;
                int c3 = (a + 5) % 3;
                // R_mc = R_cm的xyz坐标轴进行切换
                // TODO 他这样进行不断的切换轴的目的是？
                R_mc.at<float>(0, 0) = R_cm.at<float>(0, c1);
                R_mc.at<float>(0, 1) = R_cm.at<float>(0, c2);
                R_mc.at<float>(0, 2) = R_cm.at<float>(0, c3);
                R_mc.at<float>(1, 0) = R_cm.at<float>(1, c1);
                R_mc.at<float>(1, 1) = R_cm.at<float>(1, c2);
                R_mc.at<float>(1, 2) = R_cm.at<float>(1, c3);
                R_mc.at<float>(2, 0) = R_cm.at<float>(2, c1);
                R_mc.at<float>(2, 1) = R_cm.at<float>(2, c2);
                R_mc.at<float>(2, 2) = R_cm.at<float>(2, c3);
                cv::Mat R_mc_new = R_mc.t();
//                cout << "R_mc_new" << R_mc_new << endl;
                // 投影法向量到圆锥-Conic
                // a代表第几个轴
                // 获取到这个轴圆锥内的法向量
                vaxiSNV[a - 1] = ProjectSN2Conic(a, R_mc_new, vSurfaceNormal, vVanishingDirection);
                // 计算有多少向量在这个轴上
                numInCone[a - 1] = vaxiSNV[a - 1].SNVector.size();
                //cout<<"2 a:"<<vaxiSNV[a-1].axis<<",vector:"<<numInCone[a - 1]<<endl;
            }
            //chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
            //chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
            //cout << "first sN time: " << time_used.count() << endl;
            // 分布在各个轴上的最小法向量个数
            int minNumOfSN = vSurfaceNormal.size() / 20;
            //cout<<"minNumOfSN"<<minNumOfSN<<endl;
            //排序  a<b<c
            int a = numInCone[0];
            int b = numInCone[1];
            int c = numInCone[2];
            //cout<<"a:"<<a<<",b:"<<b<<",c:"<<c<<endl;
            // 将各个轴上的法向量数量进行排序
            int temp = 0;
            if (a > b) temp = a, a = b, b = temp;
            if (b > c) temp = b, b = c, c = temp;
            if (a > b) temp = a, a = b, b = temp;
            //cout<<"sequence  a:"<<a<<",b:"<<b<<",c:"<<c<<endl;
            if (b < minNumOfSN) {
                minNumOfSN = (b + a) / 2;
                cout << " normal vector deficiency... change minNumOfSN:thr" << minNumOfSN << endl;
            }

            //cout<<"new  minNumOfSN"<<minNumOfSN<<endl;
            //chrono::steady_clock::time_point t3 = chrono::steady_clock::now();
            //
            for (int a = 1; a < 4; a++) {
                cv::Mat R_mc = cv::Mat::zeros(cv::Size(3, 3), CV_32F);
                int c1 = (a + 3) % 3;
                int c2 = (a + 4) % 3;
                int c3 = (a + 5) % 3;
                R_mc.at<float>(0, 0) = R_cm.at<float>(0, c1);
                R_mc.at<float>(0, 1) = R_cm.at<float>(0, c2);
                R_mc.at<float>(0, 2) = R_cm.at<float>(0, c3);
                R_mc.at<float>(1, 0) = R_cm.at<float>(1, c1);
                R_mc.at<float>(1, 1) = R_cm.at<float>(1, c2);
                R_mc.at<float>(1, 2) = R_cm.at<float>(1, c3);
                R_mc.at<float>(2, 0) = R_cm.at<float>(2, c1);
                R_mc.at<float>(2, 1) = R_cm.at<float>(2, c2);
                R_mc.at<float>(2, 2) = R_cm.at<float>(2, c3);
                cv::Mat R_mc_new = R_mc.t();
                vector<SurfaceNormal> *tempVVSN;
                vector<FrameLine> *tempLineDirection;
                // 上面的for是做坐标轴进行旋转 先面的for是找到与上面进行旋转对应的轴
                // 然后获取对应的 点云法向量以及线的灭点
                for (int i = 0; i < 3; i++) {
                    if (vaxiSNV[i].axis == a) {

                        tempVVSN = &vaxiSNV[i].SNVector;
                        tempLineDirection = &vaxiSNV[i].Linesvector;
                        break;
                    }

                }
                // 将隶属于这个轴上的法向量投影到MF
                ResultOfMS RD_temp = ProjectSN2MF(a, R_mc_new, *tempVVSN, *tempLineDirection, minNumOfSN);

                //chrono::steady_clock::time_point t4 = chrono::steady_clock::now();
                //chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t4-t3);
                //cout << "second SN time: " << time_used.count() << endl;

                //cout << "test projectSN2MF" << ra << endl;
                if (sum(RD_temp.R_cm_Rec)[0] != 0) {
                    numDirectionFound += 1;
                    if (a == 1) directionFound1 = 1;//第一个轴
                    else if (a == 2) directionFound2 = 1;
                    else if (a == 3) directionFound3 = 1;
                    R_cm_update.at<float>(0, a - 1) = RD_temp.R_cm_Rec.at<float>(0, 0);
                    R_cm_update.at<float>(1, a - 1) = RD_temp.R_cm_Rec.at<float>(1, 0);
                    R_cm_update.at<float>(2, a - 1) = RD_temp.R_cm_Rec.at<float>(2, 0);
                    //RD_temp.s_j_density;

                    vDensity.push_back(cv::Point2f(RD_temp.axis, RD_temp.s_j_density));

                }
            }

            if (numDirectionFound < 2) {
                cout << "oh, it has happened" << endl;
                R_cm_update = R_cm;
                numDirectionFound = 0;
                isTracked = 0;
                directionFound1 = 0;
                directionFound2 = 0;
                directionFound3 = 0;
                break;
            } else if (numDirectionFound == 2) {
                if (directionFound1 && directionFound2) {
                    cv::Mat v1 = R_cm_update.colRange(0, 1).clone();
                    cv::Mat v2 = R_cm_update.colRange(1, 2).clone();
                    cv::Mat v3 = v1.cross(v2);
                    R_cm_update.at<float>(0, 2) = v3.at<float>(0, 0);
                    R_cm_update.at<float>(1, 2) = v3.at<float>(1, 0);
                    R_cm_update.at<float>(2, 2) = v3.at<float>(2, 0);
                    if (abs(cv::determinant(R_cm_update) + 1) < 0.5) {
                        R_cm_update.at<float>(0, 2) = -v3.at<float>(0, 0);
                        R_cm_update.at<float>(1, 2) = -v3.at<float>(1, 0);
                        R_cm_update.at<float>(2, 2) = -v3.at<float>(2, 0);
                    }

                } else if (directionFound2 && directionFound3) {
                    cv::Mat v2 = R_cm_update.colRange(1, 2).clone();
                    cv::Mat v3 = R_cm_update.colRange(2, 3).clone();
                    cv::Mat v1 = v3.cross(v2);
                    R_cm_update.at<float>(0, 0) = v1.at<float>(0, 0);
                    R_cm_update.at<float>(1, 0) = v1.at<float>(1, 0);
                    R_cm_update.at<float>(2, 0) = v1.at<float>(2, 0);
                    if (abs(cv::determinant(R_cm_update) + 1) < 0.5) {
                        R_cm_update.at<float>(0, 0) = -v1.at<float>(0, 0);
                        R_cm_update.at<float>(1, 0) = -v1.at<float>(1, 0);
                        R_cm_update.at<float>(2, 0) = -v1.at<float>(2, 0);
                    }
                } else if (directionFound1 && directionFound3) {
                    cv::Mat v1 = R_cm_update.colRange(0, 1).clone();
                    cv::Mat v3 = R_cm_update.colRange(2, 3).clone();
                    cv::Mat v2 = v1.cross(v3);
                    R_cm_update.at<float>(0, 1) = v2.at<float>(0, 0);
                    R_cm_update.at<float>(1, 1) = v2.at<float>(1, 0);
                    R_cm_update.at<float>(2, 1) = v2.at<float>(2, 0);
                    if (abs(cv::determinant(R_cm_update) + 1) < 0.5) {
                        R_cm_update.at<float>(0, 1) = -v2.at<float>(0, 0);
                        R_cm_update.at<float>(1, 1) = -v2.at<float>(1, 0);
                        R_cm_update.at<float>(2, 1) = -v2.at<float>(2, 0);
                    }

                }
            }
            //cout<<"svd before"<<R_cm_update<<endl;
            SVD svd;
            cv::Mat U, W, VT;

            svd.compute(R_cm_update, W, U, VT);

            R_cm_update = U* VT;
            vDensity.clear();
            if (acos((trace(R_cm.t() * R_cm_update)[0] - 1.0)) / 2 < 0.001) {
                cout << "go outside" << endl;
                break;
            }
        }
        isTracked = 1;
        return R_cm_update.clone();
    }

    sMS Tracking::MeanShift(vector<cv::Point2d> &v2D) {
        sMS tempMS;
        int numPoint = v2D.size();
        float density;
        cv::Point2d nominator;
        double denominator = 0;
        double nominator_x = 0;
        double nominator_y = 0;
        for (int i = 0; i < numPoint; i++) {
            double k = exp(-20 * norm(v2D.at(i)) * norm(v2D.at(i)));
            nominator.x += k * v2D.at(i).x;
            nominator.y += k * v2D.at(i).y;
            denominator += k;
        }
        tempMS.centerOfShift = nominator / denominator;
        tempMS.density = denominator / numPoint;

        return tempMS;
    }

    void Tracking::StereoInitialization() {
        // 初始化要求特征点数量超过50,线特征超过15
        if (mCurrentFrame.N > 50 || mCurrentFrame.NL > 15) {
            // Set Frame pose to the origin
            // 设定初始位姿为单位旋转，平移为0
            cv::Mat E = cv::Mat::eye(4, 4, CV_32F);
            cv::Mat Rotation_cw = Rotation_cm;

            Rotation_cw.copyTo(E.rowRange(0,3).colRange(0,3));

            mCurrentFrame.SetPose(E);
            cout <<"**********##########\n"<< E << endl;
            // Create KeyFrame
            // 初始化时将当前帧设定为关键帧
            // KeyFrame包含Frame、地图3D点、以及BoW

            KeyFrame *pKFini = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB,this->mImRGB, this->mCurrentFrame.depth);
//            KeyFrame *pKFini = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);
            // Insert KeyFrame in the map
            // 在地图中添加初始关键帧
            mpMap->AddKeyFrame(pKFini);

            // Create MapPoints and asscoiate to KeyFrame
            // 为每一个特征点构造地图点
            for (int i = 0; i < mCurrentFrame.N; i++) {
                float z = mCurrentFrame.mvDepth[i];
                if (z > 0) {
                    // 通过反投影得到该特征点的世界坐标系下的3D坐标
                    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                    // 将3D点构造为MapPoint
                    MapPoint *pNewMP = new MapPoint(x3D, pKFini, mpMap);
                    // 为该MapPoint添加属性：
                    // a.观测到该MapPoint的关键帧
                    // b.该MapPoint的描述子
                    // c.该MapPoint的平均观测方向和深度范围
                    pNewMP->AddObservation(pKFini, i);
                    pKFini->AddMapPoint(pNewMP, i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    // 在地图中添加该MapPoint
                    mpMap->AddMapPoint(pNewMP);
                    // 将该MapPoint添加到当前帧的mvpMapPoints中
                    // 为当前Frame的特征点与MapPoint之间建立索引
                    mCurrentFrame.mvpMapPoints[i] = pNewMP;
                }
            }


//            for (int i = 0; i < mCurrentFrame.NL; i++) {
//
//                float z = mCurrentFrame.mvDepthLine[i];
//
//                if (z > 0) {
//                    Vector6d line3D = mCurrentFrame.obtain3DLine(i);
//                    MapLine *pNewML = new MapLine(line3D, pKFini, mpMap);
//                    pNewML->AddObservation(pKFini, i);
//                    pKFini->AddMapLine(pNewML, i);
//                    pNewML->ComputeDistinctiveDescriptors();
//                    pNewML->UpdateAverageDir();
//                    mpMap->AddMapLine(pNewML);
//                    mCurrentFrame.mvpMapLines[i] = pNewML;
//                }
//            }

            for (int i = 0; i < mCurrentFrame.mnPlaneNum; ++i) {
                // 初始化时
                // 将检测到的每一个面 投影到世界坐标系下
                cv::Mat p3D = mCurrentFrame.ComputePlaneWorldCoeff(i);
                // 创建转换到世界坐标系下的平面
                MapPlane *pNewMP = new MapPlane(p3D, pKFini, mpMap);
                // 添加观测到该平面的关键帧  该平面 在关键帧中检测的第id个平面中看到
                // 这个世界下的面在当前关键帧可观测，且对应当前帧平面的id
                pNewMP->AddObservation(pKFini,i);
                pKFini->AddMapPlane(pNewMP, i);
                pNewMP->UpdateCoefficientsAndPoints();
                mpMap->AddMapPlane(pNewMP);
                // 这里相机plane投影至世界坐标系下的Plane
                mCurrentFrame.mvpMapPlanes[i] = pNewMP;
            }

//            mpPointCloudMapping->print();


            mpLocalMapper->InsertKeyFrame(pKFini);

            mLastFrame = Frame(mCurrentFrame);
            mnLastKeyFrameId = mCurrentFrame.mnId;
            mpLastKeyFrame = pKFini;

            mvpLocalKeyFrames.push_back(pKFini);
            mvpLocalMapPoints = mpMap->GetAllMapPoints();
            mvpLocalMapLines = mpMap->GetAllMapLines();

            mpReferenceKF = pKFini;
            mCurrentFrame.mpReferenceKF = pKFini;

            mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
            mpMap->SetReferenceMapLines(mvpLocalMapLines);


            mpMap->mvpKeyFrameOrigins.push_back(pKFini);

            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

            mState = OK;
        }
    }

    void Tracking::MonocularInitialization() {
        int num = 100;
        // 如果单目初始器还没有没创建，则创建单目初始器
        if (!mpInitializer) {
            // Set Reference Frame
            if (mCurrentFrame.mvKeys.size() > num) {
                // step 1：得到用于初始化的第一帧，初始化需要两帧
                mInitialFrame = Frame(mCurrentFrame);
                // 记录最近的一帧
                mLastFrame = Frame(mCurrentFrame);
                // mvbPreMatched最大的情况就是当前帧所有的特征点都被匹配上
                mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
                for (size_t i = 0; i < mCurrentFrame.mvKeysUn.size(); i++)
                    mvbPrevMatched[i] = mCurrentFrame.mvKeysUn[i].pt;

                if (mpInitializer)
                    delete mpInitializer;

                // 由当前帧构造初始化器， sigma:1.0    iterations:200
                mpInitializer = new Initializer(mCurrentFrame, 1.0, 200);

                fill(mvIniMatches.begin(), mvIniMatches.end(), -1);

                return;
            }
        } else {
            // Try to initialize
            if ((int) mCurrentFrame.mvKeys.size() <= num) {
                delete mpInitializer;
                mpInitializer = static_cast<Initializer *>(NULL);
                fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
                return;
            }

            // Find correspondences
            ORBmatcher matcher(0.9, true);
            int nmatches = matcher.SearchForInitialization(mInitialFrame, mCurrentFrame, mvbPrevMatched, mvIniMatches,
                                                           100);

            LSDmatcher lmatcher;   //建立线特征之间的匹配
            int lineMatches = lmatcher.SerachForInitialize(mInitialFrame, mCurrentFrame, mvLineMatches);

            if (nmatches < 100) {
                delete mpInitializer;
                mpInitializer = static_cast<Initializer *>(NULL);
                return;
            }

            cv::Mat Rcw; // Current Camera Rotation
            cv::Mat tcw; // Current Camera Translation
            vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)


#if 0
                                                                                                                                    if(mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated))
    {
        for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++)
        {
            if(mvIniMatches[i]>=0 && !vbTriangulated[i])
            {
                mvIniMatches[i]=-1;
                nmatches--;
            }
        }

        // Set Frame Poses
        // 将初始化的第一帧作为世界坐标系，因此第一帧变换矩阵为单位矩阵
        mInitialFrame.SetPose(cv::Mat::eye(4,4,CV_32F));
        cv::Mat Tcw = cv::Mat::eye(4,4,CV_32F);
        Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
        tcw.copyTo(Tcw.rowRange(0,3).col(3));
        mCurrentFrame.SetPose(Tcw);

        // step6：将三角化得到的3D点包装成MapPoints
        /// 如果要修改，应该是从这个函数开始
        CreateInitialMapMonocular();
    }
#else
            if (0)//mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated, mvLineMatches, mvLineS3D, mvLineE3D, mvbLineTriangulated))
            {
                for (size_t i = 0, iend = mvIniMatches.size(); i < iend; i++) {
                    if (mvIniMatches[i] >= 0 && !vbTriangulated[i]) {
                        mvIniMatches[i] = -1;
                        nmatches--;
                    }
                }

                // Set Frame Poses
                // 将初始化的第一帧作为世界坐标系，因此第一帧变换矩阵为单位矩阵
                mInitialFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));
                cv::Mat Tcw = cv::Mat::eye(4, 4, CV_32F);
                Rcw.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
                tcw.copyTo(Tcw.rowRange(0, 3).col(3));
                mCurrentFrame.SetPose(Tcw);

                // step6：将三角化得到的3D点包装成MapPoints
                /// 如果要修改，应该是从这个函数开始
//            CreateInitialMapMonocular();
                CreateInitialMapMonoWithLine();
            }
#endif
        }
    }

    void Tracking::CreateInitialMapMonocular() {
        // Create KeyFrames
        KeyFrame *pKFini = new KeyFrame(mInitialFrame, mpMap, mpKeyFrameDB);
        KeyFrame *pKFcur = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

        pKFini->ComputeBoW();
        pKFcur->ComputeBoW();

        // Insert KFs in the map
        mpMap->AddKeyFrame(pKFini);
        mpMap->AddKeyFrame(pKFcur);

        // Create MapPoints and asscoiate to keyframes
        for (size_t i = 0; i < mvIniMatches.size(); i++) {
            if (mvIniMatches[i] < 0)
                continue;

            //Create MapPoint.
            cv::Mat worldPos(mvIniP3D[i]);

            MapPoint *pMP = new MapPoint(worldPos, pKFcur, mpMap);

            pKFini->AddMapPoint(pMP, i);
            pKFcur->AddMapPoint(pMP, mvIniMatches[i]);

            pMP->AddObservation(pKFini, i);
            pMP->AddObservation(pKFcur, mvIniMatches[i]);

            // b.从众多观测到该MapPoint的特征点中挑选出区分度最高的描述子
            pMP->ComputeDistinctiveDescriptors();

            // c.更新该MapPoint的平均观测方向以及观测距离的范围
            pMP->UpdateNormalAndDepth();

            //Fill Current Frame structure
            mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
            mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

            //Add to Map
            mpMap->AddMapPoint(pMP);
        }

        // Update Connections
        pKFini->UpdateConnections();
        pKFcur->UpdateConnections();

        // Bundle Adjustment
        cout << "New Map created with " << mpMap->MapPointsInMap() << " points" << endl;

        Optimizer::GlobalBundleAdjustemnt(mpMap, 20);

        // Set median depth to 1
        float medianDepth = pKFini->ComputeSceneMedianDepth(2);
        float invMedianDepth = 1.0f / medianDepth;

        if (medianDepth < 0 || pKFcur->TrackedMapPoints(1) < 100) {
            cout << "Wrong initialization, reseting..." << endl;
            Reset();
            return;
        }

        // Scale initial baseline
        cv::Mat Tc2w = pKFcur->GetPose();
        Tc2w.col(3).rowRange(0, 3) = Tc2w.col(3).rowRange(0, 3) * invMedianDepth;
        pKFcur->SetPose(Tc2w);

        // Scale points
        vector<MapPoint *> vpAllMapPoints = pKFini->GetMapPointMatches();
        for (size_t iMP = 0; iMP < vpAllMapPoints.size(); iMP++) {
            if (vpAllMapPoints[iMP]) {
                MapPoint *pMP = vpAllMapPoints[iMP];
                pMP->SetWorldPos(pMP->GetWorldPos() * invMedianDepth);
            }
        }

        mpLocalMapper->InsertKeyFrame(pKFini);
        mpLocalMapper->InsertKeyFrame(pKFcur);

        mCurrentFrame.SetPose(pKFcur->GetPose());
        mnLastKeyFrameId = mCurrentFrame.mnId;
        mpLastKeyFrame = pKFcur;

        mvpLocalKeyFrames.push_back(pKFcur);
        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints = mpMap->GetAllMapPoints();
        mpReferenceKF = pKFcur;
        mCurrentFrame.mpReferenceKF = pKFcur;

        mLastFrame = Frame(mCurrentFrame);

        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

        mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

        mpMap->mvpKeyFrameOrigins.push_back(pKFini);

        mState = OK;  //至此，初始化成功
    }

#if 1

/**
* @brief 为单目摄像头三角化生成带有线特征的Map，包括MapPoints和MapLine
*/
    void Tracking::CreateInitialMapMonoWithLine() {
        // step1:
        KeyFrame *pKFini = new KeyFrame(mInitialFrame, mpMap, mpKeyFrameDB);
        KeyFrame *pKFcur = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

        // step2：
        pKFini->ComputeBoW();
        pKFcur->ComputeBoW();

        // step3：
        mpMap->AddKeyFrame(pKFini);
        mpMap->AddKeyFrame(pKFcur);

        // step4：
        for (size_t i = 0; i < mvIniMatches.size(); i++) {
            if (mvIniMatches[i] < 0)
                continue;

            // Create MapPoint
            cv::Mat worldPos(mvIniP3D[i]);

            // step4.1：
            MapPoint *pMP = new MapPoint(worldPos, pKFcur, mpMap);

            // step4.2：

            // step4.3：
            pKFini->AddMapPoint(pMP, i);
            pKFcur->AddMapPoint(pMP, mvIniMatches[i]);

            // a.
            pMP->AddObservation(pKFini, i);
            pMP->AddObservation(pKFcur, mvIniMatches[i]);

            // b.
            pMP->UpdateNormalAndDepth();

            // Fill Current Frame structure
            mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
            mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

            // Add to Map
            // step4.4：
            mpMap->AddMapPoint(pMP);
        }

        // step5：
        for (size_t i = 0; i < mvLineMatches.size(); i++) {
            if (!mvbLineTriangulated[i])
                continue;

            // Create MapLine
            Vector6d worldPos;
            worldPos << mvLineS3D[i].x, mvLineS3D[i].y, mvLineS3D[i].z, mvLineE3D[i].x, mvLineE3D[i].y, mvLineE3D[i].z;

            //step5.1：
            MapLine *pML = new MapLine(worldPos, pKFcur, mpMap);

            pKFini->AddMapLine(pML, i);
            pKFcur->AddMapLine(pML, i);

            //a.
            pML->AddObservation(pKFini, i);
            pML->AddObservation(pKFcur, i);

            //b.
            pML->ComputeDistinctiveDescriptors();

            //c.
            pML->UpdateAverageDir();

            // Fill Current Frame structure
            mCurrentFrame.mvpMapLines[i] = pML;
            mCurrentFrame.mvbLineOutlier[i] = false;

            // step5.4: Add to Map
            mpMap->AddMapLine(pML);
        }


        float medianDepth = pKFini->ComputeSceneMedianDepth(2);
        float invMedianDepth = 1.0f / medianDepth;

//        cout << "medianDepth = " << medianDepth << endl;
//        cout << "pKFcur->TrackedMapPoints(1) = " << pKFcur->TrackedMapPoints(1) << endl;

        if (medianDepth < 0 || pKFcur->TrackedMapPoints(1) < 80) {
            cout << "Wrong initialization, reseting ... " << endl;
            Reset();
            return;
        }

        // Scale initial baseline
        cv::Mat Tc2w = pKFcur->GetPose();
        Tc2w.col(3).rowRange(0, 3) = Tc2w.col(3).rowRange(0, 3) * invMedianDepth;
        pKFcur->SetPose(Tc2w);

        // Scale Points
        vector<MapPoint *> vpAllMapPoints = pKFini->GetMapPointMatches();
        for (size_t iMP = 0; iMP < vpAllMapPoints.size(); ++iMP) {
            if (vpAllMapPoints[iMP]) {
                MapPoint *pMP = vpAllMapPoints[iMP];
                pMP->SetWorldPos(pMP->GetWorldPos() * invMedianDepth);
            }
        }

        // Scale Line Segments
        vector<MapLine *> vpAllMapLines = pKFini->GetMapLineMatches();
        for (size_t iML = 0; iML < vpAllMapLines.size(); iML++) {
            if (vpAllMapLines[iML]) {
                MapLine *pML = vpAllMapLines[iML];
                pML->SetWorldPos(pML->GetWorldPos() * invMedianDepth);
            }
        }

        mpLocalMapper->InsertKeyFrame(pKFini);
        mpLocalMapper->InsertKeyFrame(pKFcur);

        mCurrentFrame.SetPose(pKFcur->GetPose());
        mnLastKeyFrameId = mCurrentFrame.mnId;
        mpLastKeyFrame = pKFcur;

        mvpLocalKeyFrames.push_back(pKFcur);
        mvpLocalKeyFrames.push_back(pKFini);
        mvpLocalMapPoints = mpMap->GetAllMapPoints();
        mvpLocalMapLines = mpMap->GetAllMapLines();
        mpReferenceKF = pKFcur;
        mCurrentFrame.mpReferenceKF = pKFcur;

        mLastFrame = Frame(mCurrentFrame);

        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
        mpMap->SetReferenceMapLines(mvpLocalMapLines);

        mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

        mpMap->mvpKeyFrameOrigins.push_back(pKFini);

        mState = OK;
    }

#endif

    void Tracking::StereoInitializationWithMap()
    {
        if(mCurrentFrame.N>100)
        {

            // New map is loaded. First, let's relocalize the current frame.
            mCurrentFrame.SetPose(cv::Mat::eye(4,4,CV_32F));

            bool bOK;
            bOK = Relocalization(false);
            if(bOK != false)
                cout << "Relocalization succeful!"<< endl;
            // Then the rest few lines in this funtion is the same as the last half part of the Tracking::Track() function.
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

            // If we have an initial estimation of the camera pose and matching. Track the local map.
            if(!mbOnlyTracking)
            {
                if(bOK)
                     cout << "Tracking.cc :: Track, now into TrackLocalMap()." << endl;
                    bOK = TrackLocalMap(false);
            }
            else
            {
                // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
                // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
                // the camera we will use the local map again.
                if(bOK && !mbVO)
                    bOK = TrackLocalMap(false);
            }

            if(bOK){
                 cout << "Tracking.cc :: Track(), Now mState is OK." << endl;
                mState = OK;
            }
            else{
                 cout << "Tracking.cc :: Track(), Now mState is LOST." << endl;
                mState=LOST;
            }

            // Update drawer
            mpFrameDrawer->Update(this);
//            cv::waitKey(0);
            // If tracking were good, check if we insert a keyframe
            if(bOK)
            {
                // Update motion model
                if(!mLastFrame.mTcw.empty())
                {
                    cv::Mat LastTwc = cv::Mat::eye(4,4,CV_32F);
                    mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0,3).colRange(0,3));
                    mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0,3).col(3));
                    mVelocity = mCurrentFrame.mTcw*LastTwc;
                }
                else
                    mVelocity = cv::Mat();

                mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);

                // Clean VO matches
                for(int i=0; i<mCurrentFrame.N; i++)
                {
                    MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
                    if(pMP)
                        if(pMP->Observations()<1)
                        {
                            mCurrentFrame.mvbOutlier[i] = false;
                            mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                        }
                }

                // Delete temporal MapPoints
                for(list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end(); lit!=lend; lit++)
                {
                    MapPoint* pMP = *lit;
                    delete pMP;
                }
                mlpTemporalPoints.clear();

                // Check if we need to insert a new keyframe
                if(NeedNewKeyFrame())
                    CreateNewKeyFrame();

                // We allow points with high innovation (considererd outliers by the Huber Function)
                // pass to the new keyframe, so that bundle adjustment will finally decide
                // if they are outliers or not. We don't want next frame to estimate its position
                // with those points so we discard them in the frame.
                for(int i=0; i<mCurrentFrame.N;i++)
                {
                    if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                        mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL);
                }
            }

            // Reset if the camera get lost soon after initialization
            if(mState==LOST)
            {
                if(mpMap->KeyFramesInMap()<=5)
                {
                    mpSystem->Reset();
                    return;
                }
            }

            if(!mCurrentFrame.mpReferenceKF)
                mCurrentFrame.mpReferenceKF = mpReferenceKF;

            mLastFrame = Frame(mCurrentFrame);
        }
    }



    void Tracking::CheckReplacedInLastFrame() {
        for (int i = 0; i < mLastFrame.N; i++) {
            MapPoint *pMP = mLastFrame.mvpMapPoints[i];

            if (pMP) {
                MapPoint *pRep = pMP->GetReplaced();
                if (pRep) {
                    mLastFrame.mvpMapPoints[i] = pRep;
                }
            }
        }

        for (int i = 0; i < mLastFrame.NL; i++) {
            MapLine *pML = mLastFrame.mvpMapLines[i];

            if (pML) {
                MapLine *pReL = pML->GetReplaced();
                if (pReL) {
                    mLastFrame.mvpMapLines[i] = pReL;
                }
            }
        }

        for (int i = 0; i < mLastFrame.mnPlaneNum; i++) {
            MapPlane *pMP = mLastFrame.mvpMapPlanes[i];

            if (pMP) {
                MapPlane *pRep = pMP->GetReplaced();
                if (pRep) {
                    mLastFrame.mvpMapPlanes[i] = pRep;
                }
            }
        }
    }

    /**
    * @brief 根据恒定速度模型用上一帧地图点来对当前帧进行跟踪
    * Step 1：更新上一帧的位姿；对于双目或RGB-D相机，还会根据深度值生成临时地图点
    * Step 2：根据上一帧特征点对应地图点进行投影匹配
    * Step 3：优化当前帧位姿
    * Step 4：剔除地图点中外点
     * @return 如果匹配数大于10，认为跟踪成功，返回true
 */
    bool Tracking::TrackWithMotionModel(bool bStruct){
        ORBmatcher matcher(0.9, true);
        LSDmatcher lmatcher;
        // 对平面匹配进行初始化操作
        PlaneMatcher pmatcher(mfDThRef, mfAThRef, mfVerTh, mfParTh);

        // Update last frame pose according to its reference keyframe
        // Create "visual odometry" points if in Localization Mode
        // Step 1：更新上一帧的位姿；对于双目或RGB-D相机，还会根据深度值生成临时地图点
        UpdateLastFrame();
        // initial estimate
        mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);

        // Project points seen in previous frame
        int th = 15;
        // 清空当前帧地图点
        fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));
        int nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, th, mSensor == System::MONOCULAR);
//        cout << "TrackWithMotionModel line matching"<< endl;
        int planeMatches=0;
        int lmatches = 0;
        //清空当前帧 线
        if(!mbOnlyTracking) {
        vector<MapLine *> vpMapLineMatches;
        fill(mCurrentFrame.mvpMapLines.begin(), mCurrentFrame.mvpMapLines.end(), static_cast<MapLine *>(NULL));
        lmatches = lmatcher.SearchByDescriptor(mpReferenceKF, mCurrentFrame, vpMapLineMatches);
        mCurrentFrame.mvpMapLines = vpMapLineMatches;
        }

//        cout << "TrackWithMotionModel line matched"<< endl;
        // 匹配点太少
        if (nmatches < 40) {
            fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(),
                 static_cast<MapPoint *>(NULL));
            nmatches = matcher.MatchORBPoints(mCurrentFrame, mLastFrame);
            mUpdateMF = true;
        }
        if(!mbOnlyTracking) {
            planeMatches = pmatcher.SearchMapByCoefficients(mCurrentFrame, mpMap->GetAllMapPlanes(),mRotation_wc);

        }

        int initialMatches = nmatches + lmatches + planeMatches;

        if (initialMatches < 10) {
            cout << "TranslationWithMotionModel: Before: Not enough matches" << endl;
            return false;
        }
//        cout << "start optimaztion "<< endl;
        Optimizer::PoseOptimization(&mCurrentFrame,bStruct);
//        cout << "finised optimaztion "<< endl;
        // 返回当前依据track的可信度    可信度不够，用MF重新优化
//        bool track_quality = true;
//        if(!mbOnlyTracking) {
//             track_quality =  pmatcher.bMatchStatus(mCurrentFrame, mpMap->GetAllMapPlanes(),true,mRotation_wc.t());
//        }
//        if(track_quality == false) {
//            // Optimize frame pose with all matches
//            // 屏蔽掉解耦位姿
//            // 在匹配的时候使用的是匀速模型 仅在位姿优化的时候使用的曼哈顿旋转
//            mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);
//            mRotation_wc.copyTo(mCurrentFrame.mTcw.rowRange(0,3).colRange(0,3));
////             Optimize frame pose with all matches
////            Optimizer::PoseOptimization(&mCurrentFrame,bStruct);
//            Optimizer::TranslationOptimization(&mCurrentFrame,false);
//        }

        // Discard outliers
        float nmatchesMap = 0;
        int nmatchesLineMap = 0;
        int nmatchesPlaneMap = 0;

        for (int i = 0; i < mCurrentFrame.N; i++) {
            if (mCurrentFrame.mvpMapPoints[i]) {
                if (mCurrentFrame.mvbOutlier[i]) {
                    MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                    mCurrentFrame.mvbOutlier[i] = false;
                    pMP->mbTrackInView = false;
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    nmatches--;
                } else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                    nmatchesMap++;
            }
        }

        for (int i = 0; i < mCurrentFrame.NL; i++) {
            if (mCurrentFrame.mvpMapLines[i]) {
                if (mCurrentFrame.mvbLineOutlier[i]) {
                    MapLine *pML = mCurrentFrame.mvpMapLines[i];
                    mCurrentFrame.mvpMapLines[i] = static_cast<MapLine *>(NULL);
                    mCurrentFrame.mvbLineOutlier[i] = false;
                    pML->mbTrackInView = false;
                    pML->mnLastFrameSeen = mCurrentFrame.mnId;
                    lmatches--;
                } else if (mCurrentFrame.mvpMapLines[i]->Observations() > 0)
                    nmatchesLineMap++;
            }

        }

        for (int i = 0; i < mCurrentFrame.mnPlaneNum; i++) {
            if (mCurrentFrame.mvpMapPlanes[i]) {
                if (mCurrentFrame.mvbPlaneOutlier[i]) {
                    mCurrentFrame.mvpMapPlanes[i] = static_cast<MapPlane *>(nullptr);
                    planeMatches--;
                } else
                    nmatchesPlaneMap++;
            }

            if (mCurrentFrame.mvpParallelPlanes[i]) {
                if (mCurrentFrame.mvbParPlaneOutlier[i]) {
                    mCurrentFrame.mvpParallelPlanes[i] = static_cast<MapPlane *>(nullptr);
                    mCurrentFrame.mvbParPlaneOutlier[i] = false;
                }
            }

            if (mCurrentFrame.mvpVerticalPlanes[i]) {
                if (mCurrentFrame.mvbVerPlaneOutlier[i]) {
                    mCurrentFrame.mvpVerticalPlanes[i] = static_cast<MapPlane *>(nullptr);
                    mCurrentFrame.mvbVerPlaneOutlier[i] = false;
                }
            }
        }

        float finalMatches = nmatchesMap + nmatchesLineMap + nmatchesPlaneMap;

        mnMatchesInliers = finalMatches;

        if (mbOnlyTracking) {
            mbVO = nmatchesMap < 6;
            return nmatches > 10;
        }

        if (finalMatches < 7) {
            return false;
        }

        return true;
    }
    bool Tracking::TrackReferenceKeyFrame(bool bStruct) {
        // Compute Bag of Words vector
        // Step 1：将当前帧的描述子转化为BoW向量
        mCurrentFrame.ComputeBoW();

        // We perform first an ORB matching with the reference keyframe
        // If enough matches are found we setup a PnP solver
        // Step 2：通过词袋BoW加速当前帧与参考帧之间的特征点匹配
        ORBmatcher matcher(0.7, true);
        LSDmatcher lmatcher;
        PlaneMatcher pmatcher(mfDThRef, mfAThRef, mfVerTh, mfParTh);

        vector<MapPoint *> vpMapPointMatches;
        vector<MapLine *> vpMapLineMatches;
        vector<pair<int, int>> vLineMatches;

        int nmatches = matcher.SearchByBoW(mpReferenceKF, mCurrentFrame, vpMapPointMatches);
        int lmatches = 0;
        if(!mbOnlyTracking) {
            lmatches = lmatcher.SearchByDescriptor(mpReferenceKF, mCurrentFrame, vpMapLineMatches);
        }
//        vpMapLineMatches = vector<MapLine *>(mCurrentFrame.NL, static_cast<MapLine *>(NULL));
//        int lmatches = 0;

        mCurrentFrame.SetPose(mLastFrame.mTcw);

        int planeMatches = 0;
        if(!mbOnlyTracking) {
            planeMatches = pmatcher.SearchMapByCoefficients(mCurrentFrame, mpMap->GetAllMapPlanes(), mRotation_wc);
        }
//        int planeMatches = 0;

        int initialMatches = nmatches + lmatches + planeMatches;

//        cout << "TranslationEstimation: Before: Point matches: " << nmatches << " , Line Matches:"
//             << lmatches << ", Plane Matches:" << planeMatches << endl;

        if (initialMatches < 5) {
            cout << "******************TranslationEstimation: Before: Not enough matches" << endl;
            return false;
        }

        mCurrentFrame.mvpMapPoints = vpMapPointMatches;
        mCurrentFrame.mvpMapLines = vpMapLineMatches;


        //cout << "translation reference,pose before opti" << mCurrentFrame.mTcw << endl;
//        Optimizer::TranslationOptimization(&mCurrentFrame);
        // 现在的工作是暂时屏蔽掉曼哈顿R对系统的影响，单独考虑点线面
        Optimizer::PoseOptimization(&mCurrentFrame,bStruct);
        //cout << "translation reference,pose after opti" << mCurrentFrame.mTcw << endl;

        int nmatchesMap = 0;
        int nmatchesLineMap = 0;
        int nmatchesPlaneMap = 0;

        // Discard outliers
        for (int i = 0; i < mCurrentFrame.N; i++) {
            if (mCurrentFrame.mvpMapPoints[i]) {
                if (mCurrentFrame.mvbOutlier[i]) {
                    MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                    mCurrentFrame.mvbOutlier[i] = false;
                    pMP->mbTrackInView = false;
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    nmatches--;

                } else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                    nmatchesMap++;
            }
        }

//        for (int i = 0; i < mCurrentFrame.NL; i++) {
//            if (mCurrentFrame.mvpMapLines[i]) {
//                if (mCurrentFrame.mvbLineOutlier[i]) {
//                    MapLine *pML = mCurrentFrame.mvpMapLines[i];
//                    mCurrentFrame.mvpMapLines[i] = static_cast<MapLine *>(NULL);
//                    mCurrentFrame.mvbLineOutlier[i] = false;
//                    pML->mbTrackInView = false;
//                    pML->mnLastFrameSeen = mCurrentFrame.mnId;
//                    lmatches--;
//                } else if (mCurrentFrame.mvpMapLines[i]->Observations() > 0)
//                    nmatchesLineMap++;
//
//            }
//        }

        int nDiscardPlane = 0;
        for (int i = 0; i < mCurrentFrame.mnPlaneNum; i++) {
            if (mCurrentFrame.mvpMapPlanes[i]) {
                if (mCurrentFrame.mvbPlaneOutlier[i]) {
                } else
                    nmatchesPlaneMap++;
            }

            if (mCurrentFrame.mvpParallelPlanes[i]) {
                if (mCurrentFrame.mvbParPlaneOutlier[i]) {
                    mCurrentFrame.mvpParallelPlanes[i] = static_cast<MapPlane *>(nullptr);
                    mCurrentFrame.mvbParPlaneOutlier[i] = false;
                }
            }

            if (mCurrentFrame.mvpVerticalPlanes[i]) {
                if (mCurrentFrame.mvbVerPlaneOutlier[i]) {
                    mCurrentFrame.mvpVerticalPlanes[i] = static_cast<MapPlane *>(nullptr);
                    mCurrentFrame.mvbVerPlaneOutlier[i] = false;
                }
            }
        }

        float finalMatches = nmatchesMap + nmatchesLineMap + nmatchesPlaneMap;

        mnMatchesInliers = finalMatches;

        if ( finalMatches < 3) {
            cout << "TranslationEstimation: After: Not enough matches" << endl;
            mCurrentFrame.SetPose(mLastFrame.mTcw);
            return false;
        }

        return true;
    }
    bool Tracking::TranslationEstimation(bool bStruct) {

        // Compute Bag of Words vector
        // Step 1：将当前帧的描述子转化为BoW向量
        mCurrentFrame.ComputeBoW();

        // We perform first an ORB matching with the reference keyframe
        // If enough matches are found we setup a PnP solver
        // Step 2：通过词袋BoW加速当前帧与参考帧之间的特征点匹配
        ORBmatcher matcher(0.7, true);
        LSDmatcher lmatcher;
        PlaneMatcher pmatcher(mfDThRef, mfAThRef, mfVerTh, mfParTh);

        vector<MapPoint *> vpMapPointMatches;
        vector<MapLine *> vpMapLineMatches;
        vector<pair<int, int>> vLineMatches;

        int nmatches = matcher.SearchByBoW(mpReferenceKF, mCurrentFrame, vpMapPointMatches);
        int lmatches = lmatcher.SearchByDescriptor(mpReferenceKF, mCurrentFrame, vpMapLineMatches);
//        vpMapLineMatches = vector<MapLine *>(mCurrentFrame.NL, static_cast<MapLine *>(NULL));
//        int lmatches = 0;

        mCurrentFrame.SetPose(mLastFrame.mTcw);

        int planeMatches = pmatcher.SearchMapByCoefficients(mCurrentFrame, mpMap->GetAllMapPlanes(),mRotation_wc);
//        int planeMatches = 0;

        int initialMatches = nmatches + lmatches + planeMatches;

//        cout << "TranslationEstimation: Before: Point matches: " << nmatches << " , Line Matches:"
//             << lmatches << ", Plane Matches:" << planeMatches << endl;

        if (initialMatches < 5) {
            cout << "******************TranslationEstimation: Before: Not enough matches" << endl;
            return false;
        }

        mCurrentFrame.mvpMapPoints = vpMapPointMatches;
        mCurrentFrame.mvpMapLines = vpMapLineMatches;


        //cout << "translation reference,pose before opti" << mCurrentFrame.mTcw << endl;
        Optimizer::TranslationOptimization(&mCurrentFrame,bStruct);
        //cout << "translation reference,pose after opti" << mCurrentFrame.mTcw << endl;

        int nmatchesMap = 0;
        int nmatchesLineMap = 0;
        int nmatchesPlaneMap = 0;

        // Discard outliers
        for (int i = 0; i < mCurrentFrame.N; i++) {
            if (mCurrentFrame.mvpMapPoints[i]) {
                if (mCurrentFrame.mvbOutlier[i]) {
                    MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];

                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                    mCurrentFrame.mvbOutlier[i] = false;
                    pMP->mbTrackInView = false;
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    nmatches--;

                } else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                    nmatchesMap++;
            }
        }

//        for (int i = 0; i < mCurrentFrame.NL; i++) {
//            if (mCurrentFrame.mvpMapLines[i]) {
//                if (mCurrentFrame.mvbLineOutlier[i]) {
//                    MapLine *pML = mCurrentFrame.mvpMapLines[i];
//                    mCurrentFrame.mvpMapLines[i] = static_cast<MapLine *>(NULL);
//                    mCurrentFrame.mvbLineOutlier[i] = false;
//                    pML->mbTrackInView = false;
//                    pML->mnLastFrameSeen = mCurrentFrame.mnId;
//                    lmatches--;
//                } else if (mCurrentFrame.mvpMapLines[i]->Observations() > 0)
//                    nmatchesLineMap++;
//
//            }
//        }

        int nDiscardPlane = 0;
        for (int i = 0; i < mCurrentFrame.mnPlaneNum; i++) {
            if (mCurrentFrame.mvpMapPlanes[i]) {
                if (mCurrentFrame.mvbPlaneOutlier[i]) {
                } else
                    nmatchesPlaneMap++;
            }

            if (mCurrentFrame.mvpParallelPlanes[i]) {
                if (mCurrentFrame.mvbParPlaneOutlier[i]) {
                    mCurrentFrame.mvpParallelPlanes[i] = static_cast<MapPlane *>(nullptr);
                    mCurrentFrame.mvbParPlaneOutlier[i] = false;
                }
            }

            if (mCurrentFrame.mvpVerticalPlanes[i]) {
                if (mCurrentFrame.mvbVerPlaneOutlier[i]) {
                    mCurrentFrame.mvpVerticalPlanes[i] = static_cast<MapPlane *>(nullptr);
                    mCurrentFrame.mvbVerPlaneOutlier[i] = false;
                }
            }
        }

        float finalMatches = nmatchesMap + nmatchesLineMap + nmatchesPlaneMap;

        mnMatchesInliers = finalMatches;

        if ( finalMatches < 3) {
            cout << "TranslationEstimation: After: Not enough matches" << endl;
            mCurrentFrame.SetPose(mLastFrame.mTcw);
            return false;
        }

        return true;
    }

    bool Tracking::TranslationWithMotionModel(Eigen::Quaterniond &GroundTruth_R,bool bSturct) {
        ORBmatcher matcher(0.9, true);
        LSDmatcher lmatcher;
        // 对平面匹配进行初始化操作
        PlaneMatcher pmatcher(mfDThRef, mfAThRef, mfVerTh, mfParTh);

        // Update last frame pose according to its reference keyframe
        // Create "visual odometry" points if in Localization Mode
        // Step 1：更新上一帧的位姿；对于双目或RGB-D相机，还会根据深度值生成临时地图点
        UpdateLastFrame();

        // initial estimate
        mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);
        mRotation_wc.copyTo(mCurrentFrame.mTcw.rowRange(0,3).colRange(0,3));
        mCurrentFrame.SetPose(mCurrentFrame.mTcw);
        // Project points seen in previous frame
        int th;
        if (mSensor != System::STEREO)
            th = 15;
        else
            th = 7;// 双目来说相对比较准 搜索范围就比较小
        // 清空当前帧的地图点
        fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));

        // 将上一帧中的额地图点投影到当前帧中进行匹配   这样的话 他们匹配成功的地图点就相互之间构成了观测
        int nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, th, mSensor == System::MONOCULAR);

        vector<MapLine *> vpMapLineMatches;
        int lmatches = lmatcher.SearchByDescriptor(mpReferenceKF, mCurrentFrame, vpMapLineMatches);
        mCurrentFrame.mvpMapLines = vpMapLineMatches;


        // 匹配点太少
        if (nmatches <50) {
            fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));
            nmatches = matcher.MatchORBPoints(mCurrentFrame, mLastFrame);
            mUpdateMF = true;
        }

        // 并不是相邻帧之间的面匹配 而是跟整个地图的平面进行匹配
        int planeMatches = pmatcher.SearchMapByCoefficients(mCurrentFrame, mpMap->GetAllMapPlanes(),mRotation_wc);


        int initialMatches = nmatches + lmatches + planeMatches;

        if (initialMatches < 10) {
            cout << "TranslationWithMotionModel: Before: Not enough matches" << endl;
            return false;
        }

        // Optimize frame pose with all matches
        // 屏蔽掉解耦位姿
        // 在匹配的时候使用的是匀速模型 仅在位姿优化的时候使用的曼哈顿旋转
        mRotation_wc.copyTo(mCurrentFrame.mTcw.rowRange(0,3).colRange(0,3));

        //cout << "translation motion model,pose before opti" << mCurrentFrame.mTcw << endl;
        // Step 4：利用3D-2D投影关系，优化当前帧位姿
        // 按照做的的说法，这里应该是固定R求t
        // 我现在想要做的是，将R解禁    固定R不动应该是在观测边的雅克比J对应的变量置0
        Optimizer::TranslationOptimization(&mCurrentFrame,bSturct);
        //cout << "translation motion model,pose after opti" << mCurrentFrame.mTcw << endl;


        // Discard outliers
        int nmatchesMap = 0;
        int nmatchesLineMap = 0;
        int nmatchesPlaneMap = 0;

        for (int i = 0; i < mCurrentFrame.N; i++) {
            if (mCurrentFrame.mvpMapPoints[i]) {
                if (mCurrentFrame.mvbOutlier[i]) {
                    MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                    mCurrentFrame.mvbOutlier[i] = false;
                    pMP->mbTrackInView = false;
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    nmatches--;
                } else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                    nmatchesMap++;
            }
        }

        for (int i = 0; i < mCurrentFrame.NL; i++) {
            if (mCurrentFrame.mvpMapLines[i]) {
                if (mCurrentFrame.mvbLineOutlier[i]) {
                    MapLine *pML = mCurrentFrame.mvpMapLines[i];
                    mCurrentFrame.mvpMapLines[i] = static_cast<MapLine *>(NULL);
                    mCurrentFrame.mvbLineOutlier[i] = false;
                    pML->mbTrackInView = false;
                    pML->mnLastFrameSeen = mCurrentFrame.mnId;
                    lmatches--;
                } else if (mCurrentFrame.mvpMapLines[i]->Observations() > 0)
                    nmatchesLineMap++;
            }

        }

        int nDiscardPlane = 0;
        for (int i = 0; i < mCurrentFrame.mnPlaneNum; i++) {
            if (mCurrentFrame.mvpMapPlanes[i]) {
                if (mCurrentFrame.mvbPlaneOutlier[i]) {
                } else
                    nmatchesPlaneMap++;
            }

            if (mCurrentFrame.mvpParallelPlanes[i]) {
                if (mCurrentFrame.mvbParPlaneOutlier[i]) {
                    mCurrentFrame.mvpParallelPlanes[i] = static_cast<MapPlane *>(nullptr);
                    mCurrentFrame.mvbParPlaneOutlier[i] = false;
                }
            }

            if (mCurrentFrame.mvpVerticalPlanes[i]) {
                if (mCurrentFrame.mvbVerPlaneOutlier[i]) {
                    mCurrentFrame.mvpVerticalPlanes[i] = static_cast<MapPlane *>(nullptr);
                    mCurrentFrame.mvbVerPlaneOutlier[i] = false;
                }
            }
        }

        if (mbOnlyTracking) {
            mbVO = nmatchesMap < 10;
            return nmatches > 20;
        }

        float finalMatches = nmatchesMap + nmatchesLineMap + nmatchesPlaneMap;

        mnMatchesInliers = finalMatches;

        if (nmatchesMap < 3 ||finalMatches < 3) {
            cout << "TranslationWithMotionModel: After: Not enough matches" << endl;
            mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);
            return false;
        }

        return true;
    }

    void Tracking::UpdateLastFrame() {
        // Update pose according to reference keyframe
        // Step 1：利用参考关键帧更新上一帧在世界坐标系下的位姿
        // 上一普通帧的参考关键帧，注意这里用的是参考关键帧（位姿准）而不是上上一帧的普通帧
        KeyFrame *pRef = mLastFrame.mpReferenceKF;
        // ref_keyframe 到 lastframe的位姿变换
        cv::Mat Tlr = mlRelativeFramePoses.back();
        // 将上一帧的世界坐标系下的位姿计算出来
        // l:last, r:reference, w:world
        // Tlw = Tlr*Trw
        mLastFrame.SetPose(Tlr * pRef->GetPose());
        // 如果上一帧为关键帧，或者单目的情况，则退出
        if (mnLastKeyFrameId == mLastFrame.mnId || mSensor == System::MONOCULAR || !mbOnlyTracking)
            return;

        // Create "visual odometry" MapPoints
        // We sort points according to their measured depth by the stereo/RGB-D sensor
        // Step 2：对于双目或rgbd相机，为上一帧生成新的临时地图点
        // 注意这些地图点只是用来跟踪，不加入到地图中，跟踪完后会删除

        // Step 2.1：得到上一帧中具有有效深度值的特征点（不一定是地图点）
        vector<pair<float, int> > vDepthIdx;
        vDepthIdx.reserve(mLastFrame.N);
        for (int i = 0; i < mLastFrame.N; i++) {
            float z = mLastFrame.mvDepth[i];
            if (z > 0) {
                // vDepthIdx第一个元素是某个点的深度,第二个元素是对应的特征点id
                vDepthIdx.push_back(make_pair(z, i));
            }
        }
        // 按照深度从小到大排序
        sort(vDepthIdx.begin(), vDepthIdx.end());

        // We insert all close points (depth<mThDepth)
        // If less than 100 close points, we insert the 100 closest ones.
        // Step 2.2：从中找出不是地图点的部分
        int nPoints = 0;
        for (size_t j = 0; j < vDepthIdx.size(); j++) {
            int i = vDepthIdx[j].second;

            bool bCreateNew = false;

            // 如果这个点对应在上一帧中的地图点没有,或者创建后就没有被观测到,那么就生成一个临时的地图点
            MapPoint *pMP = mLastFrame.mvpMapPoints[i];
            if (!pMP)
                bCreateNew = true;
            else if (pMP->Observations() < 1) {
                // 地图点被创建后就没有被观测，认为不靠谱，也需要重新创建
                bCreateNew = true;
            }

            if (bCreateNew) {
                // Step 2.3：需要创建的点，包装为地图点。只是为了提高双目和RGBD的跟踪成功率，并没有添加复杂属性，因为后面会扔掉
                // 反投影到世界坐标系中
                cv::Mat x3D = mLastFrame.UnprojectStereo(i);
                MapPoint *pNewMP = new MapPoint(x3D, mpMap, &mLastFrame, i);
                // 加入上一帧的地图点中
                mLastFrame.mvpMapPoints[i] = pNewMP;

                // 标记为临时添加的MapPoint，之后在CreateNewKeyFrame之前会全部删除
                mlpTemporalPoints.push_back(pNewMP);
                nPoints++;
            } else {
                nPoints++;
            }
            // Step 2.4：如果地图点质量不好，停止创建地图点
            // 停止新增临时地图点必须同时满足以下条件：
            // 1、当前的点的深度已经超过了设定的深度阈值（35倍基线）
            // 2、nPoints已经超过100个点，说明距离比较远了，可能不准确，停掉退出
            if (vDepthIdx[j].first > mThDepth && nPoints > 100)
                break;
        }

        // 线的原理同点一样
        // Create "visual odometry" MapLines
        // We sort points according to their measured depth by the stereo/RGB-D sensor
        vector<pair<float, int> > vLineDepthIdx;
        vLineDepthIdx.reserve(mLastFrame.NL);
        int nLines = 0;
        for (int i = 0; i < mLastFrame.NL; i++) {
            float z = mLastFrame.mvDepthLine[i];
            if (z == 1) {
                bool bCreateNew = false;
                vLineDepthIdx.push_back(make_pair(z, i));
                MapLine *pML = mLastFrame.mvpMapLines[i];
                if (!pML)
                    bCreateNew = true;
                else if (pML->Observations() < 1) {
                    bCreateNew = true;
                }
                if (bCreateNew) {
                    Vector6d line3D = mLastFrame.obtain3DLine(i);//mvLines3D[i];
                    MapLine *pNewML = new MapLine(line3D, mpMap, &mLastFrame, i);
                    //Vector6d x3D = mLastFrame.mvLines3D(i);
                    //MapPoint *pNewMP = new MapPoint(x3D, mpMap, &mLastFrame, i);
                    mLastFrame.mvpMapLines[i] = pNewML;

                    mlpTemporalLines.push_back(pNewML);
                    nLines++;
                } else {
                    nLines++;
                }

                if (nLines > 30)
                    break;

            }
        }


    }

    bool Tracking::TrackLocalMap(bool bStruct)  {
        PlaneMatcher pmatcher(mfDThRef, mfAThRef, mfVerTh, mfParTh);

        UpdateLocalMap();
        // Step 2：筛选局部地图中新增的在视野范围内的地图点，投影到当前帧搜索匹配，得到更多的匹配关系
        thread threadPoints(&Tracking::SearchLocalPoints, this);
//        thread threadLines(&Tracking::SearchLocalLines, this);
        thread threadPlanes(&Tracking::SearchLocalPlanes, this);
        threadPoints.join();
//        threadLines.join();
        threadPlanes.join();

        // 在这里传入R_MF
        pmatcher.SearchMapByCoefficients(mCurrentFrame, mpMap->GetAllMapPlanes(),mRotation_wc);

        //cout << "tracking localmap with lines, pose before opti" << endl << mCurrentFrame.mTcw << endl;
        Optimizer::PoseOptimization(&mCurrentFrame,bStruct);
        //cout << "tracking localmap with lines, pose after opti" << mCurrentFrame.mTcw << endl;

        // 返回当前依据track的可信度    可信度不够，用MF重新优化
        //根据平面之间约束再次优化
//        Optimizer::PoseOptimization_strict(&mCurrentFrame); // 观测太少 优化变量多  欠定



        mnMatchesInliers = 0;

        // Update MapPoints Statistics
        for (int i = 0; i < mCurrentFrame.N; i++) {
            if (mCurrentFrame.mvpMapPoints[i]) {
                if (!mCurrentFrame.mvbOutlier[i]) {
                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                    if (!mbOnlyTracking) {
                        if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
                            mnMatchesInliers++;
                    } else
                        mnMatchesInliers++;
                } else if (mSensor == System::STEREO)
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);

            }
        }

//        for (int i = 0; i < mCurrentFrame.NL; i++) {
//            if (mCurrentFrame.mvpMapLines[i]) {
//                if (!mCurrentFrame.mvbLineOutlier[i]) {
//                    mCurrentFrame.mvpMapLines[i]->IncreaseFound();
//                    if (!mbOnlyTracking) {
//                        if (mCurrentFrame.mvpMapLines[i]->Observations() > 0)
//                            mnMatchesInliers++;
//                    } else
//                        mnMatchesInliers++;
//                } else if (mSensor == System::STEREO)
//                    mCurrentFrame.mvpMapLines[i] = static_cast<MapLine *>(NULL);
//            }
//        }

        int nDiscardPlane = 0;
        for (int i = 0; i < mCurrentFrame.mnPlaneNum; i++) {
            if (mCurrentFrame.mvpMapPlanes[i]) {
                if (mCurrentFrame.mvbPlaneOutlier[i]) {
//                    mCurrentFrame.mvpMapPlanes[i] = static_cast<MapPlane *>(nullptr);
//                    mCurrentFrame.mvbPlaneOutlier[i]=false;
//                    nDiscardPlane++;
                } else {
                    mCurrentFrame.mvpMapPlanes[i]->IncreaseFound();
                    mnMatchesInliers++;
                }
            }

            if (mCurrentFrame.mvpParallelPlanes[i]) {
                if (mCurrentFrame.mvbParPlaneOutlier[i]) {
                    mCurrentFrame.mvpParallelPlanes[i] = static_cast<MapPlane *>(nullptr);
                    mCurrentFrame.mvbParPlaneOutlier[i] = false;
                }
            }

            if (mCurrentFrame.mvpVerticalPlanes[i]) {
                if (mCurrentFrame.mvbVerPlaneOutlier[i]) {
                    mCurrentFrame.mvpVerticalPlanes[i] = static_cast<MapPlane *>(nullptr);
                    mCurrentFrame.mvbVerPlaneOutlier[i] = false;
                }
            }
        }

        // Decide if the tracking was succesful
        // More restrictive if there was a relocalization recently
        if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && mnMatchesInliers < 7) {
            cout << "TrackLocalMapWithLines: After: Not enough matches" << endl;
            return false;
        }


        if (mnMatchesInliers < 7) {
            cout << "TrackLocalMapWithLines: After: Not enough matches" << endl;
            return false;
        } else
            return true;
    }


/**
 * @brief 判断当前帧是否需要插入关键帧
 *
 * Step 1：纯VO模式下不插入关键帧，如果局部地图被闭环检测使用，则不插入关键帧
 * Step 2：如果距离上一次重定位比较近，或者关键帧数目超出最大限制，不插入关键帧
 * Step 3：得到参考关键帧跟踪到的地图点数量
 * Step 4：查询局部地图管理器是否繁忙,也就是当前能否接受新的关键帧
 * Step 5：对于双目或RGBD摄像头，统计可以添加的有效地图点总数 和 跟踪到的地图点数量
 * Step 6：决策是否需要插入关键帧
 * @return true         需要
 * @return false        不需要
 */
    bool Tracking::NeedNewKeyFrame() {
        // 纯VO模式下不插入关键帧
        if (mbOnlyTracking)
            return false;

        // If Local Mapping is freezed by a Loop Closure do not insert keyframes
        // Step 2：如果局部地图线程被闭环检测使用，则不插入关键帧
        if (mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
            return false;
        // 获取当前地图中的关键帧数目
        const int nKFs = mpMap->KeyFramesInMap();

        // Do not insert keyframes if not enough frames have passed from last relocalisation
        // mCurrentFrame.mnId是当前帧的ID
        // mnLastRelocFrameId是最近一次重定位帧的ID
        // mMaxFrames等于图像输入的帧率
        //  Step 3：如果距离上一次重定位比较近，并且关键帧数目超出最大限制，不插入关键帧
        if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && nKFs > mMaxFrames)
            return false;

        // Tracked MapPoints in the reference keyframe
        // 地图点的最小观测次数
        int nMinObs = 3;
        if (nKFs <= 2)
            nMinObs = 2;
        // 参考关键帧地图点中观测的数目>= nMinObs的地图点数目
        int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

        // Local Mapping accept keyframes?
        // Step 5：查询局部地图线程是否繁忙，当前能否接受新的关键帧
        bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

// Stereo & RGB-D: Ratio of close "matches to map"/"total matches"
// "total matches = matches to map + visual odometry matches"
// Visual odometry matches will become MapPoints if we insert a keyframe.
// This ratio measures how many MapPoints we could create if we insert a keyframe.
// Step 6：对于双目或RGBD摄像头，统计成功跟踪的近点的数量，如果跟踪到的近点太少，没有跟踪到的近点较多，可以插入关键帧
        int nMap = 0; //nTrackedClose //双目或RGB-D中成功跟踪的近点（三维点）
        int nTotal = 0;
        int nNonTrackedClose = 0;//双目或RGB-D中没有跟踪到的近点
        if (mSensor != System::MONOCULAR) {
            for (int i = 0; i < mCurrentFrame.N; i++) {
                if (mCurrentFrame.mvDepth[i] > 0 && mCurrentFrame.mvDepth[i] < mThDepth) {
                    nTotal++;
                    if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                        nMap++;
                    else
                        nNonTrackedClose++;
                }
            }
        } else {
            // There are no visual odometry matches in the monocular case
            nMap = 1;
            nTotal = 1;
        }

        bool bNeedToInsertClose = (nMap<100) && (nNonTrackedClose>70);

        const float ratioMap = (float) nMap / fmax(1.0f, nTotal);
        // Step 7：决策是否需要插入关键帧
        // Thresholds
        // Step 7.1：设定比例阈值，当前帧和参考关键帧跟踪到点的比例，比例越大，越倾向于增加关键帧
        float thRefRatio = 0.75f;
        if (nKFs < 2)
            thRefRatio = 0.4f;

        if (mSensor == System::MONOCULAR)
            thRefRatio = 0.9f;

        float thMapRatio = 0.35f;
        if (mnMatchesInliers > 300)
            thMapRatio = 0.20f;

// Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
        // Step 7.2：很长时间没有插入关键帧，可以插入
        const bool c1a = mCurrentFrame.mnId >= mnLastKeyFrameId + mMaxFrames;
// Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
        const bool c1b = (mCurrentFrame.mnId >= mnLastKeyFrameId + mMinFrames && bLocalMappingIdle);
//Condition 1c: tracking is weak
        // Step 7.4：在双目，RGB-D的情况下当前帧跟踪到的点比参考关键帧的0.25倍还少，或者满足bNeedToInsertClose
        const bool c1c = mSensor != System::MONOCULAR && (mnMatchesInliers < nRefMatches * 0.25 || bNeedToInsertClose);
// Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
        // Step 7.5：和参考帧相比当前跟踪到的点太少 或者满足bNeedToInsertClose；同时跟踪到的内点还不能太少
        const bool c2 = ((mnMatchesInliers < nRefMatches * thRefRatio || bNeedToInsertClose) &&
                         mnMatchesInliers > 15);



//        bool bNeedToInsertClose = (nMap<100) && (nNonTrackedClose>70);
//        // Condition 1c: tracking is weak
//        // Step 7.4：在双目，RGB-D的情况下当前帧跟踪到的点比参考关键帧的0.25倍还少，或者满足bNeedToInsertClose
//        const bool c1c =  mSensor!=System::MONOCULAR &&             //只考虑在双目，RGB-D的情况
//                          (mnMatchesInliers<nRefMatches*0.25 ||       //当前帧和地图点匹配的数目非常少
//                           bNeedToInsertClose) ;                     //需要插入
//
//        // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
//        // Step 7.5：和参考帧相比当前跟踪到的点太少 或者满足bNeedToInsertClose；同时跟踪到的内点还不能太少
//        const bool c2 = ((mnMatchesInliers<nRefMatches*thRefRatio|| bNeedToInsertClose) && mnMatchesInliers>15);

//           std::cout << c1a << " "<< c1b << " "<< c1c <<" "<< mCurrentFrame.mbNewPlane<<std::endl;
//        if (((c1a || c1b || c1c) && c2) || mCurrentFrame.mbNewPlane) {
          if (((c1a || c1b || c1c) && c2)) {
            // If the mapping accepts keyframes, insert keyframe.
            // Otherwise send a signal to interrupt BA
            // Step 7.6：local mapping空闲时可以直接插入，不空闲的时候要根据情况插入
            if (bLocalMappingIdle) {
                return true;
            } else {
                // 队列里不能阻塞太多关键帧
                // tracking插入关键帧不是直接插入，而且先插入到mlNewKeyFrames中，
                // 然后localmapper再逐个pop出来插入到mspKeyFrames
                mpLocalMapper->InterruptBA();
                if (mSensor != System::MONOCULAR) {
                    if (mpLocalMapper->KeyframesInQueue() < 3)
                        return true;
                    else
                        return false;
                } else
                    return false;
            }
        }

        return false;
    }

    /**
 * @brief 创建新的关键帧
 * 对于非单目的情况，同时创建新的MapPoints
 *
 * Step 1：将当前帧构造成关键帧
 * Step 2：将当前关键帧设置为当前帧的参考关键帧
 * Step 3：对于双目或rgbd摄像头，为当前帧生成新的MapPoints
 */
    void Tracking::CreateNewKeyFrame() {
        // 如果局部建图线程关闭了,就无法插入关键帧
        if (!mpLocalMapper->SetNotStop(true))
            return;

        // Step 1：将当前帧构造成关键帧
        KeyFrame *pKF = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB,this->mImRGB, this->mCurrentFrame.depth);
//        KeyFrame *pKF = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);
        // Step 2：将当前关键帧设置为当前帧的参考关键帧
        // 在UpdateLocalKeyFrames函数中会将与当前关键帧共视程度最高的关键帧设定为当前帧的参考关键帧
        mpReferenceKF = pKF;
        mCurrentFrame.mpReferenceKF = pKF;

        // 这段代码和 Tracking::UpdateLastFrame 中的那一部分代码功能相同
        // Step 3：对于双目或rgbd摄像头，为当前帧生成新的地图点；单目无操作
        if (mSensor != System::MONOCULAR) {

            mCurrentFrame.UpdatePoseMatrices();
            // We sort points by the measured depth by the stereo/RGBD sensor.
            // We create all those MapPoints whose depth < mThDepth.
            // If there are less than 100 close points we create the 100 closest.\
            // Step 3.1：得到当前帧有深度值的特征点（不一定是地图点）
            vector<pair<float, int> > vDepthIdx;
            vDepthIdx.reserve(mCurrentFrame.N);

            for (int i = 0; i < mCurrentFrame.N; i++) {
                float z = mCurrentFrame.mvDepth[i];
                if (z > 0) {
                    // 第一个元素是深度,第二个元素是对应的特征点的id
                    vDepthIdx.push_back(make_pair(z, i));
                }
            }

            if (!vDepthIdx.empty()) {
                // Step 3.2：按照深度从小到大排序
                sort(vDepthIdx.begin(), vDepthIdx.end());

                // Step 3.3：从中找出不是地图点的生成临时地图点
                // 处理的近点的个数
                int nPoints = 0;
                for (size_t j = 0; j < vDepthIdx.size(); j++) {
                    int i = vDepthIdx[j].second;

                    bool bCreateNew = false;

                    // 如果这个点对应在上一帧中的地图点没有,或者创建后就没有被观测到,那么就生成一个临时的地图点
                    MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                    if (!pMP)
                        bCreateNew = true;
                    else if (pMP->Observations() < 1) {
                        bCreateNew = true;
                        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                    }

                    // 如果需要就新建地图点，这里的地图点不是临时的，是全局地图中新建地图点，用于跟踪
                    if (bCreateNew) {
                        cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                        MapPoint *pNewMP = new MapPoint(x3D, pKF, mpMap);
                        // 这些添加属性的操作是每次创建MapPoint后都要做的
                        pNewMP->AddObservation(pKF, i);
                        pKF->AddMapPoint(pNewMP, i);
                        pNewMP->ComputeDistinctiveDescriptors();
                        pNewMP->UpdateNormalAndDepth();
                        mpMap->AddMapPoint(pNewMP);

                        mCurrentFrame.mvpMapPoints[i] = pNewMP;
                        nPoints++;
                    } else {
                        nPoints++;
                    }

                    if (vDepthIdx[j].first > mThDepth && nPoints > 100)
                        break;
                }
            }

            vector<pair<float, int>> vLineDepthIdx;
            vLineDepthIdx.reserve(mCurrentFrame.NL);

            for (int i = 0; i < mCurrentFrame.NL; i++) {
                float z = mCurrentFrame.mvDepthLine[i];
                if (z > 0) {
                    vLineDepthIdx.push_back(make_pair(z, i));
                }
            }

            if (!vLineDepthIdx.empty()) {
                sort(vLineDepthIdx.begin(),vLineDepthIdx.end());

                int nLines = 0;
                for (size_t j = 0; j < vLineDepthIdx.size(); j++) {
                    int i = vLineDepthIdx[j].second;

                    bool bCreateNew = false;

                    MapLine *pMP = mCurrentFrame.mvpMapLines[i];
                    if (!pMP)
                        bCreateNew = true;
                    else if (pMP->Observations() < 1) {
                        bCreateNew = true;
                        mCurrentFrame.mvpMapLines[i] = static_cast<MapLine *>(NULL);
                    }

                    if (bCreateNew) {
                        Vector6d line3D = mCurrentFrame.obtain3DLine(i);//mvLines3D[i];
                        MapLine *pNewML = new MapLine(line3D, pKF, mpMap);
                        pNewML->AddObservation(pKF, i);
                        pKF->AddMapLine(pNewML, i);
                        pNewML->ComputeDistinctiveDescriptors();
                        pNewML->UpdateAverageDir();
                        mpMap->AddMapLine(pNewML);
                        mCurrentFrame.mvpMapLines[i] = pNewML;
                        nLines++;
                    } else {
                        nLines++;
                    }

//                    if (nLines > 20)
                    if (nLines > 50)
                        break;
                }
            }

            for (int i = 0; i < mCurrentFrame.mnPlaneNum; ++i) {
                bool Regularity = false;
                if (mCurrentFrame.mvpParallelPlanes[i] && !mCurrentFrame.mvbPlaneOutlier[i]) {
                    mCurrentFrame.mvpParallelPlanes[i]->AddParObservation(pKF, i);
                    Regularity = true;
                }
                if (mCurrentFrame.mvpVerticalPlanes[i] && !mCurrentFrame.mvbPlaneOutlier[i]) {
                    Regularity = true;
                    mCurrentFrame.mvpVerticalPlanes[i]->AddVerObservation(pKF, i);
                }

                if (mCurrentFrame.mvpMapPlanes[i]) {
                    mCurrentFrame.mvpMapPlanes[i]->AddObservation(pKF, i);
                    continue;
                }

                if (mCurrentFrame.mvbPlaneOutlier[i]|| Regularity == false) {
                    mCurrentFrame.mvbPlaneOutlier[i] = false;
                    continue;
                }

                //添加新的平面   他虽然是新的平面，但是不符合平行和垂直的约束同样不能插入进去
                if(Regularity == true) {
//                     满足平行或垂直的约束 添加新的平面到地图中
                    cout << "满足平行或垂直的约束 添加新的平面到地图中"<< endl;
                    cv::Mat p3D = mCurrentFrame.ComputePlaneWorldCoeff(i);
                    cv::Mat A = p3D(cvRect(0,0,1,3));
//
//                    // 根据约束，更改他在平面中的朝向 用匹配到的平面对他进行替换
//                    if(mCurrentFrame.mvpParallelPlanes[i])
//                        mCurrentFrame.mvpParallelPlanes[i]->GetWorldPos().rowRange(1,3).copyTo(A);
//                    else {
//                        int number_pl = mpMap->MapPlanesInMap();
//                        const std::vector<MapPlane *> vpMapPlanes = mpMap->GetAllMapPlanes();
//                        cv::Mat Plane_A = mCurrentFrame.mvpVerticalPlanes[i]->GetWorldPos().rowRange(1, 3);
//                        Eigen::Vector3d Normal_A;
//                        for (int j = 0; j < number_pl; j++) {
//                            if (vpMapPlanes[j]->isBad()) continue;
//                            cv::Mat Plane_B = vpMapPlanes[j]->GetWorldPos().rowRange(1, 3);
//                            Eigen::Vector3d Normal_B;
//                            Eigen::Vector3d Normal_C;
//                            cv::cv2eigen(Plane_A, Normal_A);
//                            cv::cv2eigen(Plane_B, Normal_B);
//                            Normal_C = Normal_A.cross(Normal_B);
//                            Normal_C.normalize();
//                            cv::Mat Plane_C;
//                            cv::eigen2cv(Normal_C, Plane_C);
//
//                            float angle = A.at<double>(0, 0) * Plane_C.at<double>(0, 0) +
//                                          A.at<double>(1, 0) * Plane_C.at<double>(1, 0) +
//                                          A.at<double>(2, 0) * Plane_C.at<double>(2, 0);
//                            if (angle > 0.98)
//                                Plane_C.copyTo(A);
//                            else if (angle < -0.98) {
//                                cv::Mat Plane_D = -Plane_C;
//                                Plane_D.copyTo(A);
//                            }
//                        }
//                    }

                    MapPlane *pNewMP = new MapPlane(p3D, pKF, mpMap);
                    pNewMP->AddObservation(pKF, i);
                    // 添加新的平面
                    pKF->AddMapPlane(pNewMP, i);
                    pNewMP->UpdateCoefficientsAndPoints();
                    mpMap->AddMapPlane(pNewMP);
                    cout << "*********************mpMap->MapPlanesInMap() " << mpMap->MapPlanesInMap() << endl;
                }
            }

//            mpPointCloudMapping->print();

//            cout << "New map created with " << mpMap->MapPlanesInMap() << " planes" << endl;
        }

        mpLocalMapper->InsertKeyFrame(pKF);

        mpLocalMapper->SetNotStop(false);

        mnLastKeyFrameId = mCurrentFrame.mnId;
        mpLastKeyFrame = pKF;
    }

    /*利用反投影原理更新面的参数
     * */
    void Tracking::UpdatePlane(cv::Mat R_cm) {
        int N = mCurrentFrame.mnPlaneNum;

        cv::Mat P_normal = cv::Mat::eye(3,3,CV_32F);

        for(int i=0; i<N; i++) {
            cv::Mat pM = mCurrentFrame.mvPlaneCoefficients[i];
//            cout << pM << endl;
            for(int j=0;j<3;j++) {
                float angle = pM.at<float>(0,0) *R_cm.at<float>(0,j)+
                              pM.at<float>(1,0) *R_cm.at<float>(1,j)+
                              pM.at<float>(2,0) *R_cm.at<float>(2,j);
//                cout << "acos(angle) " << angle<< endl;
                if (fabs(angle) > 0.96 )
                {
                    cv::Mat m = R_cm.col(j).clone();
                    if(angle < 0)
                        m = -m;
                    cv::Mat B = mCurrentFrame.mvPlaneCoefficients[i](cvRect(0,0,1,3));
                    m.copyTo(B);
//                    cout << "b" << B.t() << endl;
//                    cout << "wangwen" << mCurrentFrame.mvPlaneCoefficients[i]<<endl;
                }
            }
        }

    }

    void Tracking::SearchLocalPoints() {
// Do not search map points already matched
        for (vector<MapPoint *>::iterator vit = mCurrentFrame.mvpMapPoints.begin(), vend = mCurrentFrame.mvpMapPoints.end();
             vit != vend; vit++) {
            MapPoint *pMP = *vit;
            if (pMP) {
                if (pMP->isBad()) {
                    *vit = static_cast<MapPoint *>(NULL);
                } else {
                    pMP->IncreaseVisible();
                    pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                    pMP->mbTrackInView = false;
                }
            }
        }

        int nToMatch = 0;

// Project points in frame and check its visibility
        for (vector<MapPoint *>::iterator vit = mvpLocalMapPoints.begin(), vend = mvpLocalMapPoints.end();
             vit != vend; vit++) {
            MapPoint *pMP = *vit;
            if (pMP->mnLastFrameSeen == mCurrentFrame.mnId)
                continue;
            if (pMP->isBad())
                continue;
            // Project (this fills MapPoint variables for matching)
            if (mCurrentFrame.isInFrustum(pMP, 0.5)) {
                pMP->IncreaseVisible();
                nToMatch++; //将要match的
            }
        }

        if (nToMatch > 0) {
            ORBmatcher matcher(0.8);
            int th = 1;
            if (mSensor == System::RGBD)
                th = 3;
            // If the camera has been relocalised recently, perform a coarser search
            if (mCurrentFrame.mnId < mnLastRelocFrameId + 2)
                th = 5;
            matcher.SearchByProjection(mCurrentFrame, mvpLocalMapPoints, th);
        }
    }

    void Tracking::SearchLocalLines() {
        // step1：
        for (vector<MapLine *>::iterator vit = mCurrentFrame.mvpMapLines.begin(), vend = mCurrentFrame.mvpMapLines.end();
             vit != vend; vit++) {
            MapLine *pML = *vit;
            if (pML) {
                if (pML->isBad()) {
                    *vit = static_cast<MapLine *>(NULL);
                } else {
                    //
                    pML->IncreaseVisible();
                    pML->mnLastFrameSeen = mCurrentFrame.mnId;
                    pML->mbTrackInView = false;
                }
            }
        }

        int nToMatch = 0;

        // step2：
        for (vector<MapLine *>::iterator vit = mvpLocalMapLines.begin(), vend = mvpLocalMapLines.end();
             vit != vend; vit++) {
            MapLine *pML = *vit;

            if (pML->mnLastFrameSeen == mCurrentFrame.mnId)
                continue;
            if (pML->isBad())
                continue;

            // step2.1
            if (mCurrentFrame.isInFrustum(pML, 0.6)) {
                pML->IncreaseVisible();
                nToMatch++;
            }
        }

        if (nToMatch > 0) {
            LSDmatcher matcher;
            int th = 1;

            if (mCurrentFrame.mnId < mnLastRelocFrameId + 2)
                th = 5;

            matcher.SearchByProjection(mCurrentFrame, mvpLocalMapLines, th);
        }
    }

    void Tracking::SearchLocalPlanes() {
        for (vector<MapPlane *>::iterator vit = mCurrentFrame.mvpMapPlanes.begin(), vend = mCurrentFrame.mvpMapPlanes.end();
             vit != vend; vit++) {
            MapPlane *pMP = *vit;
            if (pMP) {
                if (pMP->isBad()) {
                    *vit = static_cast<MapPlane *>(NULL);
                } else {
                    pMP->IncreaseVisible();
                }
            }
        }
    }


    void Tracking::UpdateLocalMap() {
// This is for visualization
        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);
        mpMap->SetReferenceMapLines(mvpLocalMapLines);

// Update
        UpdateLocalKeyFrames();
        //cout << "the size of local keyframe" << mvpLocalKeyFrames.size() << endl;

        UpdateLocalPoints();
        UpdateLocalLines();
    }

    void Tracking::UpdateLocalLines() {
        //cout << "Tracking: UpdateLocalLines()" << endl;
        // step1：
        mvpLocalMapLines.clear();

        // step2：
        for (vector<KeyFrame *>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end();
             itKF != itEndKF; itKF++) {
            KeyFrame *pKF = *itKF;
            const vector<MapLine *> vpMLs = pKF->GetMapLineMatches();

            //step3：将局部关键帧的MapLines添加到mvpLocalMapLines
            for (vector<MapLine *>::const_iterator itML = vpMLs.begin(), itEndML = vpMLs.end();
                 itML != itEndML; itML++) {
                MapLine *pML = *itML;
                if (!pML)
                    continue;
                if (pML->mnTrackReferenceForFrame == mCurrentFrame.mnId)
                    continue;
                if (!pML->isBad()) {
                    mvpLocalMapLines.push_back(pML);
                    pML->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                }
            }
        }
    }

    void Tracking::UpdateLocalPoints() {
        mvpLocalMapPoints.clear();

        for (vector<KeyFrame *>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end();
             itKF != itEndKF; itKF++) {
            KeyFrame *pKF = *itKF;
            const vector<MapPoint *> vpMPs = pKF->GetMapPointMatches();

            for (vector<MapPoint *>::const_iterator itMP = vpMPs.begin(), itEndMP = vpMPs.end();
                 itMP != itEndMP; itMP++) {
                MapPoint *pMP = *itMP;
                if (!pMP)
                    continue;
                if (pMP->mnTrackReferenceForFrame == mCurrentFrame.mnId)
                    continue;
                if (!pMP->isBad()) {
                    mvpLocalMapPoints.push_back(pMP);
                    pMP->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                }
            }
        }
    }


    void Tracking::UpdateLocalKeyFrames() {
// Each map point vote for the keyframes in which it has been observed
        map<KeyFrame *, int> keyframeCounter;
        for (int i = 0; i < mCurrentFrame.N; i++) {
            if (mCurrentFrame.mvpMapPoints[i]) {
                MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                if (!pMP->isBad()) {
                    const map<KeyFrame *, size_t> observations = pMP->GetObservations();
                    for (map<KeyFrame *, size_t>::const_iterator it = observations.begin(), itend = observations.end();
                         it != itend; it++)
                        keyframeCounter[it->first]++;
                } else {
                    mCurrentFrame.mvpMapPoints[i] = NULL;
                }
            }
        }

        if (keyframeCounter.empty())
            return;

        int max = 0;
        KeyFrame *pKFmax = static_cast<KeyFrame *>(NULL);

        mvpLocalKeyFrames.clear();
        mvpLocalKeyFrames.reserve(3 * keyframeCounter.size());

// All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
        for (map<KeyFrame *, int>::const_iterator it = keyframeCounter.begin(), itEnd = keyframeCounter.end();
             it != itEnd; it++) {
            KeyFrame *pKF = it->first;

            if (pKF->isBad())
                continue;

            if (it->second > max) {
                max = it->second;
                pKFmax = pKF;
            }

            mvpLocalKeyFrames.push_back(it->first);
            pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
        }


// Include also some not-already-included keyframes that are neighbors to already-included keyframes
        for (vector<KeyFrame *>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end();
             itKF != itEndKF; itKF++) {
            // Limit the number of keyframes
            if (mvpLocalKeyFrames.size() > 80)
                break;

            KeyFrame *pKF = *itKF;

            const vector<KeyFrame *> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

            for (vector<KeyFrame *>::const_iterator itNeighKF = vNeighs.begin(), itEndNeighKF = vNeighs.end();
                 itNeighKF != itEndNeighKF; itNeighKF++) {
                KeyFrame *pNeighKF = *itNeighKF;
                if (!pNeighKF->isBad()) {
                    if (pNeighKF->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
                        mvpLocalKeyFrames.push_back(pNeighKF);
                        pNeighKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                        break;
                    }
                }
            }

            const set<KeyFrame *> spChilds = pKF->GetChilds();
            for (set<KeyFrame *>::const_iterator sit = spChilds.begin(), send = spChilds.end(); sit != send; sit++) {
                KeyFrame *pChildKF = *sit;
                if (!pChildKF->isBad()) {
                    if (pChildKF->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
                        mvpLocalKeyFrames.push_back(pChildKF);
                        pChildKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                        break;
                    }
                }
            }

            KeyFrame *pParent = pKF->GetParent();
            if (pParent) {
                if (pParent->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
                    mvpLocalKeyFrames.push_back(pParent);
                    pParent->mnTrackReferenceForFrame = mCurrentFrame.mnId;
                    break;
                }
            }

        }

        if (pKFmax) {
            mpReferenceKF = pKFmax;
            mCurrentFrame.mpReferenceKF = mpReferenceKF;
        }
    }

    bool Tracking::Relocalization(bool bStruct) {
        cout << "Tracking:localization" << endl;
        // Compute Bag of Words Vector
        mCurrentFrame.ComputeBoW();

        // Relocalization is performed when tracking is lost
        // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
        vector<KeyFrame *> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);

        cout << "Tracking,vpCandidateKFs" << vpCandidateKFs.size() << endl;
        if (vpCandidateKFs.empty())
            return false;

        const int nKFs = vpCandidateKFs.size();

        // We perform first an ORB matching with each candidate
        // If enough matches are found we setup a PnP solver
        ORBmatcher matcher(0.75, true);
        vector<PnPsolver *> vpPnPsolvers;
        vpPnPsolvers.resize(nKFs);
        vector<vector<MapPoint *> > vvpMapPointMatches;
        vvpMapPointMatches.resize(nKFs);
        vector<bool> vbDiscarded;
        vbDiscarded.resize(nKFs);

        int nCandidates = 0;
        for (int i = 0; i < nKFs; i++) {
            KeyFrame *pKF = vpCandidateKFs[i];
            if (pKF->isBad())
                vbDiscarded[i] = true;
            else {
                int nmatches = matcher.SearchByBoW(pKF, mCurrentFrame, vvpMapPointMatches[i]);
                if (nmatches < 15) {
                    vbDiscarded[i] = true;
                    continue;
                } else {
                    PnPsolver *pSolver = new PnPsolver(mCurrentFrame, vvpMapPointMatches[i]);
                    pSolver->SetRansacParameters(0.99, 10, 300, 4, 0.5, 5.991);
                    vpPnPsolvers[i] = pSolver;
                    nCandidates++;
                }
            }
        }

// Alternatively perform some iterations of P4P RANSAC
// Until we found a camera pose supported by enough inliers
        bool bMatch = false;
        ORBmatcher matcher2(0.9, true);

        while (nCandidates > 0 && !bMatch) {
            for (int i = 0; i < nKFs; i++) {
                if (vbDiscarded[i])
                    continue;

                // Perform 5 Ransac Iterations
                vector<bool> vbInliers;
                int nInliers;
                bool bNoMore;

                PnPsolver *pSolver = vpPnPsolvers[i];
                cv::Mat Tcw = pSolver->iterate(5, bNoMore, vbInliers, nInliers);

                // If Ransac reachs max. iterations discard keyframe
                if (bNoMore) {
                    vbDiscarded[i] = true;
                    nCandidates--;
                }

                // If a Camera Pose is computed, optimize
                if (!Tcw.empty()) {
                    Tcw.copyTo(mCurrentFrame.mTcw);

                    set<MapPoint *> sFound;

                    const int np = vbInliers.size();

                    for (int j = 0; j < np; j++) {
                        if (vbInliers[j]) {
                            mCurrentFrame.mvpMapPoints[j] = vvpMapPointMatches[i][j];
                            sFound.insert(vvpMapPointMatches[i][j]);
                        } else
                            mCurrentFrame.mvpMapPoints[j] = NULL;
                    }

                    int nGood = Optimizer::PoseOptimization(&mCurrentFrame,bStruct);

                    if (nGood < 10)
                        continue;

                    for (int io = 0; io < mCurrentFrame.N; io++)
                        if (mCurrentFrame.mvbOutlier[io])
                            mCurrentFrame.mvpMapPoints[io] = static_cast<MapPoint *>(NULL);

                    // If few inliers, search by projection in a coarse window and optimize again
                    if (nGood < 50) {
                        int nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 10,
                                                                      100);

                        if (nadditional + nGood >= 50) {
                            nGood = Optimizer::PoseOptimization(&mCurrentFrame,bStruct);

                            // If many inliers but still not enough, search by projection again in a narrower window
                            // the camera has been already optimized with many points
                            if (nGood > 30 && nGood < 50) {
                                sFound.clear();
                                for (int ip = 0; ip < mCurrentFrame.N; ip++)
                                    if (mCurrentFrame.mvpMapPoints[ip])
                                        sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
                                nadditional = matcher2.SearchByProjection(mCurrentFrame, vpCandidateKFs[i], sFound, 3,
                                                                          64);

                                // Final optimization
                                if (nGood + nadditional >= 50) {
                                    nGood = Optimizer::PoseOptimization(&mCurrentFrame,bStruct);

                                    for (int io = 0; io < mCurrentFrame.N; io++)
                                        if (mCurrentFrame.mvbOutlier[io])
                                            mCurrentFrame.mvpMapPoints[io] = NULL;
                                }
                            }
                        }
                    }


                    // If the pose is supported by enough inliers stop ransacs and continue
                    if (nGood >= 50) {
                        bMatch = true;
                        break;
                    }
                }
            }

            if (!bMatch) {

            }
        }

        if (!bMatch) {
            return false;
        } else {
            mnLastRelocFrameId = mCurrentFrame.mnId;
            return true;
        }

    }

    void Tracking::Reset() {
        mpViewer->RequestStop();

        cout << "System Reseting" << endl;
        while (!mpViewer->isStopped())
            usleep(3000);

// Reset Local Mapping
        cout << "Reseting Local Mapper...";
        mpLocalMapper->RequestReset();
        cout << " done" << endl;

// Reset Loop Closing
        cout << "Reseting Loop Closing...";
        mpLoopClosing->RequestReset();
        cout << " done" << endl;

// Clear BoW Database
        cout << "Reseting Database...";
        mpKeyFrameDB->clear();
        cout << " done" << endl;

// Clear Map (this erase MapPoints and KeyFrames)
        mpMap->clear();

        KeyFrame::nNextId = 0;
        Frame::nNextId = 0;
        mState = NO_IMAGES_YET;

        if (mpInitializer) {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer *>(NULL);
        }

        mlRelativeFramePoses.clear();
        mlpReferences.clear();
        mlFrameTimes.clear();
        mlbLost.clear();

        mpViewer->Release();
    }

    void Tracking::ChangeCalibration(const string &strSettingPath) {
        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
        float fx = fSettings["Camera.fx"];
        float fy = fSettings["Camera.fy"];
        float cx = fSettings["Camera.cx"];
        float cy = fSettings["Camera.cy"];

        cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
        K.at<float>(0, 0) = fx;
        K.at<float>(1, 1) = fy;
        K.at<float>(0, 2) = cx;
        K.at<float>(1, 2) = cy;
        K.copyTo(mK);

        cv::Mat DistCoef(4, 1, CV_32F);
        DistCoef.at<float>(0) = fSettings["Camera.k1"];
        DistCoef.at<float>(1) = fSettings["Camera.k2"];
        DistCoef.at<float>(2) = fSettings["Camera.p1"];
        DistCoef.at<float>(3) = fSettings["Camera.p2"];
        const float k3 = fSettings["Camera.k3"];
        if (k3 != 0) {
            DistCoef.resize(5);
            DistCoef.at<float>(4) = k3;
        }
        DistCoef.copyTo(mDistCoef);

        mbf = fSettings["Camera.bf"];

        Frame::mbInitialComputations = true;
    }

    void Tracking::InformOnlyTracking(const bool &flag) {
        mbOnlyTracking = flag;
    }

    void Tracking::SaveMesh(const string &filename){
//        mpPointCloudMapping->SaveMeshModel(filename);

    }

    //**refer: Bechmarking 6DOF Outdoor VIsual Localization in Changing Conditions**
    // 2 cos(|alpha|) = trace(R_inv * R) - 1
    double Tracking::MatrixResidual(cv::Mat Mahattan_R, cv::Mat Track_R) {
        cv::Mat Residual  = Mahattan_R.t() * Track_R;
        Eigen::Matrix3d R;

        cv::cv2eigen(Residual,R);
        double trace = R.trace();
//        std::cout << "trace " << trace << std::endl;
        double alpha = 180*( acos(0.5 *(trace - 1) ))/3.1415;

        return alpha;
    }
    void Tracking::SetSemanticer(YOLOX *detector)
    {
        Semanticer = detector;
    }


} //namespace Planar_SLAM
