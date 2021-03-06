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

// ????????????RGB???RGBA??????????????????
// 1??????????????????mImGray???imDepth????????????mCurrentFrame
// 2?????????tracking??????
// ????????????????????????????????????????????????????????????
    cv::Mat Tracking::GrabImageRGBD(const cv::Mat &imRGB, const cv::Mat &imD, const double &timestamp,Eigen::Quaterniond &GroundTruth_R) {
        mImRGB = imRGB; //????????????
        mImGray = imRGB;//????????????
        mImDepth = imD; //????????????

        // step 1??????RGB???RGBA????????????????????????
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
        // ??????Frame  RGB??? ????????? ????????? ????????? ORB??????????????? ?????? ?????????????????? ???????????????????????? ????????????*???????????? ??????????????????????????? ?????????????????? ??????YOLOX??????
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
 * track???????????????????????????????????????????????????
 *
 * Step 1????????????
 * Step 2?????????
 * Step 3??????????????????????????????????????????
 */
    void Tracking::Track(Eigen::Quaterniond &GroundTruth_R) {
        // mState???tracking?????????????????? SYSTME_NOT_READY, NO_IMAGE_YET, NOT_INITIALIZED, OK, LOST
        // ??????????????????????????????????????????????????????NO_IMAGE_YET??????
        if (mState == NO_IMAGES_YET) {
            mState = NOT_INITIALIZED;
        }
        // mLastProcessedState ?????????Tracking????????????????????????FrameDrawer????????????
        mLastProcessedState = mState;

        // Get Map Mutex -> Map cannot be changed
        // ??????????????????????????????????????????????????????
        // ??????:??????????????????????????????????????????????
        // ?????????????????????????????????????????????????????????????????????,??????????????????????????????????????????,??????????????????????????????
        unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

        mvpSurfaceNormals_M.mvpsurfacenormal = mCurrentFrame.vSurfaceNormal;
        mvpSurfaceNormals_M.bsurfacenormal_inline =vector<bool>(mCurrentFrame.vSurfaceNormal.size(), false);

        if (mState == NOT_INITIALIZED &&  mpMap->GetMaxKFid() == 0) {
            if(mCurrentFrame.mnPlaneNum < 2) return;
            if (mSensor == System::STEREO || mSensor == System::RGBD) {
                cout << "mCurrentFrame.mnPlaneNum " <<mCurrentFrame.mnPlaneNum << endl;
                Rotation_cm = cv::Mat::zeros(cv::Size(3, 3), CV_32F);
                // ????????????????????????????????????????????????????????????
                Rotation_cm = mpMap->FindManhattan(mCurrentFrame, mfVerTh, true);
                //Rotation_cm=SeekManhattanFrame(mCurrentFrame.vSurfaceNormal,mCurrentFrame.mVF3DLines).clone();
                // ????????????????????????????????????????????????
                Rotation_cm = TrackManhattanFrame(Rotation_cm, mCurrentFrame.vSurfaceNormal, mCurrentFrame.mVF3DLines).clone();
                Rotation_cm = TrackManhattanFrame(Rotation_cm, mCurrentFrame.vSurfaceNormal, mCurrentFrame.mVF3DLines).clone();
                Rotation_cm = TrackManhattanFrame(Rotation_cm, mCurrentFrame.vSurfaceNormal, mCurrentFrame.mVF3DLines).clone();

                //??????RGBD????????????????????????????????????
                StereoInitialization();
                Rotation_wm = cv::Mat::eye(3, 3, CV_32F);
                /*???????????????????????????????????????
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
            // bOK????????????????????????????????????????????????????????????
            bool bOK = false;
            bool bManhattan = false;
//            cout << "INITIALIZED"<< endl;
            floor_normal.copyTo(mCurrentFrame.Floor_Normal);
            // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
            // mbOnlyTracking??????false????????????SLAM???????????????+??????????????????mbOnlyTracking??????true?????????????????????
            // tracking ?????????????????????false
            if (!mbOnlyTracking) {
                mUpdateMF = true;
                cv::Mat MF_can = cv::Mat::zeros(cv::Size(3, 3), CV_32F);
                cv::Mat MF_can_T = cv::Mat::zeros(cv::Size(3, 3), CV_32F);

                MF_can = TrackManhattanFrame(mLastRcm, mCurrentFrame.vSurfaceNormal, mCurrentFrame.mVF3DLines).clone();// MF to camera
                MF_can.copyTo(mLastRcm);//.clone();
                MF_can = TrackManhattanFrame(mLastRcm, mCurrentFrame.vSurfaceNormal, mCurrentFrame.mVF3DLines).clone();// MF to camera
                MF_can.copyTo(mLastRcm);//.clone();
                MF_can = TrackManhattanFrame(mLastRcm, mCurrentFrame.vSurfaceNormal, mCurrentFrame.mVF3DLines).clone();// MF to camera

                /*???????????????????????????????????????
                 * */
//                UpdatePlane(MF_can);

                /*????????????????????????????????????Rcm
                 * ????????????????????????????????????????????????????????????????????????IMU??????????????????????????????????????????????????????
                 *
                 *
                 * MF_can = UpadateManhattanFram(MF_can,mCurrentFrame);
                 *
                 * */

                MF_can.copyTo(mLastRcm);//.clone();
                MF_can_T = MF_can.t();//MF_can_T: camera to MF
                // Rotation_cm -> Rotation_mw
                mRotation_wc = Rotation_wm * MF_can_T;//Rotation_cm ?????????MF to camera ?????????????????????????????????????????????????????????????????? ???????????????MF to World
                mRotation_wc = mRotation_wc.t(); // ?????????w???c
                // Step 2.1 ????????????????????????????????????MapPoints
                // ???????????????????????????????????????????????????????????????.?????????????????????
                CheckReplacedInLastFrame();

                if (mVelocity.empty() || mCurrentFrame.mnId < mnLastRelocFrameId + 2) {
//                    bOK = TranslationEstimation(false);
                    bOK = TrackReferenceKeyFrame(false);

                } else {
                    // ????????????????????????????????????????????????
                    // ????????????????????????????????????????????????
                    // ?????????????????????????????????????????????????????????????????????
                    // ??????????????????????????????3D????????????????????????????????????
//                    bOK = TranslationWithMotionModel(GroundTruth_R,false);
                    bOK = TrackWithMotionModel(false);
//                    if(!bOK){
//                        bOK = TranslationWithMotionModel(GroundTruth_R ,false);
//                    }

                    if (!bOK) {
                        //??????????????????????????????????????????????????????????????????
                        cout << "??????????????????????????????????????????????????????????????????" << endl;
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

            // ????????????????????????????????????????????????
            cv::Mat GT_rotation;
            GT_rotation = Converter::toCvMat(GroundTruth_R.toRotationMatrix());

//            double alpha = MatrixResidual(mRotation_wc,GT_rotation);
//            std::cout << "***********Manhataan : "<< alpha << std::endl;
//            double blpha = MatrixResidual(mCurrentFrame.mRcw,GT_rotation);
//            std::cout << "***********Estimation: "<< blpha << std::endl;

            mpMap->SetSurfaceNormal_M(mvpSurfaceNormals_M);
            mpMap->SetRmc(mLastRcm);
            mCurrentFrame.mpReferenceKF = mpReferenceKF;


            //Pm  Rmw * Pw   ???y=0  ???????????????
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
            //Pm  Rmw * Pw   ???y=0  ???????????????
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
                // ?????????P???????????????????????????   [ *, 0, *]
                /*  ????????????????????????????????????   [ 0, 1, 0] =R_dis*Rmc*I=Rmw*Rwc*I
                 *                         [ *, 0, *]
                 *  ???Rmc*I????????????
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
            // ???????????????????????????????????????????????????
//            cout << "mpFrameDrawer->Update"<< endl;
            mpFrameDrawer->Update(this);
//            cout << "finish"<< endl;
            mpMap->FlagMatchedPlanePoints(mCurrentFrame, mfDThRef);

            //Update Planes
            // ????????????
//            cout << "Update Planes" << endl;
            for (int i = 0; i < mCurrentFrame.mnPlaneNum; ++i) {
                MapPlane *pMP = mCurrentFrame.mvpMapPlanes[i];// ??????????????????
                if (pMP) {
                    pMP->UpdateCoefficientsAndPoints(mCurrentFrame, i);
                } else if (!mCurrentFrame.mvbPlaneOutlier[i])  {
                    //????????????????????????????????????????????????
                    //  todo ?????????????????????????????????????????????????????????anyone, ???????????????????????????????????????
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
            //?????????????????????????????????????????????????????????
            if (bOK) {
                // Update motion model
                // Step 5??????????????????????????????????????????
//                cout << "Update motion model"<< endl;
                if (!mLastFrame.mTcw.empty()) {
                    cv::Mat LastTwc = cv::Mat::eye(4, 4, CV_32F);
                    mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0, 3).colRange(0, 3));
                    mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0, 3).col(3));
                    // mVelocity = Tcl = Tcw * Twl,??????????????????????????????????????? ?????? Twl = LastTwc
                    mVelocity = mCurrentFrame.mTcw * LastTwc;
                } else
                    //??????????????????
                    mVelocity = cv::Mat();
                //????????????????????????
                mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw);
                // Clean VO matches
                // Step 6?????????????????????????????????
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
                    MapPlane *pMP = mCurrentFrame.mvpMapPlanes[i];//??????????????????????????????????????????????????????
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
                // Step 7?????????????????????????????? UpdateLastFrame??????????????????????????????MapPoints???????????????rgbd???
                // ??????6?????????????????????????????????MapPoints??????????????????MapPoints??????????????????
                // ?????????????????????????????????????????????rgbd?????????????????????????????????????????????????????????????????????????????????
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
                // Step 8?????????????????????????????????????????????RGB-D????????????????????????
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
            //???????????????Rotation
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
                //???????????????
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
            R_cm_Rec = R_cm_Rec / norm(R_cm_Rec); //?????????
            return R_cm_Rec;
        }

        return R_cm_NULL;

    }

    /*refer Paper: Divide and Conquer Efficient  ensity-based Tracking of 3D Sensors in Manhattan Worlds
     * ??????????????????????????????????????????????????????????????????
     * a : ????????????
     * R_mc: Camera to Mahattam
     * vTempSurfaceNormal: ??????????????????????????????????????????
     * vVanishingDirection??? ????????????
     * numOfSN??? ????????????????????????????????????
     * */
    // ???????????????????????????????????????
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
            // ?????????????????????????????????????????????pass
            if (lambda < sin(0.2518)) //0.25
            {
                double tan_alfa = lambda / std::abs(n_ini.z);
                double alfa = asin(lambda); //?????????

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
            // note: ?????????mean shit ????????????????????????????????????
            // reject outlier: ????????????
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
            R_cm_Rec = R_cm_Rec / norm(R_cm_Rec); //?????????
            RandDen.R_cm_Rec = R_cm_Rec;
            RandDen.s_j_density = s_j_density;

            return RandDen;
        }
        RandDen.R_cm_Rec = R_cm_NULL;
        return RandDen;
    }

    /*refer Paper: Divide and Conquer Efficient  ensity-based Tracking of 3D Sensors in Manhattan Worlds
     * ??????????????????????????????????????????????????????????????????
     * a : ????????????
     * R_mc: Camera to Mahattam
     * vTempSurfaceNormal: ??????????????????????????????????????????
     * vVanishingDirection??? ????????????
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
            // ???????????????
            if (i < vTempSurfaceNormal.size()) {
                // Rmc * [x,y,z] = ????????????Manhattan????????????????????? x y z
                n_ini.x = R_mc.at<float>(0, 0) * vTempSurfaceNormal[i].normal.x +
                          R_mc.at<float>(0, 1) * vTempSurfaceNormal[i].normal.y +
                          R_mc.at<float>(0, 2) * vTempSurfaceNormal[i].normal.z;
                n_ini.y = R_mc.at<float>(1, 0) * vTempSurfaceNormal[i].normal.x +
                          R_mc.at<float>(1, 1) * vTempSurfaceNormal[i].normal.y +
                          R_mc.at<float>(1, 2) * vTempSurfaceNormal[i].normal.z;
                n_ini.z = R_mc.at<float>(2, 0) * vTempSurfaceNormal[i].normal.x +
                          R_mc.at<float>(2, 1) * vTempSurfaceNormal[i].normal.y +
                          R_mc.at<float>(2, 2) * vTempSurfaceNormal[i].normal.z;

                // ??????x y ?????????????????? ?????????????????? ???????????????0 ??? ????????????????????????[0,0,1]
                // ???????????????????????????????????????0.25?????????x,y?????????????????????????????????????????????????????????????????????z??????
                double lambda = sqrt(n_ini.x * n_ini.x + n_ini.y * n_ini.y);//at(k).y*n_a.at(k).y);
                //cout<<lambda<<endl;
                // note: ???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
                if (lambda < sin(0.2018)) //0.25
                {
                    //vSNCadidate.push_back(vTempSurfaceNormal[i]);
                    //numInConic++;
                    // ?????????????????????a?????????????????????????????????
                    tempaxiSNV.SNVector.push_back(vTempSurfaceNormal[i]);
                    mvpSurfaceNormals_M.bsurfacenormal_inline[i] = true;
                }
            } else {
                // ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
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

    /* @brief ????????????????????????????????????????????????????????????
     * @brief ????????????
     * @param[in] origin_point ?????????????????????
     * wangwen
     * */
    vector<cv::Point2d> Tracking::Outlier_Filter(vector<cv::Point2d> &origin_point)
    {
        vector<cv::Point2d> final_points;

        // ?????????????????????????????????????????????????????????????????????

        // ????????????????????? txt  ????????????



        ofstream outfile;
        outfile.open("/home/wangwen/catkin_ws/test.txt");

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>); // ??????????????????
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>); // ??????????????????
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

//        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> Static;   //?????????????????????
//        Static.setInputCloud (cloud);                           //????????????????????????
//        Static.setMeanK(500);                               //???????????????????????????????????????????????????
//        Static.setStddevMulThresh (0.05);                      //???????????????????????????????????????
//        Static.filter (*cloud_filtered);                    //??????
//
//        for(size_t i=0 ;i < cloud_filtered->size();i++)
//        {
//            cv::Point2d p;
//            p.x = cloud_filtered->points[i].x;
//            p.y = cloud_filtered->points[i].y;
//            final_points.push_back(p);
//        }
//        cout << "final_points.size()"<< final_points.size() << endl;
        /* ????????? */

        return origin_point;
    }

    // ???????????????
    // ?????????????????????????????????
    /**
    * @brief ???????????????????????????????????????
    *
    * @param[in] mLastRcm                it is used as initialization point to find current unknown orientation of MF Rck_M
    * @param[in] vSurfaceNormal          ??????????????????????????????
    * @param[in] vVanishingDirection      ?????????
    */
    cv::Mat Tracking::TrackManhattanFrame(cv::Mat &mLastRcm, vector<SurfaceNormal> &vSurfaceNormal,
                                          vector<FrameLine> &vVanishingDirection) {
//        cout << "begin Tracking" << endl;
        cv::Mat R_cm_update = mLastRcm.clone();
        int isTracked = 0;
        vector<double> denTemp(3, 0.00001);

        for (int i = 0; i <1; i++) {

            cv::Mat R_cm = R_cm_update;//cv::Mat::eye(cv::Size(3,3),CV_32FC1);  // ????????????1???????????????(3, 3, CV_32FC1);
            int directionFound1 = 0;
            int directionFound2 = 0;
            int directionFound3 = 0; //????????????
            int numDirectionFound = 0;

            vector<axiSNVector> vaxiSNV(4);// surface normal vector
            vector<int> numInCone = vector<int>(3, 0);
            vector<cv::Point2f> vDensity;
            //chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
            // ????????????
            // Transform the current surface normal vector nk expressed in the camera frame into n_k' expressed in MF
            // ?????????????????????
            for (int a = 1; a < 4; a++) {
                // ????????????????????????????????????
                //?????????conic????????? ???
                cv::Mat R_mc = cv::Mat::zeros(cv::Size(3, 3), CV_32F);
                int c1 = (a + 3) % 3;
                int c2 = (a + 4) % 3;
                int c3 = (a + 5) % 3;
                // R_mc = R_cm???xyz?????????????????????
                // TODO ????????????????????????????????????????????????
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
                // ????????????????????????-Conic
                // a??????????????????
                // ???????????????????????????????????????
                vaxiSNV[a - 1] = ProjectSN2Conic(a, R_mc_new, vSurfaceNormal, vVanishingDirection);
                // ????????????????????????????????????
                numInCone[a - 1] = vaxiSNV[a - 1].SNVector.size();
                //cout<<"2 a:"<<vaxiSNV[a-1].axis<<",vector:"<<numInCone[a - 1]<<endl;
            }
            //chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
            //chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
            //cout << "first sN time: " << time_used.count() << endl;
            // ?????????????????????????????????????????????
            int minNumOfSN = vSurfaceNormal.size() / 20;
            //cout<<"minNumOfSN"<<minNumOfSN<<endl;
            //??????  a<b<c
            int a = numInCone[0];
            int b = numInCone[1];
            int c = numInCone[2];
            //cout<<"a:"<<a<<",b:"<<b<<",c:"<<c<<endl;
            // ?????????????????????????????????????????????
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
                // ?????????for??????????????????????????? ?????????for??????????????????????????????????????????
                // ????????????????????? ?????????????????????????????????
                for (int i = 0; i < 3; i++) {
                    if (vaxiSNV[i].axis == a) {

                        tempVVSN = &vaxiSNV[i].SNVector;
                        tempLineDirection = &vaxiSNV[i].Linesvector;
                        break;
                    }

                }
                // ?????????????????????????????????????????????MF
                ResultOfMS RD_temp = ProjectSN2MF(a, R_mc_new, *tempVVSN, *tempLineDirection, minNumOfSN);

                //chrono::steady_clock::time_point t4 = chrono::steady_clock::now();
                //chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t4-t3);
                //cout << "second SN time: " << time_used.count() << endl;

                //cout << "test projectSN2MF" << ra << endl;
                if (sum(RD_temp.R_cm_Rec)[0] != 0) {
                    numDirectionFound += 1;
                    if (a == 1) directionFound1 = 1;//????????????
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
        // ????????????????????????????????????50,???????????????15
        if (mCurrentFrame.N > 50 || mCurrentFrame.NL > 15) {
            // Set Frame pose to the origin
            // ?????????????????????????????????????????????0
            cv::Mat E = cv::Mat::eye(4, 4, CV_32F);
            cv::Mat Rotation_cw = Rotation_cm;

            Rotation_cw.copyTo(E.rowRange(0,3).colRange(0,3));

            mCurrentFrame.SetPose(E);
            cout <<"**********##########\n"<< E << endl;
            // Create KeyFrame
            // ??????????????????????????????????????????
            // KeyFrame??????Frame?????????3D????????????BoW

            KeyFrame *pKFini = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB,this->mImRGB, this->mCurrentFrame.depth);
//            KeyFrame *pKFini = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);
            // Insert KeyFrame in the map
            // ?????????????????????????????????
            mpMap->AddKeyFrame(pKFini);

            // Create MapPoints and asscoiate to KeyFrame
            // ????????????????????????????????????
            for (int i = 0; i < mCurrentFrame.N; i++) {
                float z = mCurrentFrame.mvDepth[i];
                if (z > 0) {
                    // ?????????????????????????????????????????????????????????3D??????
                    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                    // ???3D????????????MapPoint
                    MapPoint *pNewMP = new MapPoint(x3D, pKFini, mpMap);
                    // ??????MapPoint???????????????
                    // a.????????????MapPoint????????????
                    // b.???MapPoint????????????
                    // c.???MapPoint????????????????????????????????????
                    pNewMP->AddObservation(pKFini, i);
                    pKFini->AddMapPoint(pNewMP, i);
                    pNewMP->ComputeDistinctiveDescriptors();
                    pNewMP->UpdateNormalAndDepth();
                    // ?????????????????????MapPoint
                    mpMap->AddMapPoint(pNewMP);
                    // ??????MapPoint?????????????????????mvpMapPoints???
                    // ?????????Frame???????????????MapPoint??????????????????
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
                // ????????????
                // ??????????????????????????? ???????????????????????????
                cv::Mat p3D = mCurrentFrame.ComputePlaneWorldCoeff(i);
                // ??????????????????????????????????????????
                MapPlane *pNewMP = new MapPlane(p3D, pKFini, mpMap);
                // ????????????????????????????????????  ????????? ???????????????????????????id??????????????????
                // ??????????????????????????????????????????????????????????????????????????????id
                pNewMP->AddObservation(pKFini,i);
                pKFini->AddMapPlane(pNewMP, i);
                pNewMP->UpdateCoefficientsAndPoints();
                mpMap->AddMapPlane(pNewMP);
                // ????????????plane??????????????????????????????Plane
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
        // ??????????????????????????????????????????????????????????????????
        if (!mpInitializer) {
            // Set Reference Frame
            if (mCurrentFrame.mvKeys.size() > num) {
                // step 1????????????????????????????????????????????????????????????
                mInitialFrame = Frame(mCurrentFrame);
                // ?????????????????????
                mLastFrame = Frame(mCurrentFrame);
                // mvbPreMatched???????????????????????????????????????????????????????????????
                mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
                for (size_t i = 0; i < mCurrentFrame.mvKeysUn.size(); i++)
                    mvbPrevMatched[i] = mCurrentFrame.mvKeysUn[i].pt;

                if (mpInitializer)
                    delete mpInitializer;

                // ????????????????????????????????? sigma:1.0    iterations:200
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

            LSDmatcher lmatcher;   //??????????????????????????????
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
        // ??????????????????????????????????????????????????????????????????????????????????????????
        mInitialFrame.SetPose(cv::Mat::eye(4,4,CV_32F));
        cv::Mat Tcw = cv::Mat::eye(4,4,CV_32F);
        Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
        tcw.copyTo(Tcw.rowRange(0,3).col(3));
        mCurrentFrame.SetPose(Tcw);

        // step6????????????????????????3D????????????MapPoints
        /// ????????????????????????????????????????????????
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
                // ??????????????????????????????????????????????????????????????????????????????????????????
                mInitialFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));
                cv::Mat Tcw = cv::Mat::eye(4, 4, CV_32F);
                Rcw.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
                tcw.copyTo(Tcw.rowRange(0, 3).col(3));
                mCurrentFrame.SetPose(Tcw);

                // step6????????????????????????3D????????????MapPoints
                /// ????????????????????????????????????????????????
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

            // b.?????????????????????MapPoint???????????????????????????????????????????????????
            pMP->ComputeDistinctiveDescriptors();

            // c.?????????MapPoint????????????????????????????????????????????????
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

        mState = OK;  //????????????????????????
    }

#if 1

/**
* @brief ???????????????????????????????????????????????????Map?????????MapPoints???MapLine
*/
    void Tracking::CreateInitialMapMonoWithLine() {
        // step1:
        KeyFrame *pKFini = new KeyFrame(mInitialFrame, mpMap, mpKeyFrameDB);
        KeyFrame *pKFcur = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

        // step2???
        pKFini->ComputeBoW();
        pKFcur->ComputeBoW();

        // step3???
        mpMap->AddKeyFrame(pKFini);
        mpMap->AddKeyFrame(pKFcur);

        // step4???
        for (size_t i = 0; i < mvIniMatches.size(); i++) {
            if (mvIniMatches[i] < 0)
                continue;

            // Create MapPoint
            cv::Mat worldPos(mvIniP3D[i]);

            // step4.1???
            MapPoint *pMP = new MapPoint(worldPos, pKFcur, mpMap);

            // step4.2???

            // step4.3???
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
            // step4.4???
            mpMap->AddMapPoint(pMP);
        }

        // step5???
        for (size_t i = 0; i < mvLineMatches.size(); i++) {
            if (!mvbLineTriangulated[i])
                continue;

            // Create MapLine
            Vector6d worldPos;
            worldPos << mvLineS3D[i].x, mvLineS3D[i].y, mvLineS3D[i].z, mvLineE3D[i].x, mvLineE3D[i].y, mvLineE3D[i].z;

            //step5.1???
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
    * @brief ????????????????????????????????????????????????????????????????????????
    * Step 1?????????????????????????????????????????????RGB-D???????????????????????????????????????????????????
    * Step 2????????????????????????????????????????????????????????????
    * Step 3????????????????????????
    * Step 4???????????????????????????
     * @return ?????????????????????10??????????????????????????????true
 */
    bool Tracking::TrackWithMotionModel(bool bStruct){
        ORBmatcher matcher(0.9, true);
        LSDmatcher lmatcher;
        // ????????????????????????????????????
        PlaneMatcher pmatcher(mfDThRef, mfAThRef, mfVerTh, mfParTh);

        // Update last frame pose according to its reference keyframe
        // Create "visual odometry" points if in Localization Mode
        // Step 1?????????????????????????????????????????????RGB-D???????????????????????????????????????????????????
        UpdateLastFrame();
        // initial estimate
        mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);

        // Project points seen in previous frame
        int th = 15;
        // ????????????????????????
        fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));
        int nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, th, mSensor == System::MONOCULAR);
//        cout << "TrackWithMotionModel line matching"<< endl;
        int planeMatches=0;
        int lmatches = 0;
        //??????????????? ???
        if(!mbOnlyTracking) {
        vector<MapLine *> vpMapLineMatches;
        fill(mCurrentFrame.mvpMapLines.begin(), mCurrentFrame.mvpMapLines.end(), static_cast<MapLine *>(NULL));
        lmatches = lmatcher.SearchByDescriptor(mpReferenceKF, mCurrentFrame, vpMapLineMatches);
        mCurrentFrame.mvpMapLines = vpMapLineMatches;
        }

//        cout << "TrackWithMotionModel line matched"<< endl;
        // ???????????????
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
        // ??????????????????track????????????    ?????????????????????MF????????????
//        bool track_quality = true;
//        if(!mbOnlyTracking) {
//             track_quality =  pmatcher.bMatchStatus(mCurrentFrame, mpMap->GetAllMapPlanes(),true,mRotation_wc.t());
//        }
//        if(track_quality == false) {
//            // Optimize frame pose with all matches
//            // ?????????????????????
//            // ?????????????????????????????????????????? ???????????????????????????????????????????????????
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
        // Step 1????????????????????????????????????BoW??????
        mCurrentFrame.ComputeBoW();

        // We perform first an ORB matching with the reference keyframe
        // If enough matches are found we setup a PnP solver
        // Step 2???????????????BoW???????????????????????????????????????????????????
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
        // ??????????????????????????????????????????R??????????????????????????????????????????
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
        // Step 1????????????????????????????????????BoW??????
        mCurrentFrame.ComputeBoW();

        // We perform first an ORB matching with the reference keyframe
        // If enough matches are found we setup a PnP solver
        // Step 2???????????????BoW???????????????????????????????????????????????????
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
        // ????????????????????????????????????
        PlaneMatcher pmatcher(mfDThRef, mfAThRef, mfVerTh, mfParTh);

        // Update last frame pose according to its reference keyframe
        // Create "visual odometry" points if in Localization Mode
        // Step 1?????????????????????????????????????????????RGB-D???????????????????????????????????????????????????
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
            th = 7;// ??????????????????????????? ????????????????????????
        // ???????????????????????????
        fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));

        // ???????????????????????????????????????????????????????????????   ???????????? ????????????????????????????????????????????????????????????
        int nmatches = matcher.SearchByProjection(mCurrentFrame, mLastFrame, th, mSensor == System::MONOCULAR);

        vector<MapLine *> vpMapLineMatches;
        int lmatches = lmatcher.SearchByDescriptor(mpReferenceKF, mCurrentFrame, vpMapLineMatches);
        mCurrentFrame.mvpMapLines = vpMapLineMatches;


        // ???????????????
        if (nmatches <50) {
            fill(mCurrentFrame.mvpMapPoints.begin(), mCurrentFrame.mvpMapPoints.end(), static_cast<MapPoint *>(NULL));
            nmatches = matcher.MatchORBPoints(mCurrentFrame, mLastFrame);
            mUpdateMF = true;
        }

        // ???????????????????????????????????? ??????????????????????????????????????????
        int planeMatches = pmatcher.SearchMapByCoefficients(mCurrentFrame, mpMap->GetAllMapPlanes(),mRotation_wc);


        int initialMatches = nmatches + lmatches + planeMatches;

        if (initialMatches < 10) {
            cout << "TranslationWithMotionModel: Before: Not enough matches" << endl;
            return false;
        }

        // Optimize frame pose with all matches
        // ?????????????????????
        // ?????????????????????????????????????????? ???????????????????????????????????????????????????
        mRotation_wc.copyTo(mCurrentFrame.mTcw.rowRange(0,3).colRange(0,3));

        //cout << "translation motion model,pose before opti" << mCurrentFrame.mTcw << endl;
        // Step 4?????????3D-2D????????????????????????????????????
        // ?????????????????????????????????????????????R???t
        // ??????????????????????????????R??????    ??????R???????????????????????????????????????J??????????????????0
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
        // Step 1?????????????????????????????????????????????????????????????????????
        // ????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
        KeyFrame *pRef = mLastFrame.mpReferenceKF;
        // ref_keyframe ??? lastframe???????????????
        cv::Mat Tlr = mlRelativeFramePoses.back();
        // ??????????????????????????????????????????????????????
        // l:last, r:reference, w:world
        // Tlw = Tlr*Trw
        mLastFrame.SetPose(Tlr * pRef->GetPose());
        // ???????????????????????????????????????????????????????????????
        if (mnLastKeyFrameId == mLastFrame.mnId || mSensor == System::MONOCULAR || !mbOnlyTracking)
            return;

        // Create "visual odometry" MapPoints
        // We sort points according to their measured depth by the stereo/RGB-D sensor
        // Step 2??????????????????rgbd????????????????????????????????????????????????
        // ???????????????????????????????????????????????????????????????????????????????????????

        // Step 2.1?????????????????????????????????????????????????????????????????????????????????
        vector<pair<float, int> > vDepthIdx;
        vDepthIdx.reserve(mLastFrame.N);
        for (int i = 0; i < mLastFrame.N; i++) {
            float z = mLastFrame.mvDepth[i];
            if (z > 0) {
                // vDepthIdx????????????????????????????????????,????????????????????????????????????id
                vDepthIdx.push_back(make_pair(z, i));
            }
        }
        // ??????????????????????????????
        sort(vDepthIdx.begin(), vDepthIdx.end());

        // We insert all close points (depth<mThDepth)
        // If less than 100 close points, we insert the 100 closest ones.
        // Step 2.2???????????????????????????????????????
        int nPoints = 0;
        for (size_t j = 0; j < vDepthIdx.size(); j++) {
            int i = vDepthIdx[j].second;

            bool bCreateNew = false;

            // ??????????????????????????????????????????????????????,????????????????????????????????????,???????????????????????????????????????
            MapPoint *pMP = mLastFrame.mvpMapPoints[i];
            if (!pMP)
                bCreateNew = true;
            else if (pMP->Observations() < 1) {
                // ?????????????????????????????????????????????????????????????????????????????????
                bCreateNew = true;
            }

            if (bCreateNew) {
                // Step 2.3????????????????????????????????????????????????????????????????????????RGBD????????????????????????????????????????????????????????????????????????
                // ??????????????????????????????
                cv::Mat x3D = mLastFrame.UnprojectStereo(i);
                MapPoint *pNewMP = new MapPoint(x3D, mpMap, &mLastFrame, i);
                // ??????????????????????????????
                mLastFrame.mvpMapPoints[i] = pNewMP;

                // ????????????????????????MapPoint????????????CreateNewKeyFrame?????????????????????
                mlpTemporalPoints.push_back(pNewMP);
                nPoints++;
            } else {
                nPoints++;
            }
            // Step 2.4??????????????????????????????????????????????????????
            // ????????????????????????????????????????????????????????????
            // 1???????????????????????????????????????????????????????????????35????????????
            // 2???nPoints????????????100??????????????????????????????????????????????????????????????????
            if (vDepthIdx[j].first > mThDepth && nPoints > 100)
                break;
        }

        // ????????????????????????
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
        // Step 2??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
        thread threadPoints(&Tracking::SearchLocalPoints, this);
//        thread threadLines(&Tracking::SearchLocalLines, this);
        thread threadPlanes(&Tracking::SearchLocalPlanes, this);
        threadPoints.join();
//        threadLines.join();
        threadPlanes.join();

        // ???????????????R_MF
        pmatcher.SearchMapByCoefficients(mCurrentFrame, mpMap->GetAllMapPlanes(),mRotation_wc);

        //cout << "tracking localmap with lines, pose before opti" << endl << mCurrentFrame.mTcw << endl;
        Optimizer::PoseOptimization(&mCurrentFrame,bStruct);
        //cout << "tracking localmap with lines, pose after opti" << mCurrentFrame.mTcw << endl;

        // ??????????????????track????????????    ?????????????????????MF????????????
        //????????????????????????????????????
//        Optimizer::PoseOptimization_strict(&mCurrentFrame); // ???????????? ???????????????  ??????



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
 * @brief ??????????????????????????????????????????
 *
 * Step 1??????VO?????????????????????????????????????????????????????????????????????????????????????????????
 * Step 2?????????????????????????????????????????????????????????????????????????????????????????????????????????
 * Step 3???????????????????????????????????????????????????
 * Step 4??????????????????????????????????????????,??????????????????????????????????????????
 * Step 5??????????????????RGBD?????????????????????????????????????????????????????? ??? ???????????????????????????
 * Step 6????????????????????????????????????
 * @return true         ??????
 * @return false        ?????????
 */
    bool Tracking::NeedNewKeyFrame() {
        // ???VO???????????????????????????
        if (mbOnlyTracking)
            return false;

        // If Local Mapping is freezed by a Loop Closure do not insert keyframes
        // Step 2????????????????????????????????????????????????????????????????????????
        if (mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
            return false;
        // ???????????????????????????????????????
        const int nKFs = mpMap->KeyFramesInMap();

        // Do not insert keyframes if not enough frames have passed from last relocalisation
        // mCurrentFrame.mnId???????????????ID
        // mnLastRelocFrameId??????????????????????????????ID
        // mMaxFrames???????????????????????????
        //  Step 3?????????????????????????????????????????????????????????????????????????????????????????????????????????
        if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && nKFs > mMaxFrames)
            return false;

        // Tracked MapPoints in the reference keyframe
        // ??????????????????????????????
        int nMinObs = 3;
        if (nKFs <= 2)
            nMinObs = 2;
        // ??????????????????????????????????????????>= nMinObs??????????????????
        int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

        // Local Mapping accept keyframes?
        // Step 5???????????????????????????????????????????????????????????????????????????
        bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

// Stereo & RGB-D: Ratio of close "matches to map"/"total matches"
// "total matches = matches to map + visual odometry matches"
// Visual odometry matches will become MapPoints if we insert a keyframe.
// This ratio measures how many MapPoints we could create if we insert a keyframe.
// Step 6??????????????????RGBD??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
        int nMap = 0; //nTrackedClose //?????????RGB-D???????????????????????????????????????
        int nTotal = 0;
        int nNonTrackedClose = 0;//?????????RGB-D???????????????????????????
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
        // Step 7????????????????????????????????????
        // Thresholds
        // Step 7.1?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
        float thRefRatio = 0.75f;
        if (nKFs < 2)
            thRefRatio = 0.4f;

        if (mSensor == System::MONOCULAR)
            thRefRatio = 0.9f;

        float thMapRatio = 0.35f;
        if (mnMatchesInliers > 300)
            thMapRatio = 0.20f;

// Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
        // Step 7.2???????????????????????????????????????????????????
        const bool c1a = mCurrentFrame.mnId >= mnLastKeyFrameId + mMaxFrames;
// Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
        const bool c1b = (mCurrentFrame.mnId >= mnLastKeyFrameId + mMinFrames && bLocalMappingIdle);
//Condition 1c: tracking is weak
        // Step 7.4???????????????RGB-D?????????????????????????????????????????????????????????0.25????????????????????????bNeedToInsertClose
        const bool c1c = mSensor != System::MONOCULAR && (mnMatchesInliers < nRefMatches * 0.25 || bNeedToInsertClose);
// Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
        // Step 7.5???????????????????????????????????????????????? ????????????bNeedToInsertClose??????????????????????????????????????????
        const bool c2 = ((mnMatchesInliers < nRefMatches * thRefRatio || bNeedToInsertClose) &&
                         mnMatchesInliers > 15);



//        bool bNeedToInsertClose = (nMap<100) && (nNonTrackedClose>70);
//        // Condition 1c: tracking is weak
//        // Step 7.4???????????????RGB-D?????????????????????????????????????????????????????????0.25????????????????????????bNeedToInsertClose
//        const bool c1c =  mSensor!=System::MONOCULAR &&             //?????????????????????RGB-D?????????
//                          (mnMatchesInliers<nRefMatches*0.25 ||       //?????????????????????????????????????????????
//                           bNeedToInsertClose) ;                     //????????????
//
//        // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
//        // Step 7.5???????????????????????????????????????????????? ????????????bNeedToInsertClose??????????????????????????????????????????
//        const bool c2 = ((mnMatchesInliers<nRefMatches*thRefRatio|| bNeedToInsertClose) && mnMatchesInliers>15);

//           std::cout << c1a << " "<< c1b << " "<< c1c <<" "<< mCurrentFrame.mbNewPlane<<std::endl;
//        if (((c1a || c1b || c1c) && c2) || mCurrentFrame.mbNewPlane) {
          if (((c1a || c1b || c1c) && c2)) {
            // If the mapping accepts keyframes, insert keyframe.
            // Otherwise send a signal to interrupt BA
            // Step 7.6???local mapping?????????????????????????????????????????????????????????????????????
            if (bLocalMappingIdle) {
                return true;
            } else {
                // ????????????????????????????????????
                // tracking??????????????????????????????????????????????????????mlNewKeyFrames??????
                // ??????localmapper?????????pop???????????????mspKeyFrames
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
 * @brief ?????????????????????
 * ?????????????????????????????????????????????MapPoints
 *
 * Step 1?????????????????????????????????
 * Step 2?????????????????????????????????????????????????????????
 * Step 3??????????????????rgbd????????????????????????????????????MapPoints
 */
    void Tracking::CreateNewKeyFrame() {
        // ?????????????????????????????????,????????????????????????
        if (!mpLocalMapper->SetNotStop(true))
            return;

        // Step 1?????????????????????????????????
        KeyFrame *pKF = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB,this->mImRGB, this->mCurrentFrame.depth);
//        KeyFrame *pKF = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);
        // Step 2?????????????????????????????????????????????????????????
        // ???UpdateLocalKeyFrames???????????????????????????????????????????????????????????????????????????????????????????????????
        mpReferenceKF = pKF;
        mCurrentFrame.mpReferenceKF = pKF;

        // ??????????????? Tracking::UpdateLastFrame ????????????????????????????????????
        // Step 3??????????????????rgbd???????????????????????????????????????????????????????????????
        if (mSensor != System::MONOCULAR) {

            mCurrentFrame.UpdatePoseMatrices();
            // We sort points by the measured depth by the stereo/RGBD sensor.
            // We create all those MapPoints whose depth < mThDepth.
            // If there are less than 100 close points we create the 100 closest.\
            // Step 3.1?????????????????????????????????????????????????????????????????????
            vector<pair<float, int> > vDepthIdx;
            vDepthIdx.reserve(mCurrentFrame.N);

            for (int i = 0; i < mCurrentFrame.N; i++) {
                float z = mCurrentFrame.mvDepth[i];
                if (z > 0) {
                    // ????????????????????????,???????????????????????????????????????id
                    vDepthIdx.push_back(make_pair(z, i));
                }
            }

            if (!vDepthIdx.empty()) {
                // Step 3.2?????????????????????????????????
                sort(vDepthIdx.begin(), vDepthIdx.end());

                // Step 3.3??????????????????????????????????????????????????????
                // ????????????????????????
                int nPoints = 0;
                for (size_t j = 0; j < vDepthIdx.size(); j++) {
                    int i = vDepthIdx[j].second;

                    bool bCreateNew = false;

                    // ??????????????????????????????????????????????????????,????????????????????????????????????,???????????????????????????????????????
                    MapPoint *pMP = mCurrentFrame.mvpMapPoints[i];
                    if (!pMP)
                        bCreateNew = true;
                    else if (pMP->Observations() < 1) {
                        bCreateNew = true;
                        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint *>(NULL);
                    }

                    // ?????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
                    if (bCreateNew) {
                        cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                        MapPoint *pNewMP = new MapPoint(x3D, pKF, mpMap);
                        // ??????????????????????????????????????????MapPoint???????????????
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

                //??????????????????   ??????????????????????????????????????????????????????????????????????????????????????????
                if(Regularity == true) {
//                     ?????????????????????????????? ??????????????????????????????
                    cout << "?????????????????????????????? ??????????????????????????????"<< endl;
                    cv::Mat p3D = mCurrentFrame.ComputePlaneWorldCoeff(i);
                    cv::Mat A = p3D(cvRect(0,0,1,3));
//
//                    // ????????????????????????????????????????????? ???????????????????????????????????????
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
                    // ??????????????????
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

    /*???????????????????????????????????????
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
                nToMatch++; //??????match???
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
        // step1???
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

        // step2???
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
        // step1???
        mvpLocalMapLines.clear();

        // step2???
        for (vector<KeyFrame *>::const_iterator itKF = mvpLocalKeyFrames.begin(), itEndKF = mvpLocalKeyFrames.end();
             itKF != itEndKF; itKF++) {
            KeyFrame *pKF = *itKF;
            const vector<MapLine *> vpMLs = pKF->GetMapLineMatches();

            //step3????????????????????????MapLines?????????mvpLocalMapLines
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
