#ifndef TRACKING_H
#define TRACKING_H

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include"Viewer.h"
#include"FrameDrawer.h"
#include"Map.h"
#include"LocalMapping.h"
#include"LoopClosing.h"
#include"Frame.h"
#include "ORBVocabulary.h"
#include"KeyFrameDatabase.h"
#include"ORBextractor.h"
#include "Initializer.h"
#include "MapDrawer.h"
#include "System.h"


#include <mutex>

#include "auxiliar.h"
//#include "ExtractLineSegment.h"
#include "MapLine.h"
#include "LSDmatcher.h"
#include "PlaneMatcher.h"

#include "MeshViewer.h"
#include "MapPlane.h"
#include "PangolinViewer.h"

#include "YOLOX.h"

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>

#include <cv_bridge/cv_bridge.h>

class MeshViewer;

namespace Planar_SLAM
{

class Viewer;
class FrameDrawer;
class Map;
class LocalMapping;
class LoopClosing;
class System;
class Frame;
class YOLOX;

class Tracking
{  

public:
    // TO DO

    
    Tracking(System* pSys, ORBVocabulary* pVoc, FrameDrawer* pFrameDrawer, MapDrawer* pMapDrawer, Map* pMap,
             KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor, ros::NodeHandle& nh);

    // Preprocess the input and call Track(). Extract features and performs stereo matching.
    cv::Mat GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp,Eigen::Quaterniond &GroundTruth_R);

    void SetLocalMapper(LocalMapping* pLocalMapper);
    void SetLoopClosing(LoopClosing* pLoopClosing);
    void SetViewer(Viewer* pViewer);
    void SetPangolinViewer(PangolinViewer *pViewer);

    // Load new settings
    // The focal lenght should be similar or scale prediction will fail when projecting points
    // TODO: Modify MapPoint::PredictScale to take into account focal lenght
    void ChangeCalibration(const string &strSettingPath);

    // Use this function if you have deactivated local mapping and you only want to localize the camera.
    void InformOnlyTracking(const bool &flag);

    //obtain rotation matrix from the Manhattan World assumption
    cv::Mat SeekManhattanFrame(vector<SurfaceNormal>  &vSurfaceNormal,vector<FrameLine>&vVanishingDirection);
    cv::Mat TrackManhattanFrame(cv::Mat &mLastRcm,vector<SurfaceNormal> &vSurfaceNormal,vector<FrameLine>&vVanishingDirection);

    //
    sMS MeanShift(vector<cv::Point2d> & v2D);
    cv::Mat ProjectSN2MF(int a,const cv::Mat &R_cm,const vector<SurfaceNormal> &vTempSurfaceNormal,vector<FrameLine> &vVanishingDirection);
    ResultOfMS ProjectSN2MF(int a,const cv::Mat &R_cm,const vector<SurfaceNormal> &vTempSurfaceNormal,vector<FrameLine> &vVanishingDirection,const int numOfSN);
    axiSNV ProjectSN2Conic(int a,const cv::Mat &R_cm,const vector<SurfaceNormal> &vTempSurfaceNormal,vector<FrameLine> &vVanishingDirection);
    vector<cv::Point2d> Outlier_Filter(vector<cv::Point2d> &origin_point);

    cv::Mat ClusterMultiManhattanFrame(vector<cv::Mat> &vRotationCandidate,double &clusterRatio);
    vector<vector<int>>  EasyHist(vector<float> &vDistance,int &histStart,float &histStep,int&histEnd);
    void SaveMesh(const string &filename);

    // 计算曼哈顿旋转跟运动模型R的差异
    double MatrixResidual(cv::Mat Mahattan_R,cv::Mat Track_R);
    ros::Publisher odometry_pub,depth_pub;
public:

    // Tracking states
    enum eTrackingState{
        SYSTEM_NOT_READY=-1,
        NO_IMAGES_YET=0,
        NOT_INITIALIZED=1,
        OK=2,
        LOST=3
    };

    eTrackingState mState;
    eTrackingState mLastProcessedState;

    // update MF rotation
    bool mUpdateMF;
    // Input sensor
    int mSensor;


    // Current Frame
    Frame mCurrentFrame;
    cv::Mat mImRGB;
    cv::Mat mImGray;
    cv::Mat mImDepth;
    cv::Mat floor_normal;

    // Initialization Variables (Monocular)
    std::vector<int> mvIniLastMatches;
    std::vector<int> mvIniMatches;
    std::vector<cv::Point2f> mvbPrevMatched;
    std::vector<cv::Point3f> mvIniP3D;
    Frame mInitialFrame;

    vector<pair<int, int>> mvLineMatches;
    vector<cv::Point3f> mvLineS3D;   //start point
    vector<cv::Point3f> mvLineE3D;   //end point
    vector<bool> mvbLineTriangulated;   //
    // Lists used to recover the full camera trajectory at the end of the execution.
    // Basically we store the reference keyframe for each frame and its relative transformation
    list<cv::Mat> mlRelativeFramePoses;
    list<KeyFrame*> mlpReferences;
    list<double> mlFrameTimes;
    list<bool> mlbLost;

    cv::Mat Rotation_cm;
    cv::Mat Rotation_wm;
    cv::Mat mRotation_wc;// 这里应该是cw
    cv::Mat mLastRcm;
    cv::Mat Rotation_gc;

    // True if local mapping is deactivated and we are performing only localization
    bool mbOnlyTracking;

    void Reset();

    shared_ptr<MeshViewer>  mpPointCloudMapping;

    void SetSemanticer(YOLOX *detector);//yolox

    // NOTE
    YOLOX* Semanticer;

protected:

    // Main tracking function. It is independent of the input sensor.
    void Track(Eigen::Quaterniond &GroundTruth_R);

    // Map initialization for stereo and RGB-D
    void StereoInitialization();

    // Map initialization for monocular
    void MonocularInitialization();
    void CreateInitialMapMonocular();
    void CreateInitialMapMonoWithLine();

    // Map initialization for stereo and RGB-D when reloading a map.
    void StereoInitializationWithMap();

    void CheckReplacedInLastFrame();
    bool TrackReferenceKeyFrame(bool bStruct);
    void UpdateLastFrame();
    bool TrackWithMotionModel(bool bStruct);
    bool TranslationEstimation(bool bStruct);
    bool TranslationWithMotionModel(Eigen::Quaterniond &GroundTruth_R,bool bStruct);
    bool Relocalization(bool bStruct);

    void UpdateLocalMap();
    void UpdateLocalPoints();
    void UpdateLocalLines();
    void UpdateLocalKeyFrames();

    bool TrackLocalMap(bool bStruct);
    void SearchLocalPoints();
    void SearchLocalLines();
    void SearchLocalPlanes();


    bool NeedNewKeyFrame();
    void CreateNewKeyFrame();

    void UpdatePlane(cv::Mat R_cm);

    // In case of performing only localization, this flag is true when there are no matches to
    // points in the map. Still tracking will continue if there are enough matches with temporal points.
    // In that case we are doing visual odometry. The system will try to do relocalization to recover
    // "zero-drift" localization to the map.
    bool mbVO;

    //Other Thread Pointers
    LocalMapping* mpLocalMapper;
    LoopClosing* mpLoopClosing;

    //ORB
    ORBextractor* mpORBextractorLeft, *mpORBextractorRight;
    ORBextractor* mpIniORBextractor;


    //BoW
    ORBVocabulary* mpORBVocabulary;
    KeyFrameDatabase* mpKeyFrameDB;

    // Initalization (only for monocular)
    Initializer* mpInitializer;

    //Local Map
    KeyFrame* mpReferenceKF;
    std::vector<KeyFrame*> mvpLocalKeyFrames;
    std::vector<MapPoint*> mvpLocalMapPoints;
    std::vector<MapLine*> mvpLocalMapLines;
    // Surface Normal
    SurfaceNormal_M mvpSurfaceNormals_M;

    // System
    System* mpSystem;
    
    //Drawers
    Viewer* mpViewer;
    PangolinViewer* mpPangolinViewer;
    FrameDrawer* mpFrameDrawer;
    MapDrawer* mpMapDrawer;

    //Map
    Map* mpMap;
    //Calibration matrix
    cv::Mat mK;
    cv::Mat mDistCoef;
    float mbf;

    //自己添加的，两个用于纠正畸变的映射矩阵
    cv::Mat mUndistX, mUndistY;
    //New KeyFrame rules (according to fps)
    int mMinFrames;
    int mMaxFrames;

    // Threshold close/far points
    // Points seen as close by the stereo/RGBD sensor are considered reliable
    // and inserted from just one frame. Far points requiere a match in two keyframes.
    float mThDepth;

    // For RGB-D inputs only. For some datasets (e.g. TUM) the depthmap values are scaled.
    float mDepthMapFactor;

    //Current matches in frame
    int mnMatchesInliers;
    int mnLineMatchesInliers;   //线特征

    //Last Frame, KeyFrame and Relocalisation Info
    KeyFrame* mpLastKeyFrame;
    Frame mLastFrame;
    unsigned int mnLastKeyFrameId;
    unsigned int mnLastRelocFrameId;

    //Motion Model
    cv::Mat mVelocity;

    //Color order (true RGB, false BGR, ignored if grayscale)
    bool mbRGB;

    list<MapPoint*> mlpTemporalPoints;
    list<MapLine*>mlpTemporalLines;

    // 在追踪时使用的阈值
    float mfDThRef;//平面匹配
    float mfDThMon;
    float mfAThRef;//平面匹配
    float mfAThMon;

    float mfVerTh;
    float mfParTh;

    float mMax_merge_dist;
    int mPatch_size;
    float mMax_point_dist;

    int manhattanCount;
    int fullManhattanCount;

};

} //namespace ORB_SLAM

#endif // TRACKING_H
