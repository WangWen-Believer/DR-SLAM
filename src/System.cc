#include "System.h"
#include "Converter.h"
#include <thread>
#include <pangolin/pangolin.h>
#include <iomanip>
#include <time.h>

#include "YOLOX.h"
#include "SystemSetting.h"
#include <cv_bridge/cv_bridge.h>

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>

using namespace std;
using namespace cv;
using namespace cv::line_descriptor;
using namespace Eigen;


namespace Planar_SLAM
{
    System::System(const string &strVocFile, const string &strSettingsFile, const eSensor sensor,
                   const bool bUseViewer):mSensor(sensor), mpViewer(static_cast<Viewer*>(NULL)),mbReset(false),mbActivateLocalizationMode(false),
                                                              mbDeactivateLocalizationMode(false)
    {
        // Output welcome message
        cout << endl <<
             "PlanarSLAM Updated by WangWen" << endl  <<
             "This is free software based on ORB-SLAM2, and you are welcome to redistribute it" << endl <<
             "under certain conditions. See LICENSE.txt." << endl << endl;

        //Check settings file
        cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
        if(!fsSettings.isOpened())
        {
            cerr << "Failed to open settings file at: " << strSettingsFile << endl;
            exit(-1);
        }

        // TO DO
        //float resolution = fsSettings["PointCloudMapping.Resolution"];
        //float resolution = 0.01;

        //Load ORB Vocabulary
        cout << endl << "Loading ORB Vocabulary. This could take a while..." << endl;
        clock_t tStart = clock();
        mpVocabulary = new ORBVocabulary();
        bool bVocLoad  = mpVocabulary->loadFromTextFile(strVocFile);
        //else
        //   bVocLoad = mpVocabulary->loadFromBinaryFile(strVocFile);
        if (!bVocLoad)
        {
            cerr << "Wrong path to vocabulary. " << endl;
            cerr << "Failed to open at: " << strVocFile << endl;
            exit(-1);
        }
        printf("Vocabulary loaded in %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
        cout<<"Vocabulary loaded!"<<endl<<endl;
        //Create KeyFrame Database
        mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);
        //Create the Map
        mpMap = new Map();
        //Create Drawers. These are used by the Viewer
        //这里的帧绘制器和地图绘制器将会被可视化的Viewer所使用
        mpFrameDrawer = new FrameDrawer(mpMap);
        mpMapDrawer = new MapDrawer(mpMap, strSettingsFile);

        //Initialize the Tracking thread
        //(it will live in the main thread of execution, the one that called this constructor)
        // TO DO
        //mpPointCloudMapping = make_shared<PointCloudMapping>( resolution );

//        mpTracker = new Tracking(this, mpVocabulary, mpFrameDrawer, mpMapDrawer,
//                                 mpMap, mpKeyFrameDatabase, strSettingsFile, mSensor);

        //Initialize the Local Mapping thread and launch
        mpLocalMapper = new LocalMapping(mpMap, mSensor==MONOCULAR);
        mptLocalMapping = new thread(&Planar_SLAM::LocalMapping::Run, mpLocalMapper);

        //Initialize the Loop Closing thread and launch
        mpLoopCloser = new LoopClosing(mpMap, mpKeyFrameDatabase, mpVocabulary, mSensor!=MONOCULAR);
        mptLoopClosing = new thread(&Planar_SLAM::LoopClosing::Run, mpLoopCloser);

        // _____________________________yolox_______________________________
        std::string engineFile = "/home/wangwen/Desktop/A_SLAM_Learning/YOLOX-CPP/model_trt.engine";
        mpSemanticer = new YOLOX(engineFile);

        //Initialize the Viewer thread and launch
        if(bUseViewer)
        {
            mpViewer = new Viewer(this, mpFrameDrawer,mpMapDrawer,mpTracker,strSettingsFile);
            mptViewer = new thread(&Viewer::RunWithPLP, mpViewer);
            mpTracker->SetViewer(mpViewer);
        }
        if(bUseViewer)
        {
            mpPangolinViewer = new PangolinViewer(this, 			//又是这个
                                                  mpFrameDrawer,	//帧绘制器
                                                  mpMapDrawer,		//地图绘制器
                                                  mpTracker,		//追踪器
                                                  strSettingsFile);	//配置文件的访问路径
            //新建viewer线程
            mptPangolinViewer = new thread(&PangolinViewer::Run, mpPangolinViewer);
            //给运动追踪器设置其查看器
            mpTracker->SetPangolinViewer(mpPangolinViewer);
        }

        // Choose to use pure localization mode
        char IsPureLocalization;
        cout << "Do you want to run pure localization?(y/n)" << endl;
        cin >> IsPureLocalization;
        if(IsPureLocalization == 'Y' || IsPureLocalization == 'y'){
            ActivateLocalizationMode();
        }
        // Load map
        char IsLoadMap;

        //get the current absoulte path
        std::string cwd = getcwd(NULL, 0);
        cout << "The current dir is : " << cwd << endl;
        string strPathSystemSetting = strSettingsFile.c_str();

        cout << "Your setting file path is : " << strPathSystemSetting << endl;

        string strPathMap = "/home/wangwen/catkin_Planar/src/PlanarSLAM/MapPointandKeyFrame.bin";
        cout << "Your map file path would be : " << strPathMap << endl;

//        cout << "Do you want to load the map?(y/n)" << endl;
        cin >> IsLoadMap;
        SystemSetting *mySystemSetting = new SystemSetting(mpVocabulary);
        mySystemSetting->LoadSystemSetting(strPathSystemSetting);
        if(IsLoadMap == 'Y' || IsLoadMap == 'y'){
            mpMap->Load(strPathMap, mySystemSetting, mpKeyFrameDatabase);
        }

        //Set pointers between threads
        mpTracker->SetLocalMapper(mpLocalMapper);
        mpTracker->SetLoopClosing(mpLoopCloser);

        mpLocalMapper->SetTracker(mpTracker);
        mpLocalMapper->SetLoopCloser(mpLoopCloser);

        mpLoopCloser->SetTracker(mpTracker);
        mpLoopCloser->SetLocalMapper(mpLocalMapper);

        //Set pointers between threads.
        mpTracker->SetSemanticer(mpSemanticer);
    }
System::System(const string &strVocFile, const string &strSettingsFile,const string &strMapFile, const eSensor sensor,ros::NodeHandle& nh,
               const bool bUseViewer):mSensor(sensor), mpViewer(static_cast<Viewer*>(NULL)),mbReset(false),mbActivateLocalizationMode(false),
        mbDeactivateLocalizationMode(false)
{
    // Output welcome message
    cout << endl <<
    "PlanarSLAM Updated by WangWen" << endl  <<
    "This is free software based on ORB-SLAM2, and you are welcome to redistribute it" << endl <<
    "under certain conditions. See LICENSE.txt." << endl << endl;

    //Check settings file
    cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
       cerr << "Failed to open settings file at: " << strSettingsFile << endl;
       exit(-1);
    }

    // TO DO
    //float resolution = fsSettings["PointCloudMapping.Resolution"];
    //float resolution = 0.01;

    //Load ORB Vocabulary
    cout << endl << "Loading ORB Vocabulary. This could take a while..." << endl;
    clock_t tStart = clock();
    mpVocabulary = new ORBVocabulary();
    bool bVocLoad  = mpVocabulary->loadFromTextFile(strVocFile);
	//else
	 //   bVocLoad = mpVocabulary->loadFromBinaryFile(strVocFile);
    if (!bVocLoad)
    {
        cerr << "Wrong path to vocabulary. " << endl;
        cerr << "Failed to open at: " << strVocFile << endl;
        exit(-1);
    }
    printf("Vocabulary loaded in %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
    cout<<"Vocabulary loaded!"<<endl<<endl;
    //Create KeyFrame Database
    mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);
    //Create the Map
    mpMap = new Map();
    //Create Drawers. These are used by the Viewer
    //这里的帧绘制器和地图绘制器将会被可视化的Viewer所使用
    mpFrameDrawer = new FrameDrawer(mpMap);
    mpMapDrawer = new MapDrawer(mpMap, strSettingsFile);

    //Initialize the Tracking thread
    //(it will live in the main thread of execution, the one that called this constructor)
    // TO DO
    //mpPointCloudMapping = make_shared<PointCloudMapping>( resolution );

    mpTracker = new Tracking(this, mpVocabulary, mpFrameDrawer, mpMapDrawer,
                             mpMap, mpKeyFrameDatabase, strSettingsFile, mSensor,nh);

    //Initialize the Local Mapping thread and launch
    mpLocalMapper = new LocalMapping(mpMap, mSensor==MONOCULAR);
    mptLocalMapping = new thread(&Planar_SLAM::LocalMapping::Run, mpLocalMapper);

    //Initialize the Loop Closing thread and launch
    mpLoopCloser = new LoopClosing(mpMap, mpKeyFrameDatabase, mpVocabulary, mSensor!=MONOCULAR);
    mptLoopClosing = new thread(&Planar_SLAM::LoopClosing::Run, mpLoopCloser);

    // _____________________________yolox_______________________________
    std::string engineFile = "/home/wangwen/Desktop/A_SLAM_Learning/YOLOX-CPP/model_trt.engine";
    mpSemanticer = new YOLOX(engineFile);

    //Initialize the Viewer thread and launch
    if(bUseViewer)
    {
        mpViewer = new Viewer(this, mpFrameDrawer,mpMapDrawer,mpTracker,strSettingsFile);
        mptViewer = new thread(&Viewer::RunWithPLP, mpViewer);
        mpTracker->SetViewer(mpViewer);
    }
    if(bUseViewer)
    {
        mpPangolinViewer = new PangolinViewer(this, 			//又是这个
                                              mpFrameDrawer,	//帧绘制器
                                              mpMapDrawer,		//地图绘制器
                                              mpTracker,		//追踪器
                                              strSettingsFile);	//配置文件的访问路径
        //新建viewer线程
        mptPangolinViewer = new thread(&PangolinViewer::Run, mpPangolinViewer);
        //给运动追踪器设置其查看器
        mpTracker->SetPangolinViewer(mpPangolinViewer);
    }

    // Choose to use pure localization mode
    char IsPureLocalization;
    cout << "Do you want to run pure localization?(y/n)" << endl;
    cin >> IsPureLocalization;
    if(IsPureLocalization == 'Y' || IsPureLocalization == 'y'){
        ActivateLocalizationMode();
    }
    // Load map
    char IsLoadMap;

    //get the current absoulte path
    std::string cwd = getcwd(NULL, 0);
    cout << "The current dir is : " << cwd << endl;
    string strPathSystemSetting = strSettingsFile.c_str();

    cout << "Your setting file path is : " << strPathSystemSetting << endl;

    string strPathMap = strMapFile.c_str();
    cout << "Your map file path would be : " << strPathMap << endl;

    cout << "Do you want to load the map?(y/n)" << endl;
    cin >> IsLoadMap;
    SystemSetting *mySystemSetting = new SystemSetting(mpVocabulary);
    mySystemSetting->LoadSystemSetting(strPathSystemSetting);
    if(IsLoadMap == 'Y' || IsLoadMap == 'y'){
        mpMap->Load(strPathMap, mySystemSetting, mpKeyFrameDatabase);
    }

    //Set pointers between threads
    mpTracker->SetLocalMapper(mpLocalMapper);
    mpTracker->SetLoopClosing(mpLoopCloser);

    mpLocalMapper->SetTracker(mpTracker);
    mpLocalMapper->SetLoopCloser(mpLoopCloser);

    mpLoopCloser->SetTracker(mpTracker);
    mpLoopCloser->SetLocalMapper(mpLocalMapper);

    //Set pointers between threads.
    mpTracker->SetSemanticer(mpSemanticer);

    dep_pub = nh.advertise<sensor_msgs::Image>("/camera/aligned_depth_to_color/image_raw", 100);
    odom_pub = nh.advertise<nav_msgs::Odometry>("/vins_estimator/odometry", 100);
}


cv::Mat System::TrackRGBD(const cv::Mat &im, const cv::Mat &depthmap, const double &timestamp,Eigen::Quaterniond &GroundTruth_R)
{
    if(mSensor!=RGBD)
    {
        cerr << "ERROR: you called TrackRGBD but input sensor was not set to RGBD." << endl;
        exit(-1);
    }    

    // Check mode change
    {
        unique_lock<mutex> lock(mMutexMode);
        if(mbActivateLocalizationMode)
        {
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            while(!mpLocalMapper->isStopped())
            {
                usleep(1000);
            }

            mpTracker->InformOnlyTracking(true);
            mbActivateLocalizationMode = false;
        }
        if(mbDeactivateLocalizationMode)
        {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }

    // Check reset
    {
    unique_lock<mutex> lock(mMutexReset);
    if(mbReset)
    {
        mpTracker->Reset();
        mbReset = false;
    }
    }
    cv::Mat IM;
    cv::Mat Depthmap;
    cv::resize(im,IM,Size(640,480));
    cv::resize(depthmap,Depthmap,Size(640,480));
    cv::Mat Tcw=mpTracker->GrabImageRGBD(IM, Depthmap, timestamp,GroundTruth_R);

    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
    return Tcw;
}

void System::ActivateLocalizationMode()
{
    unique_lock<mutex> lock(mMutexMode);
    mbActivateLocalizationMode = true;
}

void System::DeactivateLocalizationMode()
{
    unique_lock<mutex> lock(mMutexMode);
    mbDeactivateLocalizationMode = true;
}

void System::Reset()
{
    unique_lock<mutex> lock(mMutexReset);
    mbReset = true;
}

void System::Shutdown()
{
    mpLocalMapper->RequestFinish();
    mpLoopCloser->RequestFinish();

//    mpTracker->mpPointCloudMapping->shutdown();
    if(mpViewer)
    {
        mpViewer->RequestFinish();
        while(!mpViewer->isFinished())
            usleep(5000);
        mpPangolinViewer->RequestFinish();
    }

    // Wait until all thread have effectively stopped
    while(!mpLocalMapper->isFinished() || !mpLoopCloser->isFinished()|| mpLoopCloser->isRunningGBA())
    {
        usleep(5000);
    }
    if(mpViewer)
        pangolin::BindToContext("ORB-SLAM2: Map Viewer");
}

void System::SaveTrajectoryTUM(const string &filename)
{
    cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
    if(mSensor==MONOCULAR)
    {
        cerr << "ERROR: SaveTrajectoryTUM cannot be used for monocular." << endl;
        return;
    }

    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    cv::Mat Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    if(!f.is_open())
        cerr << "can't not open file"<<endl;
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<Planar_SLAM::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    list<bool>::iterator lbL = mpTracker->mlbLost.begin();
    for(list<cv::Mat>::iterator lit=mpTracker->mlRelativeFramePoses.begin(),
        lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++, lbL++)
    {
        if(*lbL)
            continue;

        KeyFrame* pKF = *lRit;

        cv::Mat Trw = cv::Mat::eye(4,4,CV_32F);

        // If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
        while(pKF->isBad())
        {
            Trw = Trw*pKF->mTcp;
            pKF = pKF->GetParent();
        }

        Trw = Trw*pKF->GetPose()*Two;

        cv::Mat Tcw = (*lit)*Trw;
        cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t();
        cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);

        vector<float> q = Converter::toQuaternion(Rwc);

        f << setprecision(6) << *lT << " " <<  setprecision(9) << twc.at<float>(0) << " " << twc.at<float>(1) << " " << twc.at<float>(2) << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
    }
    f.close();

    cout << endl << "trajectory saved!" << endl;
}

void System::SaveKeyFrameTrajectoryTUM(const string &filename)
{
    cout << endl << "Saving keyframe trajectory to " << filename << " ..." << endl;

    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    //cv::Mat Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];



       // pKF->SetPose(pKF->GetPose()*Two);

        if(pKF->isBad())
            continue;
        cv::Mat R = pKF->GetRotation().t();
        vector<float> q = Converter::toQuaternion(R);
        cv::Mat t = pKF->GetCameraCenter();
        f << setprecision(6) << pKF->mTimeStamp << setprecision(7) << " " << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2)
          << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;

        //ss<<setprecision(6) <<pKF->mTimeStamp;
        //ss>>filename;
        //cout<<filename<<endl;
        /*
        ofstream Pmatrix("results/"+to_string(pKF->mTimeStamp)+".P");
        cv::Mat R = pKF->GetRotation().t();
        vector<float> q = Converter::toQuaternion(R);
        cv::Mat t = pKF->GetCameraCenter();
        f << setprecision(6) << pKF->mTimeStamp << setprecision(7) << " " << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2)
          << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
        //转换成P矩阵
        float P_00=R.at<float>(0,0)*pKF->fx + R.at<float>(2,0)*pKF->cx;
        float P_01=R.at<float>(0,1)*pKF->fx+R.at<float>( 2,1 )*pKF->cx;
        float P_02=R.at<float>(0,2)*pKF->fx+R.at<float> ( 2,2 )*pKF->cx;
        float P_03= t.at<float>(0) *pKF->fx+ t.at<float>(2) *pKF->cx;

        float P_10=R.at<float>(1,0)*pKF->fy+R.at<float>( 2,0 )*pKF->cy;
        float P_11=R.at<float>(1,1)*pKF->fy+R.at<float>( 2,1 )*pKF->cy;
        float P_12=R.at<float>(1,2)*pKF->fy+R.at<float> ( 2,2 )*pKF->cy;
        float P_13=t.at<float>(1)*pKF->fy+ t.at<float>(2) *pKF->cy;

        float P_20=R.at<float>(2,0);
        float P_21=R.at<float>(2,1);
        float P_22=R.at<float>(2,2);
        float P_23=t.at<float>(2);

        Pmatrix<< P_00<< " "<<P_01<<" "<<P_02<<" "<<P_03<<endl;
        Pmatrix<< P_10<< " "<<P_11<<" "<<P_12<<" "<<P_13<<endl;
        Pmatrix<< P_20<< " "<<P_21<<" "<<P_22<<" "<<P_23<<endl;
        Pmatrix.close();
        */
    }

    f.close();
    cout << endl << "trajectory saved!" << endl;
}

void System::SaveTrajectoryManhattan(const string &filename){
        cout << endl << "Saving Manhattan trajectory to " << filename << " ..." << endl;
        vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
        sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

        // Transform all keyframes so that the first keyframe is at the origin.
        // After a loop closure the first keyframe might not be at the origin.
        cv::Mat Two = vpKFs[0]->GetPoseInverse();

        ofstream f;
        f.open(filename.c_str());
        f << fixed;

        // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
        // which is true when tracking failed (lbL).
        list<Planar_SLAM::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
        list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
        list<bool>::iterator lbL = mpTracker->mlbLost.begin();
        for(list<cv::Mat>::iterator lit=mpTracker->mlRelativeFramePoses.begin(),
                    lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++, lbL++)
        {
            if(*lbL)
                continue;

            KeyFrame* pKF = *lRit;

            cv::Mat Trw = cv::Mat::eye(4,4,CV_32F);

            // If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
            while(pKF->isBad())
            {
                Trw = Trw*pKF->mTcp;
                pKF = pKF->GetParent();
            }

            Trw = Trw*pKF->GetPose()*Two;

            cv::Mat Tcw = (*lit)*Trw;
            cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t();
            cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);

            cv::Mat Rmw = mpTracker->Rotation_cm;
            cv::Mat P_manhattan,Pow;
            Pow = twc;
            P_manhattan = Rmw*Pow;
            // << x,<< z;
            f << P_manhattan.at<float>(0) << " " << P_manhattan.at<float>(2)<< endl;
        }
        f.close();

        cout << endl << "trajectory saved!" << endl;

    }

void System::SaveMesh(const string&filename){
    cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
    mpTracker->SaveMesh(filename);
    cout<< "mesh saved!"<<endl;
}

void System::SaveMap(const string &filename){
    mpMap->Save(filename);
}

void System::Save_OccupancyMap(){
    vector<KeyFrame*> vKFs = mpMap->GetAllKeyFrames();
    for(size_t i=0; i <vKFs.size(); i++  ){
        // publish depth and odometry // camera to world
        cv::Mat Rwc = vKFs[i]->GetPoseInverse().rowRange(0,3).colRange(0,3);
        cv::Mat twc = vKFs[i]->GetPoseInverse().rowRange(0,3).col(3);

        cv::Mat R = cv::Mat::zeros(3,3,CV_32FC1);
        R.at<float>(0,0) = 1;
        R.at<float>(1,2) = 1;
        R.at<float>(2,1) = -1;

        vector<float> q_ = Planar_SLAM::Converter::toQuaternion(R*Rwc);
        Eigen::Matrix<double,3,1> t_ = Planar_SLAM::Converter::toVector3d(R*twc);
        nav_msgs::Odometry odometry;
        odometry.pose.pose.orientation.x = q_[0];
        odometry.pose.pose.orientation.y = q_[1];
        odometry.pose.pose.orientation.z = q_[2];
        odometry.pose.pose.orientation.w = q_[3];

        odometry.pose.pose.position.x = t_.x();
        odometry.pose.pose.position.y = t_.y();
        odometry.pose.pose.position.z = t_.z();
        odometry.header = std_msgs::Header();
        odometry.header.stamp = ros::Time::now();
        cout << "###########################"<< t_.transpose() << endl;

        // image
        cv_bridge::CvImage out_msg;
        cv::Mat depth = vKFs[i]->mImDep;
        out_msg.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
        out_msg.header.stamp = ros::Time::now();
        std_msgs::Header img_heager;
        img_heager.stamp = ros::Time::now();
        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(img_heager,sensor_msgs::image_encodings::TYPE_32FC1,depth).toImageMsg();

        cout << odometry.header.stamp<< "\t"<< out_msg.header.stamp<< endl;
        dep_pub.publish(msg);
        odom_pub.publish(odometry);
        usleep(100000);
    }
}

} //namespace Planar_SLAM
