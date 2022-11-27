//
// Created by wangwen on 7/13/21.
//
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Imu.h>
#include <std_msgs/Bool.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>

#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<opencv2/core/core.hpp>
#include<System.h>
#include<Mesh.h>
#include<MapPlane.h>
#include<Map.h>
#include "get_char_input.h"
#include<opencv2/core/eigen.hpp>

using namespace Planar_SLAM;


using namespace std;


template <typename T>
T readParam(ros::NodeHandle &n, std::string name)
{
    T ans;
    if (n.getParam(name, ans))
    {
        ROS_INFO_STREAM("Loaded " << name << ": " << ans);
    }
    else
    {
        ROS_ERROR_STREAM("Failed to load " << name);
        n.shutdown();
    }
    return ans;
}

class ImageGrabber
{
public:
    ImageGrabber(Planar_SLAM::System* pSLAM):mpSLAM(pSLAM){
        rootPath = "/home/wangwen/catkin_ws/src/cam_test/data";
        depthPath = rootPath + "/depth/";
        rgbPath = rootPath + "/rgb/";
        associationPath = rootPath +"/association.txt";
        fout_associate.open(associationPath.c_str(),std::ios::out);
    }

    void GrabRGBD(const sensor_msgs::ImageConstPtr& msgRGB,const sensor_msgs::ImageConstPtr& msgD);

    Planar_SLAM::System* mpSLAM;

    std::string rootPath;
    std::string depthPath;
    std::string rgbPath;
    std::string associationPath;
    ofstream fout_associate;

};

int main(int argc, char **argv)
{
    ros::init(argc,argv,"DR_SLAM_ros");
    ros::start();

    ros::NodeHandle n;

    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

    std::string config_file,vocabulary_file,map_file;

    vocabulary_file = readParam<std::string>(n, "/DR_SLAM_ros/vocabulary_file");
    config_file = readParam<std::string>(n, "/DR_SLAM_ros/config_file");
    map_file = readParam<std::string>(n, "/DR_SLAM_ros/map_file");

    Planar_SLAM::System SLAM(vocabulary_file, config_file,map_file, Planar_SLAM::System::RGBD, n,true);

    Planar_SLAM::Config::SetParameterFile(config_file);
    ImageGrabber igb(&SLAM);

    message_filters::Subscriber<sensor_msgs::Image> sub_image(n, "/camera/color/image_raw", 1);
    message_filters::Subscriber<sensor_msgs::Image> sub_depth(n, "/camera/aligned_depth_to_color/image_raw", 1);


    ros::Publisher Save_map_ = n.advertise<std_msgs::Bool>("/save_map_cmd", 10);
//    message_filters::Subscriber<sensor_msgs::Image> sub_image(n, "/cam0/color", 1);
//    message_filters::Subscriber<sensor_msgs::Image> sub_depth(n, "/cam0/depth", 1);

//    message_filters::Subscriber<sensor_msgs::Image> sub_image(n, "/d400/color/image_raw", 1);
//    message_filters::Subscriber<sensor_msgs::Image> sub_depth(n, "/d400/aligned_depth_to_color/image_raw", 1);

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> sync_pol;
    message_filters::Synchronizer<sync_pol> sync(sync_pol(10), sub_image,sub_depth);

    sync.registerCallback(boost::bind(&ImageGrabber::GrabRGBD,&igb,_1,_2));

    while(1)
    {
        char key = kbhit();
        ros::spinOnce();
        if(key == 'w')
        {
            cout << "Saving occupancy map ..."<< endl;
            SLAM.Save_OccupancyMap();
        }
        if(key =='s')
        {
            cout << "Saving navigation map"<< endl;
            std_msgs::Bool save_cmd;
            save_cmd.data = true;
            Save_map_.publish(save_cmd);
            cout << "Saving loaction map ..."<< endl;
            SLAM.SaveMap("/home/wangwen/catkin_Planar/src/PlanarSLAM/MapPointandKeyFrame.bin");
        }
        if(key == 'q')
        {
            cout << "Rmw=\n"<<SLAM.mpTracker->Rotation_cm<<endl;

//            cout << "Saving occupancy map ..."<< endl;
//            SLAM.Save_OccupancyMap();

            // Stop all threads
            SLAM.Shutdown();
            // Save camera trajectory
//            SLAM.SaveTrajectoryTUM(string("/media/wangwen/01D747F7BEB117101/DataSets/Science_Corridor/room_4_full") + string("/CameraTrajectory_DR.txt"));
//            SLAM.SaveTrajectoryManhattan(string("/media/wangwen/01D747F7BEB117101/DataSets/Science_Corridor/room_4_full")+string("/CameraTrajectory_DRManhattan.txt"));
            break;
        }
    }
    ros::shutdown();

    return 0;

}

void ImageGrabber::GrabRGBD(const sensor_msgs::ImageConstPtr& msgRGB,const sensor_msgs::ImageConstPtr& msgD)
{
    // Copy the ros image message to cv::Mat.
    cv_bridge::CvImageConstPtr cv_ptrRGB;

    try
    {
        cv_ptrRGB = cv_bridge::toCvShare(msgRGB);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    cv_bridge::CvImageConstPtr cv_ptrD;
    try
    {
        cv_ptrD = cv_bridge::toCvShare(msgD);
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
        return;
    }

    Eigen::Quaterniond a(1,0,0,0);
    mpSLAM->TrackRGBD(cv_ptrRGB->image,cv_ptrD->image,cv_ptrRGB->header.stamp.toSec(),a);

    stringstream ss;
    ss << cv_ptrRGB->header.stamp;
    std::string tmp;
    ss >> tmp;

    cv::Mat cvColorImgMat=cv_ptrRGB->image;
    cv::Mat cvDepthMat=cv_ptrD->image;

    std::string fileName = tmp + ".png";

    cv::imwrite((rgbPath+fileName),   cvColorImgMat);
    cv::imwrite((depthPath+fileName), cvDepthMat);

    // 时间戳  文件名  时间戳  文件名

    fout_associate << tmp <<" depth/" <<fileName <<" " << tmp <<" rgb/" <<fileName<<"\n";


}

