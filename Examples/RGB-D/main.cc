#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>
#include<opencv2/core/core.hpp>
#include<System.h>
#include<Mesh.h>
#include<MapPlane.h>
#include <vector>
#include "get_char_input.h"
#include<opencv2/core/eigen.hpp>

using namespace std;

void LoadImages(const string &strAssociationFilename,const string &strgroundtruthFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps,vector<Eigen::Quaterniond> &GT_cw,bool bGt);

int main(int argc, char **argv)
{
    if(argc < 5)
    {
        cerr << endl << "Usage: ./Planar_SLAM path_to_vocabulary path_to_settings path_to_sequence path_to_association" << endl;
        return 1;
    }

    // Retrieve paths to images
    vector<string> vstrImageFilenamesRGB;
    vector<string> vstrImageFilenamesD;
    vector<double> vTimestamps;
    vector<Eigen::Quaterniond> vGroundTruths_R;

    string strAssociationFilename = string(argv[4]);
    string strGroundTruthFilename;
    bool bGt = false;
    if(argc==6)
    {
        bGt = true;
        strGroundTruthFilename = string(argv[5]);
    }

    //ICL跟TUM顺序不一样
    cout << "Loading image"<< endl;
    LoadImages(strAssociationFilename,strGroundTruthFilename, vstrImageFilenamesRGB, vstrImageFilenamesD, vTimestamps,vGroundTruths_R,bGt);
//    cout << "wangwen"<< endl;
//     Check consistency in the number of images and depthmaps
    int nImages = vstrImageFilenamesRGB.size();
    if(vstrImageFilenamesRGB.empty())
    {
        cerr << endl << "No images found in provided path." << endl;
        return 1;
    }
    else if(vstrImageFilenamesD.size()!=vstrImageFilenamesRGB.size())
    {
        cerr << endl << "Different number of images for rgb and depth." << endl;
        return 1;
    }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    Planar_SLAM::System SLAM(argv[1], argv[2], Planar_SLAM::System::RGBD, true);
    Planar_SLAM::Config::SetParameterFile(argv[2]);

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    // Feed each image to the system
    cv::Mat imRGB, imD;
//    imRGB = cv::imread(string(argv[3])+"_rgb.png",CV_LOAD_IMAGE_UNCHANGED);
//    imD = cv::imread(string(argv[3])+"_dep.png",CV_LOAD_IMAGE_UNCHANGED);
    double tframe = 1;
    for(int ni=0; ni<nImages; ni++)
    {
        cout<<"PlanarSLAM Printer: This is the "<<ni<<"th image"<<endl;
        // a RGB-D pair
        imRGB = cv::imread(string(argv[3])+"/"+vstrImageFilenamesRGB[ni],CV_LOAD_IMAGE_UNCHANGED);
        imD = cv::imread(string(argv[3])+"/"+vstrImageFilenamesD[ni],CV_LOAD_IMAGE_UNCHANGED);

//        imRGB = cv::imread("/media/wangwen/01D747F7BEB117101/DataSets/Science_Corridor/corridor_full/rgb/1632016307.485407165.png",CV_LOAD_IMAGE_UNCHANGED);
//        imD = cv::imread("/media/wangwen/01D747F7BEB117101/DataSets/Science_Corridor/corridor_full/depth/1632016307.485407165.png",CV_LOAD_IMAGE_UNCHANGED);
//        imRGB = cv::imread("/home/wangwen/Desktop/SLAM_Learning/Image-Rectification/results/test1.jpeg",CV_LOAD_IMAGE_UNCHANGED);

        double tframe = vTimestamps[ni];
        if(imRGB.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(argv[3]) << "/" << vstrImageFilenamesRGB[ni] << endl;
            return 1;
        }

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
#endif
        Eigen::Quaterniond a(1,0,0,0);
//        cv::waitKey(0);
        // Pass the image to the SLAM system
        if(bGt == true)
            SLAM.TrackRGBD(imRGB,imD,tframe,vGroundTruths_R[ni]);
        else
            SLAM.TrackRGBD(imRGB,imD,tframe,a);

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

    }
    char bStop;

    cerr << "PlanarSLAM Printer: Please type 'x', if you want to shutdown windows." << endl;


    while (bStop != 'x'){
        bStop = getchar();
    }

    cout << "Rmw=\n"<<SLAM.mpTracker->Rotation_cm<<endl;

    cv::Mat Last_Position = SLAM.mpTracker->mCurrentFrame.mOw;
    Eigen::Vector3d Last_Pos;
    cv::cv2eigen(Last_Position,Last_Pos);
    // 计算欧式距离
    float dis = Last_Pos.norm();
    cout << "diastance: " << dis << " m"<< endl;
    cout << "Saving map" << endl;
    SLAM.SaveMap("MapPointandKeyFrame.bin");
    // Stop all threads
    SLAM.Shutdown();

    // Save camera trajectory
//    SLAM.SaveTrajectoryTUM(string(argv[3])+"/CameraTrajectory_DR_cape.txt");
//    SLAM.SaveTrajectoryManhattan(string(argv[3])+"/CameraTrajectory_PlannarManhattan.txt");
    return 0;
}

void LoadImages(const string &strAssociationFilename,const string &strgroundtruthFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps,vector<Eigen::Quaterniond> &GT_cw, bool bGt)
{
    ifstream fAssociation;
    ifstream fGroundtruth;
    fAssociation.open(strAssociationFilename.c_str());
    if(bGt == true)
        fGroundtruth.open(strgroundtruthFilename.c_str());

    while(!fAssociation.eof())
    {
        string s;
        getline(fAssociation,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB, sD;
            ss >> t;
            vTimestamps.push_back(t);
            if(1==1)
            {
                ss >> sD;
                vstrImageFilenamesD.push_back(sD);
                ss >> t;
                ss >> sRGB;
                vstrImageFilenamesRGB.push_back(sRGB);
            }
            else
            {
                ss >> sRGB;
                vstrImageFilenamesRGB.push_back(sRGB);
                ss >> t;
                ss >> sD;
                vstrImageFilenamesD.push_back(sD);
            }

        }
    }

    if(bGt == true)
    {
        while(!fGroundtruth.eof())
        {
            string s;
            getline (fGroundtruth,s);
            if(!s.empty())
            {
                stringstream ss;
                string str_id,str_q0,str_q1,str_q2,str_q3,str_t0,str_t1,str_t2;
                double id,q0,q1,q2,q3,t0,t1,t2;
                ss << s;

                ss>>str_id;
                id =  std::stod(str_id);

                ss >> str_t0;
                t0 = stod(str_t0);
                ss >> str_t1;
                t1 = stod(str_t1);
                ss >> str_t2;
                t2 = stod(str_t2);
                // x,y,z,w
                ss >> str_q0;
                q0 = stod(str_q0);
                ss >> str_q1;
                q1 = stod(str_q1);
                ss >> str_q2;
                q2 = stod(str_q2);
                ss >> str_q3;
                q3 = stod(str_q3);

                Eigen::Quaterniond q_wc(q3,q0,q1,q2);
                GT_cw.push_back(q_wc.inverse());
            }

        }
    }
}
