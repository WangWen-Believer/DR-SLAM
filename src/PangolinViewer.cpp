#include "PangolinViewer.h"
#include <pangolin/pangolin.h>
#include <mutex>

namespace Planar_SLAM {
    PangolinViewer::PangolinViewer(System* pSystem, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Tracking *pTracking, const string &strSettingPath):
            mpSystem(pSystem), mpFrameDrawer(pFrameDrawer),mpMapDrawer(pMapDrawer), mpTracker(pTracking),
            mbFinishRequested(false), mbFinished(true), mbStopped(false), mbStopRequested(false)
    {
        //从文件中读取相机的帧频
        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

        float fps = fSettings["Camera.fps"];
        if(fps<1)
            fps=30;
        //计算出每一帧所持续的时间
        mT = 1e3/fps;

        //从配置文件中获取图像的长宽参数
        mImageWidth = fSettings["Camera.width"];
        mImageHeight = fSettings["Camera.height"];
        if(mImageWidth<1 || mImageHeight<1)
        {
            //默认值
            mImageWidth = 640;
            mImageHeight = 480;
        }

        //读取视角
        mViewpointX = fSettings["Viewer.ViewpointX"];
        mViewpointY = fSettings["Viewer.ViewpointY"];
        mViewpointZ = fSettings["Viewer.ViewpointZ"];
        mViewpointF = fSettings["Viewer.ViewpointF"];
//        octomap_char = fSettings["Octomap.filepath"];
        octomap_file = "/home/wangwen/catkin_Planar/src/PlanarSLAM/octomap.ot";
    }

    void PangolinViewer::Run(){
        cout << "####in PangolinViewer run" << endl;
        mbFinished = false;

        pangolin::CreateWindowAndBind("Planar-SLAM: Octomap Viewer",1024,768);

        // 3D Mouse handler requires depth testing to be enabled
        glEnable(GL_DEPTH_TEST);
        // Issue specific OpenGl we might need
        glEnable (GL_BLEND);
        glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        pangolin::CreatePanel("menu_octo").SetBounds(0.0,1.0,0.0,pangolin::Attach::Pix(175));

        pangolin::Var<bool> menuFollowCamera("menu_octo.Follow Camera",true,true);
        pangolin::Var<bool> menuShowKeyFrames("menu_octo.Show KeyFrames",true,true);
        pangolin::Var<bool> menuShowGraph("menu_octo.Show Graph",true,true);
        pangolin::Var<bool> menuLocalizationMode("menu_octo.Localization Mode",false,true);
        pangolin::Var<bool>menuSave("menu_octo.Save", false, false);
        // Define Camera Render Object (for view / scene browsing)
        pangolin::OpenGlRenderState s_cam(
                pangolin::ProjectionMatrix(1024,768,mViewpointF,mViewpointF,512,389,0.1,1000),
                pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,0.0,-1.0, 0.0)
        );
        pangolin::Handler3D* handle =new pangolin::Handler3D(s_cam);
        // Add named OpenGL viewport to window and provide 3D Handler
        pangolin::View& d_cam = pangolin::CreateDisplay()
                .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)
                .SetHandler(handle);

        pangolin::OpenGlMatrix Twc;
        Twc.SetIdentity();

        bool bFollow = true;
        bool bLocalizationMode = false;

        if(menuLocalizationMode == true)
            mpMapDrawer->ReadOctoMap(octomap_file);


        while(1)
        {
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            mpMapDrawer->GetCurrentOpenGLCameraMatrix(Twc);

            if(menuFollowCamera && bFollow)
            {
                s_cam.Follow(Twc);
            }
            else if(menuFollowCamera && !bFollow)
            {
                s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(mViewpointX,mViewpointY,mViewpointZ, 0,0,0,0.0,-1.0, 0.0));
                s_cam.Follow(Twc);
                bFollow = true;
            }
            else if(!menuFollowCamera && bFollow)
            {
                bFollow = false;
            }
            d_cam.Activate(s_cam);
            glClearColor(1.0f,1.0f,1.0f,1.0f);
//#if 1
//            mpMapDrawer->DrawGrid();
            mpMapDrawer->DrawCurrentCamera(Twc);
            if(menuShowKeyFrames || menuShowGraph)
                mpMapDrawer->DrawKeyFrames(menuShowKeyFrames, menuShowGraph);
//
            mpMapDrawer->DrawOctoMap();
//#endif
            pangolin::FinishFrame();
            // note: 先不做
            if(menuSave)
            {
                cout<<"saveing!"<<endl;
//            mpSystem->SaveMap(MAP_PATH);
                mpMapDrawer->SaveOctoMap(octomap_file.c_str());
                menuSave = false;
                cout<<"save done!"<<endl;
            }
            if(Stop())
            {
                while(isStopped())
                {
                    usleep(3000);
                }
            }

            if(CheckFinish())
                break;
        }

        SetFinish();

    }

//外部函数调用,用来请求当前进程结束
    void PangolinViewer::RequestFinish()
    {
        unique_lock<mutex> lock(mMutexFinish);
        mbFinishRequested = true;
    }

//检查是否有结束当前进程的请求
    bool PangolinViewer::CheckFinish()
    {
        unique_lock<mutex> lock(mMutexFinish);
        return mbFinishRequested;
    }

//设置变量:当前进程已经结束
    void PangolinViewer::SetFinish()
    {
        unique_lock<mutex> lock(mMutexFinish);
        mbFinished = true;
    }

//判断当前进程是否已经结束
    bool PangolinViewer::isFinished()
    {
        unique_lock<mutex> lock(mMutexFinish);
        return mbFinished;
    }

//请求当前查看器停止更新
    void PangolinViewer::RequestStop()
    {
        unique_lock<mutex> lock(mMutexStop);
        if(!mbStopped)
            mbStopRequested = true;
    }

//查看当前查看器是否已经停止更新
    bool PangolinViewer::isStopped()
    {
        unique_lock<mutex> lock(mMutexStop);
        return mbStopped;
    }

//当前查看器停止更新
    bool PangolinViewer::Stop()
    {
        unique_lock<mutex> lock(mMutexStop);
        unique_lock<mutex> lock2(mMutexFinish);

        if(mbFinishRequested)
            return false;
        else if(mbStopRequested)
        {
            mbStopped = true;
            mbStopRequested = false;
            return true;
        }

        return false;

    }

//释放查看器进程,因为如果停止查看器的话,查看器进程会处于死循环状态.这个就是为了释放那个标志
    void PangolinViewer::Release()
    {
        unique_lock<mutex> lock(mMutexStop);
        mbStopped = false;
    }
}

