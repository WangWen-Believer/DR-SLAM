#include "Viewer.h"
#include <pangolin/pangolin.h>
#include <pangolin/scene/axis.h>
#include <pangolin/scene/scenehandler.h>
#include <mutex>

using namespace std;

namespace Planar_SLAM
{
void setImageData(unsigned char * imageArray, int size){
    for(int i = 0 ; i < size;i++) {
        imageArray[i] = (unsigned char)(rand()/(RAND_MAX/255.0));
    }
}

Viewer::Viewer(System* pSystem, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Tracking *pTracking, const string &strSettingPath):
    mpSystem(pSystem), mpFrameDrawer(pFrameDrawer),mpMapDrawer(pMapDrawer), mpTracker(pTracking),
    mbFinishRequested(false), mbFinished(true), mbStopped(false), mbStopRequested(false)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

    float fps = fSettings["Camera.fps"];
    if(fps<1)
        fps=30;
    mT = 1e3/fps;

    mImageWidth = fSettings["Camera.width"];
    mImageHeight = fSettings["Camera.height"];
    if(mImageWidth<1 || mImageHeight<1)
    {
        mImageWidth = 640;
        mImageHeight = 480;
    }

    mViewpointX = fSettings["Viewer.ViewpointX"];
    mViewpointY = fSettings["Viewer.ViewpointY"];
    mViewpointZ = fSettings["Viewer.ViewpointZ"];
    mViewpointF = fSettings["Viewer.ViewpointF"];
}


void Viewer::RunWithPLP()
{
        mbFinished = false;
        mbStopped = false;

        pangolin::CreateWindowAndBind("StructureSLAM: 3D Map", 1024, 768);  //用Pangolin创建显示窗口

        // 3D Mouse handler requires depth testing to be enabled
        // 启动深度测试，OpenGL只绘制最前面的一层，绘制时检查当前像素前面是否有别的像素，如果别的像素挡住了它，那它就不会绘制
        glEnable(GL_DEPTH_TEST);

        // Issue specific OpenGL we might need
        // 在OpenGL中使用颜色混合
        glEnable(GL_BLEND);
         // 选择混合选项
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        pangolin::CreatePanel("menu").SetBounds(0.0, 1.0, 0.0, pangolin::Attach::Pix(175));
        pangolin::Var<bool> menuFollowCamera("menu.Follow Camera", true, true);
        pangolin::Var<bool> menuShowPoints("menu.Show Points", true, true);     //显示特征点
        pangolin::Var<bool> menuShowLines("menu.Show Lines", true, true);       //显示特征线
        pangolin::Var<bool> menuShowPlanes("menu.Show Planes",true,true);
        pangolin::Var<bool> menuShowKeyFrames("menu.Show KeyFrames",true,true);
        pangolin::Var<bool> menuShowGraph("menu.Show Graph",true,true);
        pangolin::Var<bool> menuVideo("menu.Save 2D Frames",false,true);
        pangolin::Var<bool> menuVideo3D("menu.Save Sparse Map",false,true);
        pangolin::Var<bool> menuScreenshot("menu.Screenshot 2D Frame",false, false);
        pangolin::Var<bool> menuScreenshotMesh("menu.Screenshot Mesh",false, false);

        // Define Camera Render Object (for view / scene browsing)
        // 定义相机投影模型：ProjectionMatrix(w, h, fu, fv, u0, v0, zNear, zFar)
        // 定义观测方位向量：观测点位置：(mViewpointX mViewpointY mViewpointZ)
        //                观测目标位置：(0, 0, 0)
        //                观测的方位向量：(0.0,-1.0, 0.0)
        // 相机视角
        pangolin::OpenGlRenderState s_cam(
                pangolin::ProjectionMatrix(1024, 768, mViewpointF, mViewpointF, 512, 389, 0.1, 1000),
                pangolin::ModelViewLookAt(mViewpointX, mViewpointY, mViewpointZ, 0, 0, 0, 0.0, -1.0, 0.0)
        );
        pangolin::OpenGlRenderState s_cam2(
                pangolin::ProjectionMatrix(1024, 768, mViewpointF, mViewpointF, 512, 389, 0.1, 1000),
                pangolin::ModelViewLookAt(mViewpointX, mViewpointY, mViewpointZ, 0, 0, 0, 0.0, -1.0, 0.0)
                );

        pangolin::Renderable tree;
        tree.Add(std::make_shared<pangolin::Axis>());

        // Add named OpenGL viewport to window and provide 3D Handler
        // 定义显示面板大小，orbslam中有左右两个面板，左边显示一些按钮，右边显示图形
        // 前两个参数（0.0, 1.0）表明宽度和面板纵向宽度和窗口大小相同
        // 中间两个参数（pangolin::Attach::Pix(175), 1.0）表明右边所有部分用于显示图形
        // 最后一个参数（-1024.0f/768.0f）为显示长宽比
        pangolin::View& d_cam = pangolin::CreateDisplay()
                .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)
                .SetHandler(new pangolin::Handler3D(s_cam));
        // 窗口
        pangolin::View& d_manhatan = pangolin::CreateDisplay()
                .SetBounds(1/2.0f,1.0f,0,1/2.0f,640.0/480)
                .SetHandler(new pangolin::Handler3D(s_cam2));

        //创建一个欧式变换矩阵,存储当前的相机位姿
        pangolin::OpenGlMatrix Twc;
        Twc.SetIdentity();
        bool bFollow = true;
        bool bLocalizationMode = false;

        int cnt_2d = 0;
        int cnt_3d = 0;
        char buffer[256];
        while(1)
        {

            // 清除缓冲区中的当前可写的颜色缓冲 和 深度缓冲
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            // step1：得到最新的相机位姿
            mpMapDrawer->GetCurrentOpenGLCameraMatrix(Twc);

            // step2：根据相机的位姿调整视角
            // menuFollowCamera为按钮的状态，bFollow为真实的状态
            if(menuFollowCamera && bFollow)
            {
                s_cam.Follow(Twc);
            }
            else if(menuFollowCamera && !bFollow)
            {
                //当之前没有在跟踪相机时
                s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(mViewpointX, mViewpointY, mViewpointZ, 0, 0, 0, 0.0, -1.0, 0.0));
                s_cam.Follow(Twc);
                bFollow = true;
            }
            else if(!menuFollowCamera && bFollow)
            {
                //之前跟踪相机,但是现在菜单命令不要跟踪相机时
                bFollow = false;
            }

            d_cam.Activate(s_cam);
            // step 3：绘制地图和图像(3D部分)
            // 设置为白色，glClearColor(red, green, blue, alpha），数值范围(0, 1)
            glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

            mpMapDrawer->DrawCurrentCamera(Twc);
            //绘制当前相机
            if(menuShowKeyFrames || menuShowGraph)
                //绘制关键帧和共视图
                mpMapDrawer->DrawKeyFrames(menuShowKeyFrames, menuShowGraph);
            if(menuShowPoints)
                //绘制地图点
                mpMapDrawer->DrawMapPoints();
            if(menuShowLines)
                // 绘制地图线
                mpMapDrawer->DrawMapLines();
            if(menuShowPlanes)
                // 绘制地图面
                mpMapDrawer->DrawMapPlanes();

            //显示曼哈顿
            d_manhatan.Activate(s_cam2);
            glColor3f(1.0,1.0,1.0);
//            // 绘制曼哈顿法向量
//            mpMapDrawer->DrawSphere(); // 在窗口内画球 会将菜单栏透射
            mpMapDrawer->DrawMapNormal();
            mpMapDrawer->DrawManhattan();

//            tree.Render();

            pangolin::FinishFrame();

            // step 4:绘制当前帧图像和特征点提取匹配结果
            cv::Mat im = mpFrameDrawer->DrawFrame();
            cv::imshow("Current 2D Frame", im);
            cv::waitKey(mT);

            // step 5 相应其他请求
            //复位按钮
            if(menuScreenshot)
            {
                cv::imwrite("Screenshot.png", im);
                menuScreenshot = false;
            }

            if(menuScreenshotMesh)
            {
                mpTracker->SaveMesh("Screenshot.ply");
                menuScreenshot = false;
            }

            if (menuVideo) {
                // Save video
                sprintf(buffer, "2d/%06d.png", cnt_2d);
                cv::imwrite(buffer, im);
                cnt_2d++;
            }
            if(menuVideo3D)
            {
                sprintf(buffer,"3d/%04d",cnt_3d);
                pangolin::SaveWindowOnRender(buffer);
                cnt_3d++;

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


void Viewer::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool Viewer::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void Viewer::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}

bool Viewer::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

void Viewer::RequestStop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(!mbStopped)
        mbStopRequested = true;
}

bool Viewer::isStopped()
{
    unique_lock<mutex> lock(mMutexStop);
    return mbStopped;
}

bool Viewer::Stop()
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

void Viewer::Release()
{
    unique_lock<mutex> lock(mMutexStop);
    mbStopped = false;
}

} //namespace Planar_SLAM
