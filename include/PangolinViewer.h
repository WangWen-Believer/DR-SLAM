#ifndef PANGOLIN_VIEWER_H
#define PANGOLIN_VIEWER_H
#include "FrameDrawer.h"
#include "MapDrawer.h"
#include "Viewer.h"
#include <mutex>

// 重新构建一个线程来进行显示Octomap
namespace Planar_SLAM {
    class Viewer;
    class Tracking;
    class FrameDrawer;
    class MapDrawer;
    class System;

    class PangolinViewer{
    public:
        /**
        * @brief 进程的主函数, NOTICE 注意到这里提到了它是根据相机图像的更新帧率来绘制图像的
        * @detials Main thread function. Draw points, keyframes, the current camera pose and the last processed
        *  frame. Drawing is refreshed according to the camera fps. We use Pangolin.
        */
        PangolinViewer(System* pSystem, FrameDrawer* pFrameDrawer, MapDrawer* pMapDrawer, Tracking *pTracking, const string &strSettingPath);

        void Run();
        /** @brief 请求停止当前进程 */
        void RequestFinish();

        /** @brief 请求当前可视化进程暂停更新图像数据 */
        void RequestStop();
        /** @brief 当前是否有停止当前进程的请求 */
        bool isFinished();
        /** @brief 判断当前进程是否已经停止 */
        bool isStopped();
        /** @brief 释放变量，避免互斥关系 */
        void Release();
        void DrawObs()
        {
            glPointSize(2);
            glBegin(GL_POINTS);


            for(int i=0; i<observation.points.size(); i++)
            {
                glColor3f(observation.points[i].r/255.0, observation.points[i].g/255.0,observation.points[i].b/255.0);
                glVertex3f(observation.points[i].x,observation.points[i].y,observation.points[i].z);
            }

            glEnd();
        };

    private:

        /**
         * @brief 停止当前查看器的更新
         *
         * @return true     成功停止
         * @return false    失败,一般是因为查看器进程已经销毁或者是正在销毁
         */
        bool Stop();

        ///系统对象指针
        System* mpSystem;
        ///帧绘制器
        FrameDrawer* mpFrameDrawer;
        ///地图绘制器
        MapDrawer* mpMapDrawer;
        ///追踪线程句柄
        Tracking* mpTracker;

        // 1/fps in ms
        ///每一帧图像持续的时间
        double mT;
        ///图像的尺寸
        float mImageWidth, mImageHeight;
        ///显示窗口的的查看视角,最后一个是相机的焦距
        float mViewpointX, mViewpointY, mViewpointZ, mViewpointF;
        string octomap_file;
        /**
         * @brief 检查当前查看器进程是否已经终止
         *
         * @return true
         * @return false
         */
        bool CheckFinish();
        /**
         * @brief 设置当前线程终止
         *
         */
        void SetFinish();

        ///请求结束当前线程的标志
        bool mbFinishRequested;
        //当前线程是否已经终止
        bool mbFinished;
        ///线程锁对象,用于锁住和finsh,终止当前查看器进程相关的变量
        //? 但是我现在还是不明白,它是怎么知道我的这个线程锁对象和我的这个线程产生绑定关系的
        std::mutex mMutexFinish;

        ///当前进程是否停止
        bool mbStopped;
        ///是否头停止请求
        bool mbStopRequested;
        ///用于锁住stop,停止更新变量相关的互斥量
        std::mutex mMutexStop;

        pcl::PointCloud<pcl::PointXYZRGB>observation;
    };

}

#endif
