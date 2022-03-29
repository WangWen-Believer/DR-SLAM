#include "FrameDrawer.h"
#include "Tracking.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<mutex>

using namespace std;
using namespace cv;
using namespace cv::line_descriptor;
using namespace Eigen;

namespace Planar_SLAM
{

    FrameDrawer::FrameDrawer(Map* pMap):mpMap(pMap)
    {
        mState=Tracking::SYSTEM_NOT_READY;
        mIm = cv::Mat(480,640,CV_8UC3, cv::Scalar(0,0,0));
        mImage = cv::Mat(480,640,CV_8UC3, cv::Scalar(0,0,0));
        // Populate with random color codes
        for(int i=0; i<10;i++){
            cv::Vec3b color;
            color[0]=rand()%255;
            color[1]=rand()%255;
            color[2]=rand()%255;
            color_code.push_back(color);
        }
        // Add specific colors for planes
        color_code[0][0] = 0; color_code[0][1] = 0; color_code[0][2] = 255;
        color_code[1][0] = 255; color_code[1][1] = 0; color_code[1][2] = 204;
        color_code[2][0] = 255; color_code[2][1] = 100; color_code[2][2] = 0;
        color_code[3][0] = 0; color_code[3][1] = 153; color_code[3][2] = 255;
    }

    cv::Mat FrameDrawer::DrawFrame()
    {
        cv::Mat im;
        cv::Mat image;
        vector<cv::KeyPoint> vIniKeys; // Initialization: KeyPoints in reference frame
        vector<int> vMatches; // Initialization: correspondeces with reference keypoints
        vector<cv::KeyPoint> vCurrentKeys; // KeyPoints in current frame
        vector<bool> vbVO, vbMap; // Tracked MapPoints in current frame
        int state; // Tracking state

        vector<KeyLine> vCurrentKeyLines;
        vector<KeyLine> vIniKeyLines;
        vector<bool> vbLineVO, vbLineMap;

        //Copy variables within scoped mutex
        {
            unique_lock<mutex> lock(mMutex);// 这个对象生命周期结束后自动解锁
            state=mState;
            if(mState==Tracking::SYSTEM_NOT_READY)
                mState=Tracking::NO_IMAGES_YET;

            mIm.copyTo(im);
            mImage.copyTo(image);
            // points and lines for the initialized situation
            if(mState==Tracking::NOT_INITIALIZED)
            {

                vCurrentKeys = mvCurrentKeys;
                vIniKeys = mvIniKeys;
                vMatches = mvIniMatches;
                vCurrentKeyLines = mvCurrentKeyLines;
                vIniKeyLines = mvIniKeyLines;
            }
            // points and lines for the tracking situation
            else if(mState==Tracking::OK)
            {
                vCurrentKeys = mvCurrentKeys;
                vbVO = mvbVO;
                vbMap = mvbMap;
                vCurrentKeyLines = mvCurrentKeyLines;
                vbLineVO = mvbLineVO;
                vbLineMap = mvbLineMap;
            }
            else if(mState==Tracking::LOST)
            {
                vCurrentKeys = mvCurrentKeys;
                vCurrentKeyLines = mvCurrentKeyLines;
            }
        } // destroy scoped mutex -> release mutex

        if(im.channels()<3) //this should be always true
            cvtColor(im,im,CV_GRAY2BGR);
        seg_rz = cv::Mat(480,640,CV_8UC3, cv::Scalar(0,0,0));
        //Draw
        if(state==Tracking::NOT_INITIALIZED) //INITIALIZING
        {
            for(unsigned int i=0; i<vMatches.size(); i++)
            {
                if(vMatches[i]>=0)
                {
                    cv::line(im,vIniKeys[i].pt,vCurrentKeys[vMatches[i]].pt,cv::Scalar(0,255,0));
                }
            }
        }
        else if(state==Tracking::OK) //TRACKING
        {
            mnTracked=0;
            mnTrackedVO=0;
            const float r = 5;
            const int n = vCurrentKeys.size();

            if(1) // visualize 2D points and lines
            {
                //visualize points
                // note  对其进行替换
                /*
                for(int j=0;j<NSNx;j+=1)
                {
                    int u=mvSurfaceNormalx[j].x;
                    int v=mvSurfaceNormalx[j].y;
                    if(u>0&&v>0) {
                        cv::circle(im, cv::Point2f(u, v), 1, cv::Scalar(0, 0, 100), -1);
                    }
                }
                for(int j=0;j<NSNy;j+=1)
                {
                    int u=mvSurfaceNormaly[j].x;
                    int v=mvSurfaceNormaly[j].y;
                    if(u>0&&v>0) {
                        cv::circle(im,cv::Point2f(u,v),1,cv::Scalar(0,100,0),-1);
                    }
                }
                for(int j=0;j<NSNz;j+=1)
                {
                    int u=mvSurfaceNormalz[j].x;
                    int v=mvSurfaceNormalz[j].y;
                    if(u>0&&v>0) {
                        cv::circle(im,cv::Point2f(u,v),1,cv::Scalar(100,0,0),-1);
                    }
                }
*/
                uchar * sCode;
                unsigned char * dColor;
                uchar * srgb;
                int height = im.rows;
                int width = im.cols;
                int code;
                // 平面图像颜色填充
                for(int r=0; r<height; r++){
                    dColor = seg_rz.ptr<unsigned char>(r);
                    sCode = seg_output.ptr<unsigned char>(r);
                    srgb = image.ptr<unsigned char>(r);
                    for(int c=0; c< width; c++){
                        code = *sCode;
//                        if (code>0){//当具备面的属性时
//                            dColor[c*3] =   color_code[code-1][0]/2 + srgb[0]/2;
//                            dColor[c*3+1] = color_code[code-1][1]/2 + srgb[1]/2;
//                            dColor[c*3+2] = color_code[code-1][2]/2 + srgb[2]/2;;
//                        }else{
                            dColor[c*3] =  srgb[0];
                            dColor[c*3+1] = srgb[1];
                            dColor[c*3+2] = srgb[2];
//                        }
                        sCode++; srgb++; srgb++; srgb++;
                    }
                }
                //visualize segmented Manhattan Lines
                // Three colors for three directions
                for(size_t j=0;j<NSLx;j++)
                {
                    int u1 = mvStructLinex[j][2].x; int v1 = mvStructLinex[j][2].y;
                    int u2 = mvStructLinex[j][3].x; int v2 = mvStructLinex[j][3].y;
                    cv::line(seg_rz, cv::Point2f(u1,v1),cv::Point2f(u2,v2), cv::Scalar(255, 0, 255),4);
                }
                for(size_t j=0;j<NSLy;j++)
                {
                    int u1 = mvStructLiney[j][2].x; int v1 = mvStructLiney[j][2].y;
                    int u2 = mvStructLiney[j][3].x; int v2 = mvStructLiney[j][3].y;
                    cv::line(seg_rz, cv::Point2f(u1,v1),cv::Point2f(u2,v2), cv::Scalar(0, 255, 0),4);
                }
                for(size_t j=0;j<NSLz;j++)
                {
                    int u1 = mvStructLinez[j][2].x; int v1 = mvStructLinez[j][2].y;
                    int u2 = mvStructLinez[j][3].x; int v2 = mvStructLinez[j][3].y;
                    cv::line(seg_rz, cv::Point2f(u1,v1),cv::Point2f(u2,v2), cv::Scalar(255, 0, 0),4);
                }
            }
            // 特征点填充
            for(int i=0;i<n;i++)
            {
                if(vbVO[i] || vbMap[i])
                {
                    cv::Point2f pt1,pt2;
                    pt1.x=vCurrentKeys[i].pt.x-r;
                    pt1.y=vCurrentKeys[i].pt.y-r;
                    pt2.x=vCurrentKeys[i].pt.x+r;
                    pt2.y=vCurrentKeys[i].pt.y+r;

                    // This is a match to a MapPoint in the map
                    if(vbMap[i])
                    {
                        cv::rectangle(seg_rz,pt1,pt2,cv::Scalar(155,255,155));
                        cv::circle(seg_rz,vCurrentKeys[i].pt,3,cv::Scalar(155,255,155),-1);
                        mnTracked++;
                    }
                    else // This is match to a "visual odometry" MapPoint created in the last frame
                    {
                        cv::rectangle(seg_rz,pt1,pt2,cv::Scalar(255,0,0));
                        cv::circle(seg_rz,vCurrentKeys[i].pt,2,cv::Scalar(155,255,155),-1);
                        mnTrackedVO++;
                    }
                }
            }

            // 目标框
            DrawObjects(seg_rz,mObjects);
        }
        cv::Mat imWithInfo;
        DrawTextInfo(seg_rz,state, imWithInfo);

        return imWithInfo;
    }


    void FrameDrawer::DrawObjects(cv::Mat image, const std::vector<Object> &objects)
    {
        for (size_t i = 0; i < objects.size(); i++)
        {
            const Object &obj = objects[i];
            if (obj.label == 62)
                continue;

//            fprintf(stderr, "[OUTPUT] Label %s (%d), prob = %.5f at [%.2f,%.2f]; size = %.2f x %.2f\n", class_names[obj.label], obj.label, obj.prob,
//                    obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);
            // std::cout << "nFrames: " << obj.nFrames << "; lostFrames: " << obj.lostFrames << std::endl;

            cv::Scalar color = cv::Scalar(color_list[obj.label][0], color_list[obj.label][1], color_list[obj.label][2]);
            float c_mean = cv::mean(color)[0];
            cv::Scalar txt_color;
            if (c_mean > 0.5)
            {
                txt_color = cv::Scalar(0, 0, 0);
            }
            else
            {
                txt_color = cv::Scalar(255, 255, 255);
            }

            cv::rectangle(image, obj.rect, color * 255, 2);

            char text[256];
            sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

            int baseLine = 0;
            cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

            cv::Scalar txt_bk_color = color * 0.7 * 255;

            int x = obj.rect.x;
            int y = obj.rect.y + 1;
            if (y > image.rows)
                y = image.rows;

            cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                          txt_bk_color, -1);

            cv::putText(image, text, cv::Point(x, y + label_size.height),
                        cv::FONT_HERSHEY_SIMPLEX, 0.4, txt_color, 1);
        }
    }

    void FrameDrawer::DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText)
    {
        stringstream s;
        if(nState==Tracking::NO_IMAGES_YET)
            s << " WAITING FOR IMAGES";
        else if(nState==Tracking::NOT_INITIALIZED)
            s << " TRYING TO INITIALIZE ";
        else if(nState==Tracking::OK)
        {
            if(!mbOnlyTracking)
                s << "SLAM MODE |  ";
            else
                s << "LOCALIZATION | ";
            int nKFs = mpMap->KeyFramesInMap();
            int nMPs = mpMap->MapPointsInMap();
            s << "KFs: " << nKFs << ", MPs: " << nMPs << ", Matches: " << mnTracked;
            if(mnTrackedVO>0)
                s << ", + VO matches: " << mnTrackedVO;
        }
        else if(nState==Tracking::LOST)
        {
            s << " TRACK LOST. TRYING TO RELOCALIZE ";
        }
        else if(nState==Tracking::SYSTEM_NOT_READY)
        {
            s << " LOADING ORB VOCABULARY. PLEASE WAIT...";
        }

        int baseline=0;
        cv::Size textSize = cv::getTextSize(s.str(),cv::FONT_HERSHEY_PLAIN,1,1,&baseline);

        imText = cv::Mat(im.rows+textSize.height+10,im.cols,im.type());
        im.copyTo(imText.rowRange(0,im.rows).colRange(0,im.cols));
        imText.rowRange(im.rows,imText.rows) = cv::Mat::zeros(textSize.height+10,im.cols,im.type());
        cv::putText(imText,s.str(),cv::Point(5,imText.rows-5),cv::FONT_HERSHEY_PLAIN,1,cv::Scalar(255,255,255),1,8);

    }

    void FrameDrawer::Update(Tracking *pTracker)
    {
        unique_lock<mutex> lock(mMutex);
        // 拷贝YOLO目标框
        mObjects = pTracker->mCurrentFrame.current_objects;
        // 拷贝线程跟踪的图像
//        pTracker->mImGray.copyTo(mIm);
        pTracker->mImRGB.copyTo(mImage);
        // 拷贝跟踪线程的特征点
        mvCurrentKeys=pTracker->mCurrentFrame.mvKeys;
        N = mvCurrentKeys.size();
        mvbVO = vector<bool>(N,false);
        mvbMap = vector<bool>(N,false);
        mbOnlyTracking = pTracker->mbOnlyTracking;
        // note 拷贝面特征区域
//        seg_output = pTracker->mCurrentFrame.planeDetectionCape.seg_output;
//        seg_output = pTracker->mCurrentFrame.planeDetector.seg_output;
        seg_output=pTracker->mCurrentFrame.seg_out;
        // 拷贝面的法向量
        mvSurfaceNormalx=pTracker->mCurrentFrame.vSurfaceNormalx;
        mvSurfaceNormaly=pTracker->mCurrentFrame.vSurfaceNormaly;
        mvSurfaceNormalz=pTracker->mCurrentFrame.vSurfaceNormalz;
        NSNx=mvSurfaceNormalx.size();
        NSNy=mvSurfaceNormaly.size();
        NSNz=mvSurfaceNormalz.size();

        mvStructLinex = pTracker->mCurrentFrame.vVanishingLinex;
        mvStructLiney = pTracker->mCurrentFrame.vVanishingLiney;
        mvStructLinez = pTracker->mCurrentFrame.vVanishingLinez;
        NSLx = mvStructLinex.size();
        NSLy = mvStructLiney.size();
        NSLz = mvStructLinez.size();

        mvCurrentKeyLines = pTracker->mCurrentFrame.mvKeylinesUn;
        NL = mvCurrentKeyLines.size();  //自己添加的
        mvbLineVO = vector<bool>(NL, false);
        mvbLineMap = vector<bool>(NL, false);
        //如果上一帧的时候,追踪器没有进行初始化
        if(pTracker->mLastProcessedState==Tracking::NOT_INITIALIZED)
        {
            // 获取初始化帧的特征点、线和匹配信息
            mvIniKeys=pTracker->mInitialFrame.mvKeys;
            mvIniMatches=pTracker->mvIniMatches;
            //线特征的
            mvIniKeyLines = pTracker->mInitialFrame.mvKeylinesUn;
        }
        else if(pTracker->mLastProcessedState==Tracking::OK)
        {
            // 获取当前帧的匹配地图点信息
            for(int i=0;i<N;i++)
            {
                MapPoint* pMP = pTracker->mCurrentFrame.mvpMapPoints[i];
                if(pMP)
                {
                    if(!pTracker->mCurrentFrame.mvbOutlier[i])
                    {
                        //该mappoints可以被多帧观测到，则为有效的地图点
                        if(pMP->Observations()>0)
                            mvbMap[i]=true;
                        else
                            mvbVO[i]=true;//否则表示这个特征点是在当前帧中第一次提取得到的点
                    }
                }
            }
            for(int i=0; i<NL; i++)
            {
                MapLine* pML = pTracker->mCurrentFrame.mvpMapLines[i];
                if(pML)
                {
                    if(!pTracker->mCurrentFrame.mvbLineOutlier[i])
                    {
                        if(pML->Observations()>0)
                            mvbLineMap[i] = true;
                        else
                            mvbLineVO[i] = true;
                    }
                }
            }
        }
        mState=static_cast<int>(pTracker->mLastProcessedState);
    }

} //namespace Planar_SLAM