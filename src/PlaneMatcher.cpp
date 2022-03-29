#include "PlaneMatcher.h"

using namespace std;
using namespace cv;
using namespace Eigen;

namespace Planar_SLAM
{
    PlaneMatcher::PlaneMatcher(float dTh, float aTh, float verTh, float parTh):dTh(dTh), aTh(aTh), verTh(verTh), parTh(parTh) {}
    // 平面匹配  在匹配中引入曼哈顿匹配，这样成功率会上升
    int PlaneMatcher::SearchMapByCoefficients(Frame &pF, const std::vector<MapPlane *> &vpMapPlanes,cv::Mat Rwc_MF) {
        pF.mbNewPlane = false;

        int nmatches = 0;
        //  当前帧分割出来的平面
        for (int i = 0; i < pF.mnPlaneNum; ++i) {
            cv::Mat pM = pF.ComputePlaneWorldCoeff(i);
//            pM = pF.ComputePlaneWorldCoeff_MF(i,Rwc_MF);//////////

            float ldTh =   dTh;
            float lverTh = verTh;
            float lparTh = parTh;

            bool found = false;
            // 遍历地图中的所有plane
            for (auto vpMapPlane : vpMapPlanes) {
                if (vpMapPlane->isBad())
                    continue;
                cv::Mat pW = vpMapPlane->GetWorldPos();
                // To match an observed plane with one from the map
                // We first check the angle between their normals

                // 这个angle 是什么意思 angle = a_2 + b_2 + c_2 ? 不是应该检查他们两个之间法向量的夹角吗
                // 够眼瞎了 一个是PM 一个是PW     a * b = cos theta
                float angle = pM.at<float>(0, 0) * pW.at<float>(0, 0) +
                              pM.at<float>(1, 0) * pW.at<float>(1, 0) +
                              pM.at<float>(2, 0) * pW.at<float>(2, 0);


                // associate plane
                //根据角度和点到平面的距离来进行平面匹配   假设说在正负cos30之内
                if ((angle > aTh || angle < -aTh))
                {
                    // 将分割出来的平面的世界法向量  跟 世界原来的平面上的点云求最小距离
                    // 通过这样来确定两个平面的距离
                    // !!!todo 这个函数有问题，会将两个平面相距很远的当成一个平面
                    // 世界地图中的点到我们检测到这这个平面的距离
                    double dis = PointDistanceFromPlane(pM, vpMapPlane->mvPlanePoints);//点到平面的距离
                    //更换另一种对比方式
                    // 以面与面之间的参数来决定
                    // 假设他们两个平面现在平行 Ax + By + Cz +D1 = 0;Ax + By + Cz +D2 = 0;
                    // 空间中两个平面的距离则为fabs(D1-D2)/sqrt(a^2+b^2+c^2)
//                    double dis = fabs(pM.at<float>(3, 0)- pW.at<float>(3, 0))/
//                            sqrt(pW.at<float>(0, 0)*pW.at<float>(0, 0)+
//                                         pW.at<float>(1, 0)*pW.at<float>(1, 0)+
//                                         pW.at<float>(2, 0)*pW.at<float>(2, 0));
                    if( dis < ldTh) {
                        ldTh = dis;
                        pF.mvpMapPlanes[i] = static_cast<MapPlane*>(nullptr);
                        // 也就是说你所观测到的正式这个平面
                        pF.mvpMapPlanes[i] = vpMapPlane;
                        found = true;
                        continue;
                    }
                }

                // 如何这两个不是同一个平面，那么这两个平面之间在曼哈顿下一定满足以下两个条件：不是垂直就是平行
                // 作者在这里加入到了一个小trick，即使我满足垂直的要求，我也要找跟我最匹配平面。 在这里找最初始符合的面
                //parallel planes
                if (angle > lparTh || angle < -lparTh) {
                    lparTh = abs(angle);
                    pF.mvpParallelPlanes[i] = static_cast<MapPlane*>(nullptr);
                    pF.mvpParallelPlanes[i] = vpMapPlane;
                    continue;
                }
                // vertical planes
                if (angle < lverTh && angle > -lverTh) {
                    lverTh = abs(angle);
                    pF.mvpVerticalPlanes[i] = static_cast<MapPlane*>(nullptr);
                    pF.mvpVerticalPlanes[i] = vpMapPlane;
                    continue;
                }
                // 如果匹配不上，我觉得应该在这里将其设置为外点
            }

            if (found) {
                nmatches++;
            }
        }

        return nmatches;
    }

    bool PlaneMatcher::bMatchStatus(Frame &pF, const std::vector<MapPlane *> &vpMapPlanes, const bool MF_contrast,cv::Mat Rwc_MF) {
        pF.mbNewPlane = false;

        int nmatches = 0;
        if(pF.mnPlaneNum < 2)
            return true;
        //  当前帧分割出来的平面
        for (int i = 0; i < pF.mnPlaneNum; ++i) {

            cv::Mat pM = pF.ComputePlaneWorldCoeff(i);

            // 这里也可以根据曼哈顿来GetWorldPos
            cv::Mat pM_MF;
                pM_MF = pF.ComputePlaneWorldCoeff_MF(i, Rwc_MF);

            float ldTh = dTh;
            float lverTh = verTh;
            float lparTh = parTh;

            bool found = false;
            if(pF.mvpMapPlanes[i])
            {
                MapPlane *vpMapPlane = pF.mvpMapPlanes[i];
                if(!vpMapPlane->isBad())
                {
                    cv::Mat pW = vpMapPlane->GetWorldPos();
                    float angle = pM.at<float>(0, 0) * pW.at<float>(0, 0) +
                                  pM.at<float>(1, 0) * pW.at<float>(1, 0) +
                                  pM.at<float>(2, 0) * pW.at<float>(2, 0);
                    float angle_MF;
                    if (MF_contrast == true)
                        angle_MF = pM_MF.at<float>(0, 0) * pW.at<float>(0, 0) +
                                   pM_MF.at<float>(1, 0) * pW.at<float>(1, 0) +
                                   pM_MF.at<float>(2, 0) * pW.at<float>(2, 0);
                    if(fabs(angle) < fabs(angle_MF)-0.0005&&fabs(angle) > fabs(angle_MF)-0.05)
//                    if(fabs(angle) < fabs(angle_MF))
                    {
                        cout << "说明追踪的R不够靠谱" << endl;
                        // step1: 当前检测到的平面size是否足够大
                        cout << pF.mvPlanePoints[i].size() << endl;
                        cout << "angle angle_MF  " << angle << "," << angle_MF << endl;
                        cout << "angle - angle_MF = " << fabs(angle) - fabs(angle_MF) << endl;
                        return false;
                    }
                }
            }

//            // 遍历地图中的所有plane
//            for (auto vpMapPlane : vpMapPlanes) {
//                if (vpMapPlane->isBad())
//                    continue;
//                cv::Mat pW = vpMapPlane->GetWorldPos();
//                // To match an observed plane with one from the map
//                // We first check the angle between their normals
//
//                // 这个angle 是什么意思 angle = a_2 + b_2 + c_2 ? 不是应该检查他们两个之间法向量的夹角吗
//                // 够眼瞎了 一个是PM 一个是PW     a * b = cos theta
//                float angle = pM.at<float>(0, 0) * pW.at<float>(0, 0) +
//                              pM.at<float>(1, 0) * pW.at<float>(1, 0) +
//                              pM.at<float>(2, 0) * pW.at<float>(2, 0);
//                // MF
//                float angle_MF;
//                if (MF_contrast == true)
//                    angle_MF = pM_MF.at<float>(0, 0) * pW.at<float>(0, 0) +
//                               pM_MF.at<float>(1, 0) * pW.at<float>(1, 0) +
//                               pM_MF.at<float>(2, 0) * pW.at<float>(2, 0);
//
//
//                // associate plane
//                //根据角度和点到平面的距离来进行平面匹配   假设说在正负cos30之内
//                if ((angle > aTh || angle < -aTh)) {
//                    // 将分割出来的平面的世界法向量  跟 世界原来的平面上的点云求最小距离
//                    // 通过这样来确定两个平面的距离
//                    double dis = PointDistanceFromPlane(pM, vpMapPlane->mvPlanePoints);
////                    double dis = fabs(pM.at<float>(3, 0)- pW.at<float>(3, 0))/
////                                 sqrt(pW.at<float>(0, 0)*pW.at<float>(0, 0)+
////                                      pW.at<float>(1, 0)*pW.at<float>(1, 0)+
////                                      pW.at<float>(2, 0)*pW.at<float>(2, 0));
//                    if (dis < ldTh) {
//
//                        /*
//                         * 在这里可以根据这匹配上的这两个平面进行对比，看他们之间的角度大小，他们可以更倾向于拿个
//                         * 其实在距离预知上面就可以进行处理
//                         * */
//                        if (MF_contrast == true && fabs(angle) < fabs(angle_MF) && pF.mvPlanePoints[i].size() > 10) {
//                            cout << "说明追踪的R不够靠谱" << endl;
//                            // step1: 当前检测到的平面size是否足够大
//                            cout << pF.mvPlanePoints[i].size() << endl;
//                            cout << "angle angle_MF  " << angle << "," << angle_MF << endl;
//                            cout << "angle - angle_MF = " << angle - angle_MF << endl;
//                            found = false;
//                            return false;
//                        }
//                        ldTh = dis;
//                        pF.mvpMapPlanes[i] = static_cast<MapPlane *>(nullptr);
//                        // 也就是说你所观测到的正式这个平面
//                        pF.mvpMapPlanes[i] = vpMapPlane;
//                        found = true;
//                        continue;
//                    }
//                }
//            }
        }
        return true;

    }



    double PlaneMatcher::PointDistanceFromPlane(const cv::Mat &plane, PointCloud::Ptr pointCloud) {
        double res = 100;
        double dis_average=0;
        int num=0;
        for(auto p : pointCloud->points){
            // d = Ax0 + By0 +Cz0 + D
            // 这个方程本质是是没有问题，但是在实际测试中会将两个相互平行的但有一定距离的平面当成一个平面。
            // 地图中的点云有一些不可靠
            if(p.z != 0)
            {
                double dis = abs(plane.at<float>(0, 0) * p.x +
                                 plane.at<float>(1, 0) * p.y +
                                 plane.at<float>(2, 0) * p.z +
                                 plane.at<float>(3, 0));
//                dis_average+=dis;
                num++;
                if(dis < res)
                    res = dis;
            }
        }
//        res = dis_average/num;
        return res;
    }
}
