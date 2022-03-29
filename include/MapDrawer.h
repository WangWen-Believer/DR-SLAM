/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/Planar_SLAM>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef MAPDRAWER_H
#define MAPDRAWER_H

#include"Map.h"
#include"MapPoint.h"
#include"KeyFrame.h"
#include<pangolin/pangolin.h>

#include<mutex>

// octomap
#include <octomap/ColorOcTree.h>
#include <octomap/Pointcloud.h>
#include <octomap/octomap.h>
#include <octomap/AbstractOcTree.h>
// pcl
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>

namespace Planar_SLAM
{

class MapDrawer
{
public:
    MapDrawer(Map* pMap, const string &strSettingPath);

    Map* mpMap;

    void DrawMapPoints();
    void DrawMapLines();
    void DrawMapPlanes();
    void DrawMapNormal();
    void DrawManhattan();
    void DrawSphere();
    void DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph);
    void DrawCurrentCamera(pangolin::OpenGlMatrix &Twc);
    void SetCurrentCameraPose(const cv::Mat &Tcw);
    void SetReferenceKeyFrame(KeyFrame *pKF);
    void GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M);
    /******************/
    // note: add by wangwen
    bool GetCurrentCameraPos(cv::Mat &Rcw, cv::Mat  &Ow);
    void DrawOctoMap();
    void SaveOctoMap(const char*filename);
    void ReadOctoMap(string filename);
    void DrawObs(pcl::PointCloud<pcl::PointXYZRGB>observation);
protected:
    void GeneratePointCloud(KeyFrame* kf, pcl::PointCloud<pcl::PointXYZRGB>& ground, pcl::PointCloud<pcl::PointXYZRGB>& nonground);
    void InsertScan(octomap::point3d sensorOrigin, pcl::PointCloud<pcl::PointXYZRGB>& ground, pcl::PointCloud<pcl::PointXYZRGB>& nonground);
    bool isSpeckleNode(const octomap::OcTreeKey &nKey);
    void UpdateOctomap(vector<KeyFrame*> vKFs);
    void heightMapColor(double h, double& r, double &g, double& b);
    void LoadOctoMap(const string octofile);
    /********************************/
private:
    float mLineWidth;    //线特征的尺寸
    float mKeyFrameSize;
    float mKeyFrameLineWidth;
    float mGraphLineWidth;
    float mPointSize;
    float mCameraSize;
    float mCameraLineWidth;

    cv::Mat mCameraPose;

    std::mutex mMutexCamera;
    /*********************************/
    //note: add by wangwen
    uint16_t     lastKeyframeSize =0;
    double fx, fy, cx, cy;

    octomap::ColorOcTree *m_octree;
    octomap::KeyRay m_keyRay; // temp storage for casting
    octomap::OcTreeKey m_updateBBXMin;
    octomap::OcTreeKey m_updateBBXMax;

    double m_maxRange;
    bool m_useHeightMap;

    double m_colorFactor;
    double m_res;
    unsigned m_treeDepth;
    unsigned m_maxTreeDepth;
    bool bIsLocalization;

    int nGBA;


    octomap::OcTreeKey m_paddedMinKey, m_paddedMaxKey;
    inline static void updateMinKey(const octomap::OcTreeKey&in, octomap::OcTreeKey& min)
    {
        for(unsigned int i=0; i<3; i++)
            min[i] = std::min(in[i], min[i]);
    }
    inline static void updateMaxKey(const octomap::OcTreeKey&in, octomap::OcTreeKey& max)
    {
        for(unsigned int i=0; i<3; i++)
            max[i] = std::max(in[i], max[i]);
    }
/*********************************/
};

} //namespace ORB_SLAM

#endif // MAPDRAWER_H
