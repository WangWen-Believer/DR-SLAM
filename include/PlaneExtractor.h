#ifndef PLANEDETECTION_H
#define PLANEDETECTION_H

#include <iostream>
#include "opencv2/opencv.hpp"
#include <string>
#include <fstream>
#include <Eigen/Eigen>
#include "include/peac/AHCPlaneFitter.hpp"
#include <unordered_map>

#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/segmentation/organized_multi_plane_segmentation.h>
#include <pcl/features/integral_image_normal.h>


#include <math.h>
#include <Eigen/Dense>
#include "src/CAPE/CAPE.h"

namespace Planar_SLAM {
    typedef Eigen::Vector3d VertexType;

    const int kScaleFactor = 1000;

    const int kDepthWidth = 640;
    const int kDepthHeight = 480;

#ifdef __linux__
#define _isnan(x) isnan(x)
#endif

    struct ImagePointCloud {
        std::vector<VertexType> vertices; // 3D vertices
        int w, h;

        inline int width() const { return w; }

        inline int height() const { return h; }

        inline bool get(const int row, const int col, double &x, double &y, double &z) const {
            const int pixIdx = row * w + col;
            z = vertices[pixIdx][2];
            // Remove points with 0 or invalid depth in case they are detected as a plane
            if (z == 0 || std::_isnan(z)) return false;
            x = vertices[pixIdx][0];
            y = vertices[pixIdx][1];
            return true;
        }
    };

    class PlaneDetection {
    public:
        ImagePointCloud cloud;
        ahc::PlaneFitter<ImagePointCloud> plane_filter;
        std::vector<std::vector<int>> plane_vertices_; // vertex indices each plane contains
        cv::Mat seg_img_; // segmentation image
        cv::Mat_<uchar> seg_output;  //segmentation
        cv::Mat color_img_; // input color image
        int plane_num_;

    public:
        PlaneDetection();

        ~PlaneDetection();

        bool readColorImage(cv::Mat RGBImg);

        bool readDepthImage(cv::Mat depthImg, cv::Mat &K,const float depthfactor);

        void runPlaneDetection();

    };

    class PlaneDetection_CAPE {
    public:
        PlaneDetection_CAPE()=default;
        ~PlaneDetection_CAPE();

        typedef pcl::PointXYZRGB PointT;
        typedef pcl::PointCloud<PointT> PointCloud;

        bool readColorImage(cv::Mat RGBImg);

        bool readDepthImage(cv::Mat depthImg, cv::Mat &K);

        void runPlaneDetection();

    public:
        vector<PointCloud::Ptr> plane_cloud;
        vector<PlaneSeg> plane_params;
        vector<CylinderSeg> cylinder_params;

        int nr_planes, nr_cylinders;
        cv::Mat_<uchar> seg_output;

        cv::Mat color_img_; // input color image
        cv::Mat depth_img;
        cv::Mat K_;

        int PATCH_SIZE;
        float COS_ANGLE_MAX = cos(M_PI / 12);
        float MAX_MERGE_DIST;
        bool cylinder_detection = false;
        CAPE *plane_detector;
    };
}
#endif