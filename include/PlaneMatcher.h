#ifndef ORB_SLAM2_PLANEMATCHER_H
#define ORB_SLAM2_PLANEMATCHER_H

#include "MapPlane.h"
#include "KeyFrame.h"
#include "Frame.h"
#include <pcl/common/transforms.h>
#include <pcl/point_types.h>

namespace Planar_SLAM {
    class PlaneMatcher {
    public:
        typedef pcl::PointXYZRGB PointT;
        typedef pcl::PointCloud <PointT> PointCloud;

        PlaneMatcher(float dTh = 0.1, float aTh = 0.86, float verTh = 0.08716, float parTh = 0.9962);

        // 平面间的匹配
        int SearchMapByCoefficients(Frame &pF, const std::vector<MapPlane*> &vpMapPlanes, cv::Mat Rwc_MF);
        // 根据匹配的平面之间的夹角确定匹配质量
        bool bMatchStatus(Frame &pF, const std::vector<MapPlane *> &vpMapPlanes, const bool MF_contrast,cv::Mat Rwc_MF);

        int Fuse(KeyFrame *pKF, const std::vector<MapPlane *> &vpMapPlanes);

        int Fuse(KeyFrame* pKF, cv::Mat Scw, const std::vector<MapPlane*> &vpPlanes, const std::vector<MapPlane*> &vpVerticalPlanes,
                const std::vector<MapPlane*> &vpParallelPlanes, vector<MapPlane *> &vpReplacePlane,
                 vector<MapPlane *> &vpReplaceVerticalPlane, vector<MapPlane *> &vpReplaceParallelPlane);

    protected:
        float dTh, aTh, verTh, parTh;

        double PointDistanceFromPlane(const cv::Mat &plane, PointCloud::Ptr pointCloud);
    };
}


#endif //Planar_SLAM