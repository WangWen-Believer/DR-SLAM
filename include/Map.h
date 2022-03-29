#ifndef MAP_H
#define MAP_H

#include "MapPoint.h"
#include "KeyFrame.h"
#include <set>
#include <unordered_map>
#include <tuple>
#include "LSDextractor.h"

#include <mutex>


#include "MapLine.h"
#include "SystemSetting.h"

#include "MapPlane.h"
#include <eigen3/Eigen/Core>
#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/ModelCoefficients.h>

#include "SystemSetting.h"
#include "InitKeyFrame.h"
#include "KeyFrameDatabase.h" //When loading the map, we need to add KeyFrame to KeyFrameDatabase.
#include "Frame.h" // Used for initializing frame.nNextId and mnId
#include "Converter.h"



namespace Planar_SLAM
{

    class MapPoint;
    class KeyFrame;
    class MapLine;
    class MapPlane;
    class KeyFrameDatabase;
    class SystemSetting;
    class Frame;
    struct SurfaceNormal_M{
        std::vector<SurfaceNormal> mvpsurfacenormal;
        std::vector<bool> bsurfacenormal_inline;
    };


    class Map
    {
    public:
        typedef pcl::PointXYZRGB PointT;
        typedef pcl::PointCloud <PointT> PointCloud;

        Map();

        void AddKeyFrame(KeyFrame* pKF);
        void AddMapPoint(MapPoint* pMP);
        void EraseMapPoint(MapPoint* pMP);
        void EraseKeyFrame(KeyFrame* pKF);
        void SetReferenceMapPoints(const std::vector<MapPoint*> &vpMPs);

        void InformNewBigChange();
        int GetLastBigChangeIdx();
        void AddMapLine(MapLine* pML);
        void EraseMapLine(MapLine* pML);
        void SetReferenceMapLines(const std::vector<MapLine*> &vpMLs);
        void SetSurfaceNormal_M(const SurfaceNormal_M &vpSFs);
        void SetRmc(const cv::Mat Rmc);

        std::vector<KeyFrame*> GetAllKeyFrames();
        std::vector<MapPoint*> GetAllMapPoints();
        std::vector<MapPoint*> GetReferenceMapPoints();

        std::vector<MapLine*> GetAllMapLines();
        std::vector<MapLine*> GetReferenceMapLines();
        SurfaceNormal_M GetSurfaceNormal_Manhattan();
        long unsigned int MapLinesInMap();

        long unsigned int MapPointsInMap();
        long unsigned  KeyFramesInMap();

        long unsigned int GetMaxKFid();

        void clear();

        vector<KeyFrame*> mvpKeyFrameOrigins;

        std::mutex mMutexMapUpdate;

        // This avoid that two points are created simultaneously in separate threads (id conflict)
        std::mutex mMutexPointCreation;
        std::mutex mMutexLineCreation;

        void AddMapPlane(MapPlane* pMP);
        void EraseMapPlane(MapPlane *pMP);
        std::vector<MapPlane*> GetAllMapPlanes();
        cv::Mat GetRmc();
        long unsigned int MapPlanesInMap();

        cv::Mat FindManhattan(Frame &pF, const float &verTh, bool out = false);

        void FlagMatchedPlanePoints(Planar_SLAM::Frame &pF, const float &dTh);

        double PointDistanceFromPlane(const cv::Mat& plane, PointCloud::Ptr boundry, bool out = false);

        void Save( const string &filename );

        void Load( const string &filename, SystemSetting* mySystemSetting, KeyFrameDatabase* mpKeyFrameDatabase);
        MapPoint* LoadMapPoint( ifstream &f );
        KeyFrame* LoadKeyFrame( ifstream &f, SystemSetting* mySystemSetting );

    protected:

        void SaveMapPoint( ofstream &f, MapPoint* mp );

        void GetMapPointsIdx();

        // It saves the Index of the MapPoints that matches the ORB featurepoint
        std::map<MapPoint*, unsigned long int> mmpnMapPointsIdx;

        void SaveKeyFrame( ofstream &f, KeyFrame* kf );


        std::set<MapPoint*> mspMapPoints;

        std::set<MapLine*> mspMapLines;

        std::set<MapPlane*> mspMapPlanes;

        std::set<KeyFrame*> mspKeyFrames;

        std::vector<MapPoint*> mvpReferenceMapPoints;
        std::vector<MapLine*> mvpReferenceMapLines;
        SurfaceNormal_M mvpSurfaceNormals_M;
        cv::Mat mRmc;

        long unsigned int mnMaxKFid;

        // Index related to a big change in the map (loop closure, global BA)
        int mnBigChangeIdx;
        std::mutex mMutexMap;
    };

} //namespace Planar_SLAM

#endif // MAP_H
