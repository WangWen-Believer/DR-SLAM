#include "Map.h"

#include<mutex>

using namespace std;
using namespace cv;
using namespace cv::line_descriptor;
using namespace Eigen;

namespace Planar_SLAM {

    Map::Map() : mnMaxKFid(0), mnBigChangeIdx(0) {
    }

    void Map::AddKeyFrame(KeyFrame *pKF) {
        unique_lock<mutex> lock(mMutexMap);
        mspKeyFrames.insert(pKF);
        if (pKF->mnId > mnMaxKFid)
            mnMaxKFid = pKF->mnId;
    }

    void Map::AddMapPoint(MapPoint *pMP) {
        unique_lock<mutex> lock(mMutexMap);
        mspMapPoints.insert(pMP);
    }

    void Map::EraseMapPoint(MapPoint *pMP) {
        unique_lock<mutex> lock(mMutexMap);
        mspMapPoints.erase(pMP);

        // TODO: This only erase the pointer.
        // Delete the MapPoint
    }

    void Map::EraseKeyFrame(KeyFrame *pKF) {
        unique_lock<mutex> lock(mMutexMap);
        mspKeyFrames.erase(pKF);

        // TODO: This only erase the pointer.
        // Delete the MapPoint
    }

    void Map::SetReferenceMapPoints(const vector<MapPoint *> &vpMPs) {
        unique_lock<mutex> lock(mMutexMap);
        mvpReferenceMapPoints = vpMPs;
    }

    void Map::InformNewBigChange() {
        unique_lock<mutex> lock(mMutexMap);
        mnBigChangeIdx++;
    }

    int Map::GetLastBigChangeIdx() {
        unique_lock<mutex> lock(mMutexMap);
        return mnBigChangeIdx;
    }

    vector<KeyFrame *> Map::GetAllKeyFrames() {
        unique_lock<mutex> lock(mMutexMap);
        return vector<KeyFrame *>(mspKeyFrames.begin(), mspKeyFrames.end());
    }

    vector<MapPoint *> Map::GetAllMapPoints() {
        unique_lock<mutex> lock(mMutexMap);
        return vector<MapPoint *>(mspMapPoints.begin(), mspMapPoints.end());
    }

    long unsigned int Map::MapPointsInMap() {
        unique_lock<mutex> lock(mMutexMap);
        return mspMapPoints.size();
    }

    long unsigned int Map::KeyFramesInMap() {
        unique_lock<mutex> lock(mMutexMap);
        return mspKeyFrames.size();
    }

    vector<MapPoint *> Map::GetReferenceMapPoints() {
        unique_lock<mutex> lock(mMutexMap);
        return mvpReferenceMapPoints;
    }

    long unsigned int Map::GetMaxKFid() {
        unique_lock<mutex> lock(mMutexMap);
        return mnMaxKFid;
    }

    void Map::clear() {
        for (auto mspMapPoint : mspMapPoints)
            delete mspMapPoint;
        for (auto mspMapLine : mspMapLines)
            delete mspMapLine;
        for (auto mspMapPlane : mspMapPlanes)
            delete mspMapPlane;

        for (auto mspKeyFrame : mspKeyFrames)
            delete mspKeyFrame;

        mspMapPlanes.clear();
        mspMapPoints.clear();
        mspKeyFrames.clear();
        mspMapLines.clear();
        mnMaxKFid = 0;
        mvpReferenceMapPoints.clear();
        mvpReferenceMapLines.clear();
        mvpKeyFrameOrigins.clear();
    }

    void Map::AddMapLine(MapLine *pML) {
        unique_lock<mutex> lock(mMutexMap);
        mspMapLines.insert(pML);
    }

    void Map::EraseMapLine(MapLine *pML) {
        unique_lock<mutex> lock(mMutexMap);
        mspMapLines.erase(pML);
    }

    //
    void Map::SetReferenceMapLines(const std::vector<MapLine *> &vpMLs) {
        unique_lock<mutex> lock(mMutexMap);
        mvpReferenceMapLines = vpMLs;
    }
    void Map::SetSurfaceNormal_M(const SurfaceNormal_M &vpSFs){
        unique_lock<mutex> lock(mMutexMap);
        mvpSurfaceNormals_M = vpSFs;
    }
    void Map::SetRmc(const cv::Mat Rmc){
        unique_lock<mutex> lock(mMutexMap);
        Rmc.copyTo(mRmc);
    }

    vector<MapLine *> Map::GetAllMapLines() {
        unique_lock<mutex> lock(mMutexMap);
        return vector<MapLine *>(mspMapLines.begin(), mspMapLines.end());
    }

    vector<MapLine *> Map::GetReferenceMapLines() {
        unique_lock<mutex> lock(mMutexMap);
        return mvpReferenceMapLines;
    }
    SurfaceNormal_M Map::GetSurfaceNormal_Manhattan(){
        unique_lock<mutex> lock(mMutexMap);
        return mvpSurfaceNormals_M;
    }

    long unsigned int Map::MapLinesInMap() {
        unique_lock<mutex> lock(mMutexMap);
        return mspMapLines.size();
    }

    void Map::AddMapPlane(MapPlane *pMP) {
        unique_lock<mutex> lock(mMutexMap);
        mspMapPlanes.insert(pMP);
    }

    void Map::EraseMapPlane(MapPlane *pMP) {
        unique_lock<mutex> lock(mMutexMap);
        mspMapPlanes.erase(pMP);
    }

    vector<MapPlane *> Map::GetAllMapPlanes() {
        unique_lock<mutex> lock(mMutexMap);
        return vector<MapPlane *>(mspMapPlanes.begin(), mspMapPlanes.end());
    }
    cv::Mat Map::GetRmc()
    {
        unique_lock<mutex> lock(mMutexMap);
        return mRmc.clone();
    }

    long unsigned int Map::MapPlanesInMap() {
        unique_lock<mutex> lock(mMutexMap);
        return mspMapPlanes.size();
    }

    // verTh -- 角度阈值
    cv::Mat Map::FindManhattan(Frame &pF, const float &verTh, bool out) {
        cv::Mat bestP1, bestP2;
        float lverTh = verTh;
        int maxSize = 0;

        if(out)
            cout << "Matching planes..." << endl;

        // 关键帧中检测到的平面特征个数
        for (int i = 0; i < pF.mnPlaneNum; ++i) {
            // 获取平面参数
            cv::Mat p1 = pF.mvPlaneCoefficients[i];
            if(out)
                cout << " plane  " << i << ": " << endl;

            if(out)
                cout << " p1  " << p1.t() << ": " << endl;

            // 对当前关键帧中检测到的平面特征之间进行关联
            for (int j = i+1;j < pF.mnPlaneNum; ++j) {
                cv::Mat p2 = pF.mvPlaneCoefficients[j];

                // 计算两平面间的夹角
                // 法向量的角度cos theta=两个向量之间的点乘
                float angle = p1.at<float>(0) * p2.at<float>(0) +
                              p1.at<float>(1) * p2.at<float>(1) +
                              p1.at<float>(2) * p2.at<float>(2);

                if(out)
                    cout << j << ", p2 : " << p2.t() << endl;

                if(out)
                    cout << j << ", angle : " << angle << endl;

                // vertical planes
                // 同时找两个最大的平面进行匹配求得这两个带有垂直属性的夹角
                // 得到BestP1 and BestP2
                if (angle < lverTh && angle > -lverTh && (pF.mvPlanePoints[i].size() + pF.mvPlanePoints[j].size()) > maxSize) {
                    if(out)
                        cout << "  vertical!" << endl;
                    maxSize = pF.mvPlanePoints[i].size() + pF.mvPlanePoints[j].size();

                    if (bestP1.empty() || bestP2.empty()) {
                        bestP1 = cv::Mat::eye(cv::Size(1, 3), CV_32F);
                        bestP2 = cv::Mat::eye(cv::Size(1, 3), CV_32F);
                    }

                    bestP1.at<float>(0, 0) = p1.at<float>(0, 0);
                    bestP1.at<float>(1, 0) = p1.at<float>(1, 0);
                    bestP1.at<float>(2, 0) = p1.at<float>(2, 0);

                    bestP2.at<float>(0, 0) = p2.at<float>(0, 0);
                    bestP2.at<float>(1, 0) = p2.at<float>(1, 0);
                    bestP2.at<float>(2, 0) = p2.at<float>(2, 0);
                }
            }
        }

        // 未找到面与面之间的最佳匹配，进而转向面和线之间的最佳匹配
        if (bestP1.empty() || bestP2.empty()) {
            if(out)
                cout << "Matching planes and lines..." << endl;

            for (int i = 0; i < pF.mnPlaneNum; ++i) {
                // 获取平面参数
                cv::Mat p = pF.ComputePlaneWorldCoeff(i);
                if(out)
                    cout << " plane  " << i << ": " << endl;

                // 获取线参数  start_point and end_point
                // 面跟所有线进行匹配  这里有个问题   他是如何找到最好的bestP  --在line262 得到解决
                for (int j = 0; j < pF.mvLines3D.size(); ++j) {
                    Vector6d lineVector = pF.obtain3DLine(j);

                    cv::Mat startPoint = cv::Mat::eye(cv::Size(1, 3), CV_32F);
                    cv::Mat endPoint = cv::Mat::eye(cv::Size(1, 3), CV_32F);

                    startPoint.at<float>(0, 0) = lineVector[0];
                    startPoint.at<float>(1, 0) = lineVector[1];
                    startPoint.at<float>(2, 0) = lineVector[2];
                    endPoint.at<float>(0, 0) = lineVector[3];
                    endPoint.at<float>(1, 0) = lineVector[4];
                    endPoint.at<float>(2, 0) = lineVector[5];

                    cv::Mat line = startPoint - endPoint;
                    line /= cv::norm(line);

                    if(out)
                        cout << "line: " << line << endl;
                    // 获取点线之间的夹角
                    // 平面的法向量 直线的方向向量   最后求两向量所成角的余弦
                    float angle = p.at<float>(0, 0) * line.at<float>(0, 0) +
                                  p.at<float>(1, 0) * line.at<float>(1, 0) +
                                  p.at<float>(2, 0) * line.at<float>(2, 0);

                    if(out)
                        cout << j << ", angle : " << angle << endl;

                    if (angle < lverTh && angle > -lverTh) {
                        if(out)
                            cout << "  vertical!" << endl;
                        // 在这里更新角度阈值，来find best P
                        lverTh = abs(angle);

                        if (bestP1.empty() || bestP2.empty()) {
                            bestP1 = cv::Mat::eye(cv::Size(1, 3), CV_32F);
                            bestP2 = cv::Mat::eye(cv::Size(1, 3), CV_32F);
                        }

                        bestP1.at<float>(0, 0) = p.at<float>(0, 0);
                        bestP1.at<float>(1, 0) = p.at<float>(1, 0);
                        bestP1.at<float>(2, 0) = p.at<float>(2, 0);

                        bestP2.at<float>(0, 0) = line.at<float>(0, 0);
                        bestP2.at<float>(1, 0) = line.at<float>(1, 0);
                        bestP2.at<float>(2, 0) = line.at<float>(2, 0);
                    }
                }
            }
        }

        if(out)
            cout << "Matching done" << endl;

        cv::Mat Rotation_cm;
        Rotation_cm = cv::Mat::eye(cv::Size(3, 3), CV_32F);

        // 根据bestP  根据曼哈顿计算出旋转
        if (!bestP1.empty() && !bestP2.empty()) {

            int loc1;
            float max1 = 0;
            for (int i = 0; i < 3; i++) {
                // 获取面的法向量
                float val = bestP1.at<float>(i);
                // 寻找法向量分量最大的矢量
                if (val < 0)
                    val = -val;
                if (val > max1) {
                    loc1 = i;
                    max1 = val;
                }
            }

            if (bestP1.at<float>(loc1) < 0) {
                bestP1 = -bestP1;
            }
            // same with P1
            int loc2;
            float max2 = 0;
            for (int i = 0; i < 3; i++) {
                float val = bestP2.at<float>(i);
                if (val < 0)
                    val = -val;
                if (val > max2) {
                    loc2 = i;
                    max2 = val;
                }
            }

            if (bestP2.at<float>(loc2) < 0) {
                bestP2 = -bestP2;
            }

            cv::Mat p3;
            // 向量的叉乘
            p3 = bestP1.cross(bestP2);
            // same with P1 and P2
            int loc3;
            float max3 = 0;
            for (int i = 0; i < 3; i++) {
                float val = p3.at<float>(i);
                if (val < 0)
                    val = -val;
                if (val > max3) {
                    loc3 = i;
                    max3 = val;
                }
            }

            if (p3.at<float>(loc3) < 0) {
                p3 = -p3;
            }

            // p1 p2 p3   构成了一个坐标系
            if(out) {
                cout << "p1: " << bestP1 << endl;
                cout << "p2: " << bestP2 << endl;
                cout << "p3: " << p3 << endl;
            }

            cv::Mat first, second, third;

            std::map<int, cv::Mat> sort;
            sort[loc1] = bestP1;
            sort[loc2] = bestP2;
            sort[loc3] = p3;

            first = sort[0];
            second = sort[1];
            third = sort[2];

            // todo: refine this part
            // 有可能是反射矩阵
            // 将三个方向的矢量建立起的坐标系转化成旋转的形式
            Rotation_cm.at<float>(0, 0) = first.at<float>(0, 0);
            Rotation_cm.at<float>(1, 0) = first.at<float>(1, 0);
            Rotation_cm.at<float>(2, 0) = first.at<float>(2, 0);
            Rotation_cm.at<float>(0, 1) = second.at<float>(0, 0);
            Rotation_cm.at<float>(1, 1) = second.at<float>(1, 0);
            Rotation_cm.at<float>(2, 1) = second.at<float>(2, 0);
            Rotation_cm.at<float>(0, 2) = third.at<float>(0, 0);
            Rotation_cm.at<float>(1, 2) = third.at<float>(1, 0);
            Rotation_cm.at<float>(2, 2) = third.at<float>(2, 0);

            cv::Mat U, W, VT;

            // 进行SVD分解获得旋转矩阵
            // 在这里进行SVD分解的目的是我们根据面面匹配或线面匹配得到的旋转矩阵并不能完全满足我们
            // 旋转矩阵的性质，这时候我们通过SVD分解的形式获取较为理想的旋转矩阵
            cv::SVD::compute(Rotation_cm, W, U, VT);
            Rotation_cm = U * VT;
        }

        // mahattan to camera
        return Rotation_cm;
    }

    void Map::FlagMatchedPlanePoints(Planar_SLAM::Frame &pF, const float &dTh) {

        unique_lock<mutex> lock(mMutexMap);
        int nMatches = 0;

        for (int i = 0; i < pF.mnPlaneNum; ++i) {

            cv::Mat pM = pF.ComputePlaneWorldCoeff(i);

            if (pF.mvpMapPlanes[i]) {
                for (auto mapPoint : mspMapPoints) {
                    cv::Mat pW = mapPoint->GetWorldPos();

                    double dis = abs(pM.at<float>(0, 0) * pW.at<float>(0, 0) +
                                     pM.at<float>(1, 0) * pW.at<float>(1, 0) +
                                     pM.at<float>(2, 0) * pW.at<float>(2, 0) +
                                     pM.at<float>(3, 0));

                    if (dis < 0.5) {
                        mapPoint->SetAssociatedWithPlaneFlag(true);
                        nMatches++;
                    }
                }
            }
        }
    }

    double Map::PointDistanceFromPlane(const cv::Mat &plane, PointCloud::Ptr boundry, bool out) {
        double res = 100;
        if (out)
            cout << " compute dis: " << endl;
        for (auto p : boundry->points) {
            double dis = abs(plane.at<float>(0, 0) * p.x +
                             plane.at<float>(1, 0) * p.y +
                             plane.at<float>(2, 0) * p.z +
                             plane.at<float>(3, 0));
            if (dis < res)
                res = dis;
        }
        if (out)
            cout << endl << "ave : " << res << endl;
        return res;
    }

    void Map::Save ( const string& filename )
    {
        //Print the information of the saving map
        cerr<<"Map.cc :: Map Saving to "<<filename <<endl;
        ofstream f;
        f.open(filename.c_str(), ios_base::out|ios::binary);
        //Number of MapPoints
        unsigned long int nMapPoints = mspMapPoints.size();
        f.write((char*)&nMapPoints, sizeof(nMapPoints) );
        //Save MapPoint sequentially
        for ( auto mp: mspMapPoints ){
            //Save MapPoint
            SaveMapPoint( f, mp );
            // cerr << "Map.cc :: Saving map point number: " << mp->mnId << endl;
        }

        //Print The number of MapPoints
        cerr << "Map.cc :: The number of MapPoints is :"<<mspMapPoints.size()<<endl;


        //Grab the index of each MapPoint, count from 0, in which we initialized mmpnMapPointsIdx
        GetMapPointsIdx();

        //Print the number of KeyFrames
        cerr <<"Map.cc :: The number of KeyFrames:"<<mspKeyFrames.size()<<endl;
        //Number of KeyFrames
        unsigned long int nKeyFrames = mspKeyFrames.size();
        f.write((char*)&nKeyFrames, sizeof(nKeyFrames));

        //Save KeyFrames sequentially
        for ( auto kf: mspKeyFrames )
            SaveKeyFrame( f, kf );

        for (auto kf:mspKeyFrames )
        {
            //Get parent of current KeyFrame and save the ID of this parent
            KeyFrame* parent = kf->GetParent();
            unsigned long int parent_id = ULONG_MAX;
            if ( parent )
                parent_id = parent->mnId;
            f.write((char*)&parent_id, sizeof(parent_id));

            //Get the size of the Connected KeyFrames of the current KeyFrames
            //and then save the ID and weight of the Connected KeyFrames
            unsigned long int nb_con = kf->GetConnectedKeyFrames().size();
            f.write((char*)&nb_con, sizeof(nb_con));
            for ( auto ckf: kf->GetConnectedKeyFrames())
            {
                int weight = kf->GetWeight(ckf);
                f.write((char*)&ckf->mnId, sizeof(ckf->mnId));
                f.write((char*)&weight, sizeof(weight));
            }
        }

        // Save last Frame ID
        // SaveFrameID(f);

        f.close();
        cerr<<"Map.cc :: Map Saving Finished!"<<endl;
    }

    // Load map from file
    void Map::Load ( const string &filename, SystemSetting* mySystemSetting, KeyFrameDatabase* mpKeyFrameDatabase )
    {
        cerr << "Map.cc :: Map reading from:"<<filename<<endl;
        ifstream f;
        f.open( filename.c_str() );

        // Same as the sequence that we save the file, we first read the number of MapPoints.
        unsigned long int nMapPoints;
        f.read((char*)&nMapPoints, sizeof(nMapPoints));

        // Then read MapPoints one after another, and add them into the map
        cerr<<"Map.cc :: The number of MapPoints:"<<nMapPoints<<endl;
        for ( unsigned int i = 0; i < nMapPoints; i ++ )
        {
            MapPoint* mp = LoadMapPoint(f);
            AddMapPoint(mp);
        }

        // Get all MapPoints
        std::vector<MapPoint*> vmp = GetAllMapPoints();

        // Read the number of KeyFrames
        unsigned long int nKeyFrames;
        f.read((char*)&nKeyFrames, sizeof(nKeyFrames));
        cerr<<"Map.cc :: The number of KeyFrames:"<<nKeyFrames<<endl;

        // Then read KeyFrames one after another, and add them into the map
        vector<KeyFrame*>kf_by_order;
        for( unsigned int i = 0; i < nKeyFrames; i ++ )
        {
            KeyFrame* kf = LoadKeyFrame(f, mySystemSetting);
            AddKeyFrame(kf);
            kf_by_order.push_back(kf);
            mpKeyFrameDatabase->add(kf);
        }

        if(mnMaxKFid>0){
            Frame temp_frame = Frame( mnMaxKFid );
        }

        cerr<<"Map.cc :: Max KeyFrame ID is: " << mnMaxKFid << ", and I set mnId to this number" <<endl;


        cerr<<"Map.cc :: KeyFrame Load OVER!"<<endl;

        // Read Spanning Tree(open loop trajectory)
        map<unsigned long int, KeyFrame*> kf_by_id;
        for ( auto kf: mspKeyFrames )
            kf_by_id[kf->mnId] = kf;
        cerr<<"Map.cc :: Start Load The Parent!"<<endl;
        for( auto kf: kf_by_order )
        {
            // Read parent_id of current KeyFrame.
            unsigned long int parent_id;
            f.read((char*)&parent_id, sizeof(parent_id));

            // Add parent KeyFrame to current KeyFrame.
            // cout<<"Map::Load : Add parent KeyFrame to current KeyFrame"<<endl;
            if ( parent_id != ULONG_MAX )
                kf->ChangeParent(kf_by_id[parent_id]);

            // Read covisibility graphs.
            // Read the number of Connected KeyFrames of current KeyFrame.
            unsigned long int nb_con;
            f.read((char*)&nb_con, sizeof(nb_con));
            // Read id and weight of Connected KeyFrames of current KeyFrame,
            // and add Connected KeyFrames into covisibility graph.
            // cout<<"Map::Load : Read id and weight of Connected KeyFrames"<<endl;
            for ( unsigned long int i = 0; i < nb_con; i ++ )
            {
                unsigned long int id;
                int weight;
                f.read((char*)&id, sizeof(id));
                f.read((char*)&weight, sizeof(weight));
                kf->AddConnection(kf_by_id[id],weight);
            }
        }
        cerr<<"Map.cc :: Parent Load OVER!"<<endl;
        for ( auto mp: vmp )
        {
            // cout << "Now mp = "<< mp << endl;
            if(mp)
            {
                // cout << "compute for mp = "<< mp << endl;
                mp->ComputeDistinctiveDescriptors();
                // cout << "Computed Distinctive Descriptors." << endl;
                mp->UpdateNormalAndDepth();
                // cout << "Updated Normal And Depth." << endl;
            }
        }
        f.close();
        cerr<<"Map.cc :: Load IS OVER!"<<endl;
        return;
    }

    MapPoint* Map::LoadMapPoint( ifstream &f )
    {
        // Position and Orientation of the MapPoints.
        cv::Mat Position(3,1,CV_32F);
        long unsigned int id;
        f.read((char*)&id, sizeof(id));

        f.read((char*)&Position.at<float>(0), sizeof(float));
        f.read((char*)&Position.at<float>(1), sizeof(float));
        f.read((char*)&Position.at<float>(2), sizeof(float));

        // Initialize a MapPoint, and set its id and Position.
        MapPoint* mp = new MapPoint(Position, this );
        mp->mnId = id;
        mp->SetWorldPos( Position );

        return mp;
    }


    KeyFrame* Map::LoadKeyFrame( ifstream &f, SystemSetting* mySystemSetting )
    {
        // Since we need to initialize a lot of informatio about KeyFrame,
        // let's define a new class named InitKeyFrame.
        // It initializes with SystemSetting,
        // which helps to read the configuration files(camera amtrix, ORB features, etc.)
        // We'll create "SystemSetting.cc" and "InitKeyFrame.cc"
        // and their header files in "src" and "include" folders.


        // Declare initkf to initialize Key Frames.
        InitKeyFrame initkf(*mySystemSetting);

        // Read ID and TimeStamp of each KeyFrame.
        f.read((char*)&initkf.nId, sizeof(initkf.nId));
        f.read((char*)&initkf.TimeStamp, sizeof(double));

        // Read position and quaternion
        cv::Mat T = cv::Mat::zeros(4,4,CV_32F);
        std::vector<float> Quat(4);
        //Quat.reserve(4);
        for ( int i = 0; i < 4; i ++ )
            f.read((char*)&Quat[i],sizeof(float));
        cv::Mat R = Converter::toCvMat(Quat);
        for ( int i = 0; i < 3; i ++ )
            f.read((char*)&T.at<float>(i,3),sizeof(float));
        for ( int i = 0; i < 3; i ++ )
            for ( int j = 0; j < 3; j ++ )
                T.at<float>(i,j) = R.at<float>(i,j);
        T.at<float>(3,3) = 1;

//    for ( int i = 0; i < 4; i ++ )
//    {
//      for ( int j = 0; j < 4; j ++ )
//      {
//              f.read((char*)&T.at<float>(i,j), sizeof(float));
//              cerr<<"T.at<float>("<<i<<","<<j<<"):"<<T.at<float>(i,j)<<endl;
//      }
//    }
//
        // Read feature point number of current Key Frame
        f.read((char*)&initkf.N, sizeof(initkf.N));
        initkf.vKps.reserve(initkf.N);
        initkf.Descriptors.create(initkf.N, 32, CV_8UC1);
        vector<float>KeypointDepth;

        std::vector<MapPoint*> vpMapPoints;
        vpMapPoints = vector<MapPoint*>(initkf.N,static_cast<MapPoint*>(NULL));
        // Read Keypoints and descriptors of current KeyFrame
        std::vector<MapPoint*> vmp = GetAllMapPoints();
        for(int i = 0; i < initkf.N; i ++ )
        {
            cv::KeyPoint kp;
            f.read((char*)&kp.pt.x, sizeof(kp.pt.x));
            f.read((char*)&kp.pt.y, sizeof(kp.pt.y));
            f.read((char*)&kp.size, sizeof(kp.size));
            f.read((char*)&kp.angle,sizeof(kp.angle));
            f.read((char*)&kp.response, sizeof(kp.response));
            f.read((char*)&kp.octave, sizeof(kp.octave));

            initkf.vKps.push_back(kp);

            // Read depth value of keypoint.
            //float fDepthValue = 0.0;
            //f.read((char*)&fDepthValue, sizeof(float));
            //KeypointDepth.push_back(fDepthValue);


            // Read descriptors of keypoints
            f.read((char*)&initkf.Descriptors.cols, sizeof(initkf.Descriptors.cols));
            // for ( int j = 0; j < 32; j ++ ) // Since initkf.Descriptors.cols is always 32, for loop may also write like this.
            for ( int j = 0; j < initkf.Descriptors.cols; j ++ )
                f.read((char*)&initkf.Descriptors.at<unsigned char>(i,j),sizeof(char));

            // Read the mapping from keypoints to MapPoints.
            unsigned long int mpidx;
            f.read((char*)&mpidx, sizeof(mpidx));

            // Look up from vmp, which contains all MapPoints, MapPoint of current KeyFrame, and then insert in vpMapPoints.
            if( mpidx == ULONG_MAX )
                vpMapPoints[i] = NULL;
            else
                vpMapPoints[i] = vmp[mpidx];
        }

        // Read BoW for relocalization.
        // f.read((char*)&initkf.mBowVec, sizeof(initkf.mBowVec));

        initkf.vRight = vector<float>(initkf.N,-1);
        initkf.vDepth = vector<float>(initkf.N,-1);
        //initkf.vDepth = KeypointDepth;
        initkf.UndistortKeyPoints();
        initkf.AssignFeaturesToGrid();

        // Use initkf to initialize a KeyFrame and set parameters
        KeyFrame* kf = new KeyFrame( initkf, this, NULL, vpMapPoints );
        kf->mnId = initkf.nId;
        kf->SetPose(T);
        kf->ComputeBoW();

        for ( int i = 0; i < initkf.N; i ++ )
        {
            if ( vpMapPoints[i] )
            {
                vpMapPoints[i]->AddObservation(kf,i);
                if( !vpMapPoints[i]->GetReferenceKeyFrame())
                    vpMapPoints[i]->SetReferenceKeyFrame(kf);
            }
        }
        return kf;
    }


    void Map::SaveMapPoint( ofstream& f, MapPoint* mp)
    {
        //Save ID and the x,y,z coordinates of the current MapPoint
        f.write((char*)&mp->mnId, sizeof(mp->mnId));
        cv::Mat mpWorldPos = mp->GetWorldPos();
        f.write((char*)& mpWorldPos.at<float>(0),sizeof(float));
        f.write((char*)& mpWorldPos.at<float>(1),sizeof(float));
        f.write((char*)& mpWorldPos.at<float>(2),sizeof(float));
    }

    // Get the Index of the MapPoints that matches the ORB featurepoint
    void Map::GetMapPointsIdx()
    {
        unique_lock<mutex> lock(mMutexMap);
        unsigned long int i = 0;
        for ( auto mp: mspMapPoints )
        {
            mmpnMapPointsIdx[mp] = i;
            i += 1;
        }
    }


    void Map::SaveKeyFrame( ofstream &f, KeyFrame* kf )
    {
        //Save the ID and timesteps of current KeyFrame
        f.write((char*)&kf->mnId, sizeof(kf->mnId));
        // cout << "saving kf->mnId = " << kf->mnId <<endl;
        f.write((char*)&kf->mTimeStamp, sizeof(kf->mTimeStamp));
        //Save the Pose Matrix of current KeyFrame
        cv::Mat Tcw = kf->GetPose();

        ////Save the rotation matrix
        // for ( int i = 0; i < Tcw.rows; i ++ )
        // {
        //     for ( int j = 0; j < Tcw.cols; j ++ )
        //     {
        //         f.write((char*)&Tcw.at<float>(i,j), sizeof(float));
        //         //cerr<<"Tcw.at<float>("<<i<<","<<j<<"):"<<Tcw.at<float>(i,j)<<endl;
        //     }
        // }

        //Save the rotation matrix in Quaternion
        std::vector<float> Quat = Converter::toQuaternion(Tcw);
        for ( int i = 0; i < 4; i ++ )
            f.write((char*)&Quat[i],sizeof(float));
        //Save the translation matrix
        for ( int i = 0; i < 3; i ++ )
            f.write((char*)&Tcw.at<float>(i,3),sizeof(float));

        //Save the size of the ORB features current KeyFrame
        //cerr<<"kf->N:"<<kf->N<<endl;
        f.write((char*)&kf->N, sizeof(kf->N));
        //Save each ORB features
        for( int i = 0; i < kf->N; i ++ )
        {
            cv::KeyPoint kp = kf->mvKeys[i];
            f.write((char*)&kp.pt.x, sizeof(kp.pt.x));
            f.write((char*)&kp.pt.y, sizeof(kp.pt.y));
            f.write((char*)&kp.size, sizeof(kp.size));
            f.write((char*)&kp.angle,sizeof(kp.angle));
            f.write((char*)&kp.response, sizeof(kp.response));
            f.write((char*)&kp.octave, sizeof(kp.octave));

            //Save the Descriptors of current ORB features
            f.write((char*)&kf->mDescriptors.cols, sizeof(kf->mDescriptors.cols)); //kf->mDescriptors.cols is always 32 here.
            for (int j = 0; j < kf->mDescriptors.cols; j ++ )
                f.write((char*)&kf->mDescriptors.at<unsigned char>(i,j), sizeof(char));

            //Save the index of MapPoints that corresponds to current ORB features
            unsigned long int mnIdx;
            MapPoint* mp = kf->GetMapPoint(i);
            if (mp == NULL  )
                mnIdx = ULONG_MAX;
            else
                mnIdx = mmpnMapPointsIdx[mp];

            f.write((char*)&mnIdx, sizeof(mnIdx));
        }

        // Save BoW for relocalization.
        // f.write((char*)&kf->mBowVec, sizeof(kf->mBowVec));
    }


} //namespace Planar_SLAM
