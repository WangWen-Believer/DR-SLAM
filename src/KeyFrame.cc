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

#include "KeyFrame.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include<mutex>

using namespace std;
using namespace cv;
using namespace cv::line_descriptor;
using namespace Eigen;

namespace Planar_SLAM
{

    long unsigned int KeyFrame::nNextId=0;

    KeyFrame::KeyFrame(Frame &F, Map *pMap, KeyFrameDatabase *pKFDB):
            mnFrameId(F.mnId), mTimeStamp(F.mTimeStamp), mnGridCols(FRAME_GRID_COLS), mnGridRows(FRAME_GRID_ROWS),
            mfGridElementWidthInv(F.mfGridElementWidthInv), mfGridElementHeightInv(F.mfGridElementHeightInv),
            mnTrackReferenceForFrame(0), mnFuseTargetForKF(0), mnBALocalForKF(0), mnBAFixedForKF(0),
            mnLoopQuery(0), mnLoopWords(0), mnRelocQuery(0), mnRelocWords(0), mnBAGlobalForKF(0),
            fx(F.fx), fy(F.fy), cx(F.cx), cy(F.cy), invfx(F.invfx), invfy(F.invfy),
            mbf(F.mbf), mb(F.mb), mThDepth(F.mThDepth), N(F.N), mvKeys(F.mvKeys), mvKeysUn(F.mvKeysUn),
            mvuRight(F.mvuRight), mvDepth(F.mvDepth), mDescriptors(F.mDescriptors.clone()),
            mBowVec(F.mBowVec), mFeatVec(F.mFeatVec), mnScaleLevels(F.mnScaleLevels), mfScaleFactor(F.mfScaleFactor),
            mfLogScaleFactor(F.mfLogScaleFactor), mvScaleFactors(F.mvScaleFactors), mvLevelSigma2(F.mvLevelSigma2),
            mvInvLevelSigma2(F.mvInvLevelSigma2), mnMinX(F.mnMinX), mnMinY(F.mnMinY), mnMaxX(F.mnMaxX),
            mnMaxY(F.mnMaxY), mK(F.mK), mvpMapPoints(F.mvpMapPoints), mpKeyFrameDB(pKFDB),
            mpORBvocabulary(F.mpORBvocabulary), mbFirstConnection(true), mpParent(NULL), mbNotErase(false),
            mbToBeErased(false), mbBad(false), mHalfBaseline(F.mb/2), mpMap(pMap),
            NL(F.NL), mvKeyLines(F.mvKeylinesUn), mvDepthLine(F.mvDepthLine), mvKeyLineFunctions(F.mvKeyLineFunctions), mLineDescriptors(F.mLdesc),
            mvpMapLines(F.mvpMapLines), mv3DLineforMap(F.mv3DLineforMap), mvLines3D(F.mvLines3D), mvPlaneCoefficients(F.mvPlaneCoefficients), mnPlaneNum(F.mnPlaneNum),
            mvpMapPlanes(F.mvpMapPlanes), mbNewPlane(F.mbNewPlane), mvPlanePoints(F.mvPlanePoints),
            mvpParallelPlanes(F.mvpParallelPlanes), mvpVerticalPlanes(F.mvpVerticalPlanes)
    {
        mnId=nNextId++;

        mGrid.resize(mnGridCols);
        for(int i=0; i<mnGridCols;i++)
        {
            mGrid[i].resize(mnGridRows);
            for(int j=0; j<mnGridRows; j++)
                mGrid[i][j] = F.mGrid[i][j];
        }

        SetPose(F.mTcw);
    }
    // 重定位暂时不加入线特征，故NL幅值为0
    KeyFrame::KeyFrame(InitKeyFrame &initkf, Map *pMap, KeyFrameDatabase *pKFDB, vector<MapPoint*> &vpMapPoints):
            mnFrameId(0), mTimeStamp(initkf.TimeStamp), mnGridCols(FRAME_GRID_COLS), mnGridRows(FRAME_GRID_ROWS),
            mfGridElementWidthInv(initkf.fGridElementWidthInv), mfGridElementHeightInv(initkf.fGridElementHeightInv),
            mnTrackReferenceForFrame(0), mnFuseTargetForKF(0), mnBALocalForKF(0), mnBAFixedForKF(0),
            mnLoopQuery(0), mnLoopWords(0), mnRelocQuery(0), mnRelocWords(0), mnBAGlobalForKF(0),
            fx(initkf.fx), fy(initkf.fy), cx(initkf.cx), cy(initkf.cy), invfx(initkf.invfx),
            invfy(initkf.invfy), mbf(initkf.bf), mb(initkf.b), mThDepth(initkf.ThDepth), N(initkf.N),
            mvKeys(initkf.vKps), mvKeysUn(initkf.vKpsUn), mvuRight(initkf.vRight), mvDepth(initkf.vDepth),
            mDescriptors(initkf.Descriptors.clone()), mBowVec(initkf.BowVec), mFeatVec(initkf.FeatVec),
            mnScaleLevels(initkf.nScaleLevels), mfScaleFactor(initkf.fScaleFactor), mfLogScaleFactor(initkf.fLogScaleFactor),
            mvScaleFactors(initkf.vScaleFactors), mvLevelSigma2(initkf.vLevelSigma2),mvInvLevelSigma2(initkf.vInvLevelSigma2),
            mnMinX(initkf.nMinX), mnMinY(initkf.nMinY), mnMaxX(initkf.nMaxX), mnMaxY(initkf.nMaxY), mK(initkf.K),
            mvpMapPoints(vpMapPoints), mpKeyFrameDB(pKFDB), mpORBvocabulary(initkf.pVocabulary),
            mbFirstConnection(true), mpParent(NULL), mbNotErase(false), mbToBeErased(false), mbBad(false),
            mHalfBaseline(initkf.b/2), mpMap(pMap),NL(0)
    {
        nNextId++;

        mGrid.resize(mnGridCols);
        for(int i=0; i<mnGridCols;i++)
        {
            mGrid[i].resize(mnGridRows);
            for(int j=0; j<mnGridRows; j++)
                mGrid[i][j] = initkf.vGrid[i][j];
        }

    }


    KeyFrame::KeyFrame(Frame &F, Map *pMap, KeyFrameDatabase *pKFDB, cv::Mat rgb, cv::Mat depth):
        mnFrameId(F.mnId), mTimeStamp(F.mTimeStamp), mnGridCols(FRAME_GRID_COLS), mnGridRows(FRAME_GRID_ROWS),
        mfGridElementWidthInv(F.mfGridElementWidthInv), mfGridElementHeightInv(F.mfGridElementHeightInv),
        mnTrackReferenceForFrame(0), mnFuseTargetForKF(0), mnBALocalForKF(0), mnBAFixedForKF(0),
        mnLoopQuery(0), mnLoopWords(0), mnRelocQuery(0), mnRelocWords(0), mnBAGlobalForKF(0),
        fx(F.fx), fy(F.fy), cx(F.cx), cy(F.cy), invfx(F.invfx), invfy(F.invfy),
        mbf(F.mbf), mb(F.mb), mThDepth(F.mThDepth), N(F.N), mvKeys(F.mvKeys), mvKeysUn(F.mvKeysUn),
        mvuRight(F.mvuRight), mvDepth(F.mvDepth), mDescriptors(F.mDescriptors.clone()),
        mBowVec(F.mBowVec), mFeatVec(F.mFeatVec), mnScaleLevels(F.mnScaleLevels), mfScaleFactor(F.mfScaleFactor),
        mfLogScaleFactor(F.mfLogScaleFactor), mvScaleFactors(F.mvScaleFactors), mvLevelSigma2(F.mvLevelSigma2),
        mvInvLevelSigma2(F.mvInvLevelSigma2), mnMinX(F.mnMinX), mnMinY(F.mnMinY), mnMaxX(F.mnMaxX),
        mnMaxY(F.mnMaxY), mK(F.mK), mvpMapPoints(F.mvpMapPoints), mpKeyFrameDB(pKFDB),
        mpORBvocabulary(F.mpORBvocabulary), mbFirstConnection(true), mpParent(NULL), mbNotErase(false),
        mbToBeErased(false), mbBad(false), mHalfBaseline(F.mb/2), mpMap(pMap),
        NL(F.NL), mvKeyLines(F.mvKeylinesUn), mvDepthLine(F.mvDepthLine), mvKeyLineFunctions(F.mvKeyLineFunctions), mLineDescriptors(F.mLdesc),
        mvpMapLines(F.mvpMapLines), mv3DLineforMap(F.mv3DLineforMap), mvLines3D(F.mvLines3D), mvPlaneCoefficients(F.mvPlaneCoefficients), mnPlaneNum(F.mnPlaneNum),
        mvpMapPlanes(F.mvpMapPlanes), mbNewPlane(F.mbNewPlane), mvPlanePoints(F.mvPlanePoints),
        mvpParallelPlanes(F.mvpParallelPlanes), mvpVerticalPlanes(F.mvpVerticalPlanes)
{
    rgb.copyTo(mImRGB);
    depth.copyTo(mImDep);
    // 获取id
    mnId=nNextId++;

    // 根据指定的普通帧, 初始化用于加速匹配的网格对象信息; 其实就把每个网格中有的特征点的索引复制过来
    mGrid.resize(mnGridCols);
    for(int i=0; i<mnGridCols;i++)
    {
        mGrid[i].resize(mnGridRows);
        for(int j=0; j<mnGridRows; j++)
            mGrid[i][j] = F.mGrid[i][j];
    }

    // 设置当前关键帧的位姿
    SetPose(F.mTcw);
}



    void KeyFrame::ComputeBoW()
    {
        if(mBowVec.empty() || mFeatVec.empty())
        {
            vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
            // Feature vector associate features with nodes in the 4th level (from leaves up)
            // We assume the vocabulary tree has 6 levels, change the 4 otherwise
            mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
        }
    }

    void KeyFrame::SetPose(const cv::Mat &Tcw_)
    {
        unique_lock<mutex> lock(mMutexPose);
        Tcw_.copyTo(Tcw);
        cv::Mat Rcw = Tcw.rowRange(0,3).colRange(0,3);
        cv::Mat tcw = Tcw.rowRange(0,3).col(3);
        cv::Mat Rwc = Rcw.t();
        Ow = -Rwc*tcw;

        Twc = cv::Mat::eye(4,4,Tcw.type());
        Rwc.copyTo(Twc.rowRange(0,3).colRange(0,3));
        Ow.copyTo(Twc.rowRange(0,3).col(3));
        cv::Mat center = (cv::Mat_<float>(4,1) << mHalfBaseline, 0 , 0, 1);
        Cw = Twc*center;
    }

    cv::Mat KeyFrame::GetPose()
    {
        unique_lock<mutex> lock(mMutexPose);
        return Tcw.clone();
    }

    cv::Mat KeyFrame::GetPoseInverse()
    {
        unique_lock<mutex> lock(mMutexPose);
        return Twc.clone();
    }

    cv::Mat KeyFrame::GetCameraCenter()
    {
        unique_lock<mutex> lock(mMutexPose);
        return Ow.clone();
    }

    cv::Mat KeyFrame::GetStereoCenter()
    {
        unique_lock<mutex> lock(mMutexPose);
        return Cw.clone();
    }


    cv::Mat KeyFrame::GetRotation()
    {
        unique_lock<mutex> lock(mMutexPose);
        return Tcw.rowRange(0,3).colRange(0,3).clone();
    }

    cv::Mat KeyFrame::GetTranslation()
    {
        unique_lock<mutex> lock(mMutexPose);
        return Tcw.rowRange(0,3).col(3).clone();
    }

    void KeyFrame::AddConnection(KeyFrame *pKF, const int &weight)
    {
        {
            unique_lock<mutex> lock(mMutexConnections);
            if(!mConnectedKeyFrameWeights.count(pKF))
                mConnectedKeyFrameWeights[pKF]=weight;
            else if(mConnectedKeyFrameWeights[pKF]!=weight)
                mConnectedKeyFrameWeights[pKF]=weight;
            else
                return;
        }

        UpdateBestCovisibles();
    }

    void KeyFrame::UpdateBestCovisibles()
    {
        unique_lock<mutex> lock(mMutexConnections);
        vector<pair<int,KeyFrame*> > vPairs;
        vPairs.reserve(mConnectedKeyFrameWeights.size());
        for(map<KeyFrame*,int>::iterator mit=mConnectedKeyFrameWeights.begin(), mend=mConnectedKeyFrameWeights.end(); mit!=mend; mit++)
            vPairs.push_back(make_pair(mit->second,mit->first));

        sort(vPairs.begin(),vPairs.end());
        list<KeyFrame*> lKFs;
        list<int> lWs;
        for(size_t i=0, iend=vPairs.size(); i<iend;i++)
        {
            lKFs.push_front(vPairs[i].second);
            lWs.push_front(vPairs[i].first);
        }

        mvpOrderedConnectedKeyFrames = vector<KeyFrame*>(lKFs.begin(),lKFs.end());
        mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());
    }

    set<KeyFrame*> KeyFrame::GetConnectedKeyFrames()
    {
        unique_lock<mutex> lock(mMutexConnections);
        set<KeyFrame*> s;
        for(map<KeyFrame*,int>::iterator mit=mConnectedKeyFrameWeights.begin();mit!=mConnectedKeyFrameWeights.end();mit++)
            s.insert(mit->first);
        return s;
    }

    vector<KeyFrame*> KeyFrame::GetVectorCovisibleKeyFrames()
    {
        unique_lock<mutex> lock(mMutexConnections);
        return mvpOrderedConnectedKeyFrames;
    }

    vector<KeyFrame*> KeyFrame::GetBestCovisibilityKeyFrames(const int &N)
    {
        unique_lock<mutex> lock(mMutexConnections);
        if((int)mvpOrderedConnectedKeyFrames.size()<N)
            return mvpOrderedConnectedKeyFrames;
        else
            return vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(),mvpOrderedConnectedKeyFrames.begin()+N);

    }

    vector<KeyFrame*> KeyFrame::GetCovisiblesByWeight(const int &w)
    {
        unique_lock<mutex> lock(mMutexConnections);

        if(mvpOrderedConnectedKeyFrames.empty())
            return vector<KeyFrame*>();

        vector<int>::iterator it = upper_bound(mvOrderedWeights.begin(),mvOrderedWeights.end(),w,KeyFrame::weightComp);
        if(it==mvOrderedWeights.end())
            return vector<KeyFrame*>();
        else
        {
            int n = it-mvOrderedWeights.begin();
            return vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin()+n);
        }
    }

    int KeyFrame::GetWeight(KeyFrame *pKF)
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(mConnectedKeyFrameWeights.count(pKF))
            return mConnectedKeyFrameWeights[pKF];
        else
            return 0;
    }

    void KeyFrame::AddMapPoint(MapPoint *pMP, const size_t &idx)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        mvpMapPoints[idx]=pMP;
    }

    void KeyFrame::EraseMapPointMatch(const size_t &idx)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        mvpMapPoints[idx]=static_cast<MapPoint*>(NULL);
    }

    void KeyFrame::EraseMapPointMatch(MapPoint* pMP)
    {
        int idx = pMP->GetIndexInKeyFrame(this);
        if(idx>=0)
            mvpMapPoints[idx]=static_cast<MapPoint*>(NULL);
    }


    void KeyFrame::ReplaceMapPointMatch(const size_t &idx, MapPoint* pMP)
    {
        mvpMapPoints[idx]=pMP;
    }

    set<MapPoint*> KeyFrame::GetMapPoints()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        set<MapPoint*> s;
        for(size_t i=0, iend=mvpMapPoints.size(); i<iend; i++)
        {
            if(!mvpMapPoints[i])
                continue;
            MapPoint* pMP = mvpMapPoints[i];
            if(!pMP->isBad())
                s.insert(pMP);
        }
        return s;
    }

    int KeyFrame::TrackedMapPoints(const int &minObs)
    {
        unique_lock<mutex> lock(mMutexFeatures);

        int nPoints=0;
        const bool bCheckObs = minObs>0;
        for(int i=0; i<N; i++)
        {
            MapPoint* pMP = mvpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    if(bCheckObs)
                    {
                        if(mvpMapPoints[i]->Observations()>=minObs)
                            nPoints++;
                    }
                    else
                        nPoints++;
                }
            }
        }

        return nPoints;
    }

    vector<MapPoint*> KeyFrame::GetMapPointMatches()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return mvpMapPoints;
    }

    MapPoint* KeyFrame::GetMapPoint(const size_t &idx)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return mvpMapPoints[idx];
    }

    void KeyFrame::UpdateConnections()
    {
        map<KeyFrame*,int> KFcounter;

        vector<MapPoint*> vpMP;
        vector<MapLine*> vpML;
        vector<MapPlane*> vpMPL;

        {
            unique_lock<mutex> lockMPs(mMutexFeatures);
            vpMP = mvpMapPoints;
            vpML = mvpMapLines;
            vpMPL = mvpMapPlanes;
        }

        //For all map points in keyframe check in which other keyframes are they seen
        //Increase counter for those keyframes
        for(vector<MapPoint*>::iterator vit=vpMP.begin(), vend=vpMP.end(); vit!=vend; vit++)
        {
            MapPoint* pMP = *vit;

            if(!pMP)
                continue;

            if(pMP->isBad())
                continue;

            map<KeyFrame*,size_t> observations = pMP->GetObservations();

            for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
            {
                if(mit->first->mnId==mnId)
                    continue;
                KFcounter[mit->first]++;
            }
        }

//        for(vector<MapLine*>::iterator vit=vpML.begin(), vend=vpML.end(); vit!=vend; vit++)
//        {
//            MapLine* pML = *vit;
//
//            if(!pML)
//                continue;
//
//            if(pML->isBad())
//                continue;
//
//            map<KeyFrame*,size_t> observations = pML->GetObservations();
//
//            for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
//            {
//                if(mit->first->mnId==mnId)
//                    continue;
//                KFcounter[mit->first]++;
//            }
//        }
//
//        for(vector<MapPlane*>::iterator vit=vpMPL.begin(), vend=vpMPL.end(); vit!=vend; vit++)
//        {
//            MapPlane* vpMPL = *vit;
//
//            if(!vpMPL)
//                continue;
//
//            if(vpMPL->isBad())
//                continue;
//
//            map<KeyFrame*,size_t> observations = vpMPL->GetObservations();
//
//            for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
//            {
//                if(mit->first->mnId==mnId)
//                    continue;
//                KFcounter[mit->first]++;
//            }
//        }

        // This should not happen
        if(KFcounter.empty())
            return;

        //If the counter is greater than threshold add connection
        //In case no keyframe counter is over threshold add the one with maximum counter
        int nmax=0;
        KeyFrame* pKFmax=NULL;
        int th = 15;

        vector<pair<int,KeyFrame*> > vPairs;
        vPairs.reserve(KFcounter.size());
        for(map<KeyFrame*,int>::iterator mit=KFcounter.begin(), mend=KFcounter.end(); mit!=mend; mit++)
        {
            if(mit->second>nmax)
            {
                nmax=mit->second;
                pKFmax=mit->first;
            }
            if(mit->second>=th)
            {
                vPairs.push_back(make_pair(mit->second,mit->first));
                (mit->first)->AddConnection(this,mit->second);
            }
        }

        if(vPairs.empty())
        {
            vPairs.push_back(make_pair(nmax,pKFmax));
            pKFmax->AddConnection(this,nmax);
        }

        sort(vPairs.begin(),vPairs.end());
        list<KeyFrame*> lKFs;
        list<int> lWs;
        for(size_t i=0; i<vPairs.size();i++)
        {
            lKFs.push_front(vPairs[i].second);
            lWs.push_front(vPairs[i].first);
        }

        {
            unique_lock<mutex> lockCon(mMutexConnections);

            // mspConnectedKeyFrames = spConnectedKeyFrames;
            mConnectedKeyFrameWeights = KFcounter;
            mvpOrderedConnectedKeyFrames = vector<KeyFrame*>(lKFs.begin(),lKFs.end());
            mvOrderedWeights = vector<int>(lWs.begin(), lWs.end());

            if(mbFirstConnection && mnId!=0)
            {
                mpParent = mvpOrderedConnectedKeyFrames.front();
                mpParent->AddChild(this);
                mbFirstConnection = false;
            }

        }
    }

    void KeyFrame::AddChild(KeyFrame *pKF)
    {
        unique_lock<mutex> lockCon(mMutexConnections);
        mspChildrens.insert(pKF);
    }

    void KeyFrame::EraseChild(KeyFrame *pKF)
    {
        unique_lock<mutex> lockCon(mMutexConnections);
        mspChildrens.erase(pKF);
    }

    void KeyFrame::ChangeParent(KeyFrame *pKF)
    {
        unique_lock<mutex> lockCon(mMutexConnections);
        mpParent = pKF;
        pKF->AddChild(this);
    }

    set<KeyFrame*> KeyFrame::GetChilds()
    {
        unique_lock<mutex> lockCon(mMutexConnections);
        return mspChildrens;
    }

    KeyFrame* KeyFrame::GetParent()
    {
        unique_lock<mutex> lockCon(mMutexConnections);
        return mpParent;
    }

    bool KeyFrame::hasChild(KeyFrame *pKF)
    {
        unique_lock<mutex> lockCon(mMutexConnections);
        return mspChildrens.count(pKF);
    }

    void KeyFrame::AddLoopEdge(KeyFrame *pKF)
    {
        unique_lock<mutex> lockCon(mMutexConnections);
        mbNotErase = true;
        mspLoopEdges.insert(pKF);
    }

    set<KeyFrame*> KeyFrame::GetLoopEdges()
    {
        unique_lock<mutex> lockCon(mMutexConnections);
        return mspLoopEdges;
    }

    void KeyFrame::SetNotErase()
    {
        unique_lock<mutex> lock(mMutexConnections);
        mbNotErase = true;
    }

    void KeyFrame::SetErase()
    {
        {
            unique_lock<mutex> lock(mMutexConnections);
            if(mspLoopEdges.empty())
            {
                mbNotErase = false;
            }
        }

        if(mbToBeErased)
        {
            SetBadFlag();
        }
    }

    void KeyFrame::SetBadFlag()
    {
        {
            unique_lock<mutex> lock(mMutexConnections);
            if(mnId==0)
                return;
            else if(mbNotErase)
            {
                mbToBeErased = true;
                return;
            }
        }

        for(auto & mConnectedKeyFrameWeight : mConnectedKeyFrameWeights)
            mConnectedKeyFrameWeight.first->EraseConnection(this);

        for(auto & mvpMapPoint : mvpMapPoints)
            if(mvpMapPoint)
                mvpMapPoint->EraseObservation(this);

        for(auto & mvpMapLine : mvpMapLines)
            if(mvpMapLine)
                mvpMapLine->EraseObservation(this);

        for(auto & mvpMapPlane : mvpMapPlanes)
            if(mvpMapPlane)
                mvpMapPlane->EraseObservation(this);

        for(auto & mvpVerticalPlane : mvpVerticalPlanes)
            if(mvpVerticalPlane)
                mvpVerticalPlane->EraseVerObservation(this);

        for(auto & mvpParallelPlane : mvpParallelPlanes)
            if(mvpParallelPlane)
                mvpParallelPlane->EraseParObservation(this);

        {
            unique_lock<mutex> lock(mMutexConnections);
            unique_lock<mutex> lock1(mMutexFeatures);

            mConnectedKeyFrameWeights.clear();
            mvpOrderedConnectedKeyFrames.clear();

            // Update Spanning Tree
            set<KeyFrame*> sParentCandidates;
            sParentCandidates.insert(mpParent);

            // Assign at each iteration one children with a parent (the pair with highest covisibility weight)
            // Include that children as new parent candidate for the rest
            while(!mspChildrens.empty())
            {
                bool bContinue = false;

                int max = -1;
                KeyFrame* pC;
                KeyFrame* pP;

                for(set<KeyFrame*>::iterator sit=mspChildrens.begin(), send=mspChildrens.end(); sit!=send; sit++)
                {
                    KeyFrame* pKF = *sit;
                    if(pKF->isBad())
                        continue;

                    // Check if a parent candidate is connected to the keyframe
                    vector<KeyFrame*> vpConnected = pKF->GetVectorCovisibleKeyFrames();
                    for(size_t i=0, iend=vpConnected.size(); i<iend; i++)
                    {
                        for(set<KeyFrame*>::iterator spcit=sParentCandidates.begin(), spcend=sParentCandidates.end(); spcit!=spcend; spcit++)
                        {
                            if(vpConnected[i]->mnId == (*spcit)->mnId)
                            {
                                int w = pKF->GetWeight(vpConnected[i]);
                                if(w>max)
                                {
                                    pC = pKF;
                                    pP = vpConnected[i];
                                    max = w;
                                    bContinue = true;
                                }
                            }
                        }
                    }
                }

                if(bContinue)
                {
                    pC->ChangeParent(pP);
                    sParentCandidates.insert(pC);
                    mspChildrens.erase(pC);
                }
                else
                    break;
            }

            // If a children has no covisibility links with any parent candidate, assign to the original parent of this KF
            if(!mspChildrens.empty())
                for(set<KeyFrame*>::iterator sit=mspChildrens.begin(); sit!=mspChildrens.end(); sit++)
                {
                    (*sit)->ChangeParent(mpParent);
                }

            mpParent->EraseChild(this);
            mTcp = Tcw*mpParent->GetPoseInverse();
            mbBad = true;
        }


        mpMap->EraseKeyFrame(this);
        mpKeyFrameDB->erase(this);
    }

    bool KeyFrame::isBad()
    {
        unique_lock<mutex> lock(mMutexConnections);
        return mbBad;
    }

    void KeyFrame::EraseConnection(KeyFrame* pKF)
    {
        bool bUpdate = false;
        {
            unique_lock<mutex> lock(mMutexConnections);
            if(mConnectedKeyFrameWeights.count(pKF))
            {
                mConnectedKeyFrameWeights.erase(pKF);
                bUpdate=true;
            }
        }

        if(bUpdate)
            UpdateBestCovisibles();
    }

    vector<size_t> KeyFrame::GetFeaturesInArea(const float &x, const float &y, const float &r) const
    {
        vector<size_t> vIndices;
        vIndices.reserve(N);

        const int nMinCellX = max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));
        if(nMinCellX>=mnGridCols)
            return vIndices;

        const int nMaxCellX = min((int)mnGridCols-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
        if(nMaxCellX<0)
            return vIndices;

        const int nMinCellY = max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
        if(nMinCellY>=mnGridRows)
            return vIndices;

        const int nMaxCellY = min((int)mnGridRows-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
        if(nMaxCellY<0)
            return vIndices;

        for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
        {
            for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
            {
                const vector<size_t> vCell = mGrid[ix][iy];
                for(size_t j=0, jend=vCell.size(); j<jend; j++)
                {
                    const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                    const float distx = kpUn.pt.x-x;
                    const float disty = kpUn.pt.y-y;

                    if(fabs(distx)<r && fabs(disty)<r)
                        vIndices.push_back(vCell[j]);
                }
            }
        }

        return vIndices;
    }

    vector<size_t>
    KeyFrame::GetLinesInArea(const float &x1, const float &y1, const float &x2, const float &y2, const float &r,
                             const int minLevel, const int maxLevel) const {
        vector<size_t> vIndices;

        vector<KeyLine> vkl = this->mvKeyLines;

        const bool bCheckLevels = (minLevel > 0) || (maxLevel > 0);

        for (size_t i = 0; i < vkl.size(); i++) {
            KeyLine keyline = vkl[i];

            // 1.对比中点距离
            float distance = (0.5 * (x1 + x2) - keyline.pt.x) * (0.5 * (x1 + x2) - keyline.pt.x) +
                             (0.5 * (y1 + y2) - keyline.pt.y) * (0.5 * (y1 + y2) - keyline.pt.y);
            if (distance > r * r)
                continue;

            float slope = (y1 - y2) / (x1 - x2) - keyline.angle;
            if (slope > r * 0.01)
                continue;

            if (bCheckLevels) {
                if (keyline.octave < minLevel)
                    continue;
                if (maxLevel >= 0 && keyline.octave > maxLevel)
                    continue;
            }

            vIndices.push_back(i);
        }

        return vIndices;
    }

    bool KeyFrame::IsInImage(const float &x, const float &y) const
    {
        return (x>=mnMinX && x<mnMaxX && y>=mnMinY && y<mnMaxY);
    }

    cv::Mat KeyFrame::UnprojectStereo(int i)
    {
        const float z = mvDepth[i];
        if(z>0)
        {
            const float u = mvKeys[i].pt.x;
            const float v = mvKeys[i].pt.y;
            const float x = (u-cx)*z*invfx;
            const float y = (v-cy)*z*invfy;
            cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);

            unique_lock<mutex> lock(mMutexPose);
            return Twc.rowRange(0,3).colRange(0,3)*x3Dc+Twc.rowRange(0,3).col(3);
        }
        else
            return cv::Mat();
    }

    Vector6d KeyFrame::obtain3DLine(const int &i) {
        Vector6d Lines3D = mvLines3D[i];
        cv::Mat Ac = (Mat_<float>(3, 1) << Lines3D(0), Lines3D(1), Lines3D(2));
        cv::Mat A = Twc.rowRange(0,3).colRange(0,3) * Ac + Twc.rowRange(0,3).col(3);
        cv::Mat Bc = (Mat_<float>(3, 1) << Lines3D(3), Lines3D(4), Lines3D(5));
        cv::Mat B = Twc.rowRange(0,3).colRange(0,3) * Bc + Twc.rowRange(0,3).col(3);
        Lines3D << A.at<float>(0, 0), A.at<float>(1, 0), A.at<float>(2, 0),
                B.at<float>(0, 0), B.at<float>(1,0), B.at<float>(2, 0);
        return Lines3D;
    }

    float KeyFrame::ComputeSceneMedianDepth(const int q)
    {
        vector<MapPoint*> vpMapPoints;
        cv::Mat Tcw_;
        {
            unique_lock<mutex> lock(mMutexFeatures);
            unique_lock<mutex> lock2(mMutexPose);
            vpMapPoints = mvpMapPoints;
            Tcw_ = Tcw.clone();
        }

        vector<float> vDepths;
        vDepths.reserve(N);
        cv::Mat Rcw2 = Tcw_.row(2).colRange(0,3);
        Rcw2 = Rcw2.t();
        float zcw = Tcw_.at<float>(2,3);
        for(int i=0; i<N; i++)
        {
            if(mvpMapPoints[i])
            {
                MapPoint* pMP = mvpMapPoints[i];
                cv::Mat x3Dw = pMP->GetWorldPos();
                float z = Rcw2.dot(x3Dw)+zcw;
                vDepths.push_back(z);
            }
        }

        sort(vDepths.begin(),vDepths.end());

        return vDepths[(vDepths.size()-1)/q];
    }

    void KeyFrame::AddMapLine(MapLine *pML, const size_t &idx)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        mvpMapLines[idx]=pML;
    }

    void KeyFrame::EraseMapLineMatch(const size_t &idx)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        mvpMapLines[idx]= static_cast<MapLine*>(NULL);
    }

    void KeyFrame::EraseMapLineMatch(MapLine *pML)
    {
        int idx = pML->GetIndexInKeyFrame(this);
        if(idx>=0)
            mvpMapLines[idx]= static_cast<MapLine*>(NULL);
    }

    void KeyFrame::ReplaceMapLineMatch(const size_t &idx, MapLine *pML)
    {
        mvpMapLines[idx]=pML;
    }

    set<MapLine*> KeyFrame::GetMapLines()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        set<MapLine*> s;
        for(size_t i=0, iend=mvpMapLines.size(); i<iend; i++)
        {
            if(!mvpMapLines[i])
                continue;
            MapLine* pML = mvpMapLines[i];
            if(!pML->isBad())
                s.insert(pML);
        }
        return s;
    }

    vector<MapLine*> KeyFrame::GetMapLineMatches()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return mvpMapLines;
    }

    int KeyFrame::TrackedMapLines(const int &minObs)
    {
        unique_lock<mutex> lock(mMutexFeatures);

        int nLines = 0;
        const bool bCheckObs = minObs>0;
        for(int i=0; i<NL; i++)
        {
            MapLine* pML = mvpMapLines[i];
            if(pML)
            {
                if(!pML->isBad())
                {
                    if(bCheckObs)
                    {
                        //该MapLine是一个高质量的MapLine
                        if(mvpMapLines[i]->Observations()>=minObs)
                            nLines++;
                    } else
                        nLines++;
                }
            }
        }
        return nLines;
    }

    MapLine* KeyFrame::GetMapLine(const size_t &idx)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return mvpMapLines[idx];
    }

    void KeyFrame::lineDescriptorMAD(std::vector<std::vector<cv::DMatch>> line_matches, double &nn_mad, double &nn12_mad) const
    {
        vector<vector<DMatch>> matches_nn, matches_12;
        matches_nn = line_matches;
        matches_12 = line_matches;
//    cout << "Frame::lineDescriptorMAD——matches_nn = "<<matches_nn.size() << endl;

        // estimate the NN's distance standard deviation
        double nn_dist_median;
        sort( matches_nn.begin(), matches_nn.end(), compare_descriptor_by_NN_dist());
        nn_dist_median = matches_nn[int(matches_nn.size()/2)][0].distance;

        for(unsigned int i=0; i<matches_nn.size(); i++)
            matches_nn[i][0].distance = fabsf(matches_nn[i][0].distance - nn_dist_median);
        sort(matches_nn.begin(), matches_nn.end(), compare_descriptor_by_NN_dist());
        nn_mad = 1.4826 * matches_nn[int(matches_nn.size()/2)][0].distance;

        // estimate the NN's 12 distance standard deviation
        double nn12_dist_median;
        sort( matches_12.begin(), matches_12.end(), conpare_descriptor_by_NN12_dist());
        nn12_dist_median = matches_12[int(matches_12.size()/2)][1].distance - matches_12[int(matches_12.size()/2)][0].distance;
        for (unsigned int j=0; j<matches_12.size(); j++)
            matches_12[j][0].distance = fabsf( matches_12[j][1].distance - matches_12[j][0].distance - nn12_dist_median);
        sort(matches_12.begin(), matches_12.end(), compare_descriptor_by_NN_dist());
        nn12_mad = 1.4826 * matches_12[int(matches_12.size()/2)][0].distance;
    }

    cv::Mat KeyFrame::ComputePlaneWorldCoeff(const int &idx) {
        cv::Mat temp;
        cv::transpose(Tcw, temp);
        return temp*mvPlaneCoefficients[idx];
    }

    void KeyFrame::AddMapPlane(Planar_SLAM::MapPlane *pMP, const int &idx) {
        unique_lock<mutex> lock(mMutexFeatures);
        mvpMapPlanes[idx] = pMP;
    }

    void KeyFrame::AddMapVerticalPlane(Planar_SLAM::MapPlane *pMP, const int &idx) {
        unique_lock<mutex> lock(mMutexFeatures);
        mvpVerticalPlanes[idx] = pMP;
    }

    void KeyFrame::AddMapParallelPlane(Planar_SLAM::MapPlane *pMP, const int &idx) {
        unique_lock<mutex> lock(mMutexFeatures);
        mvpParallelPlanes[idx] = pMP;
    }

    void KeyFrame::EraseMapPlaneMatch(const int &idx) {
        unique_lock<mutex> lock(mMutexFeatures);
        mvpMapPlanes[idx]=static_cast<MapPlane*>(NULL);
    }

    void KeyFrame::EraseMapVerticalPlaneMatch(const int &idx) {
        unique_lock<mutex> lock(mMutexFeatures);
        mvpVerticalPlanes[idx]=static_cast<MapPlane*>(NULL);
    }

    void KeyFrame::EraseMapParallelPlaneMatch(const int &idx) {
        unique_lock<mutex> lock(mMutexFeatures);
        mvpParallelPlanes[idx]=static_cast<MapPlane*>(NULL);
    }

    void KeyFrame::EraseMapPlaneMatch(MapPlane *pMP) {
        int idx = pMP->GetIndexInKeyFrame(this);
        unique_lock<mutex> lock(mMutexFeatures);
        if(idx>=0)
            mvpMapPlanes[idx]=static_cast<MapPlane*>(NULL);
    }

    void KeyFrame::EraseMapVerticalPlaneMatch(MapPlane *pMP) {
        int idx = pMP->GetIndexInVerticalKeyFrame(this);
        unique_lock<mutex> lock(mMutexFeatures);
        if(idx>=0)
            mvpVerticalPlanes[idx]=static_cast<MapPlane*>(NULL);
    }

    void KeyFrame::EraseMapParallelPlaneMatch(MapPlane *pMP) {
        int idx = pMP->GetIndexInParallelKeyFrame(this);
        unique_lock<mutex> lock(mMutexFeatures);
        if(idx>=0)
            mvpParallelPlanes[idx]=static_cast<MapPlane*>(NULL);
    }

    void KeyFrame::ReplaceMapPlaneMatch(const size_t &idx, MapPlane* pMP)
    {
        mvpMapPlanes[idx]=pMP;
    }

    void KeyFrame::ReplaceMapVerticalPlaneMatch(const size_t &idx, MapPlane* pMP)
    {
        mvpVerticalPlanes[idx]=pMP;
    }

    void KeyFrame::ReplaceMapParallelPlaneMatch(const size_t &idx, MapPlane* pMP)
    {
        mvpParallelPlanes[idx]=pMP;
    }

    vector<MapPlane*> KeyFrame::GetMapPlaneMatches()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return mvpMapPlanes;
    }

    vector<MapPlane*> KeyFrame::GetMapVerticalPlaneMatches()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return mvpVerticalPlanes;
    }

    vector<MapPlane*> KeyFrame::GetMapParallelPlaneMatches()
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return mvpParallelPlanes;
    }

    MapPlane* KeyFrame::GetMapPlane(const size_t &idx)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return mvpMapPlanes[idx];
    }

    MapPlane* KeyFrame::GetMapVerticalPlane(const size_t &idx)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return mvpVerticalPlanes[idx];
    }

    MapPlane* KeyFrame::GetMapParallelPlane(const size_t &idx)
    {
        unique_lock<mutex> lock(mMutexFeatures);
        return mvpParallelPlanes[idx];
    }

    std::set<MapPlane *> KeyFrame::GetMapPlanes() {
        unique_lock<mutex> lock(mMutexFeatures);
        set<MapPlane*> s;
        for(size_t i=0, iend=mvpMapPlanes.size(); i<iend; i++)
        {
            if(!mvpMapPlanes[i])
                continue;
            MapPlane* pMP = mvpMapPlanes[i];
            if(!pMP->isBad())
                s.insert(pMP);
        }
        return s;
    }

    std::set<MapPlane *> KeyFrame::GetMapVerticalPlanes() {
        unique_lock<mutex> lock(mMutexFeatures);
        set<MapPlane*> s;
        for(size_t i=0, iend=mvpVerticalPlanes.size(); i<iend; i++)
        {
            if(!mvpVerticalPlanes[i])
                continue;
            MapPlane* pMP = mvpVerticalPlanes[i];
            if(!pMP->isBad())
                s.insert(pMP);
        }
        return s;
    }

    std::set<MapPlane *> KeyFrame::GetMapParallelPlanes() {
        unique_lock<mutex> lock(mMutexFeatures);
        set<MapPlane*> s;
        for(size_t i=0, iend=mvpParallelPlanes.size(); i<iend; i++)
        {
            if(!mvpParallelPlanes[i])
                continue;
            MapPlane* pMP = mvpParallelPlanes[i];
            if(!pMP->isBad())
                s.insert(pMP);
        }
        return s;
    }

} //namespace ORB_SLAM