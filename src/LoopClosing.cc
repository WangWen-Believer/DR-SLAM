/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
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

#include "LoopClosing.h"

#include "Sim3Solver.h"

#include "Converter.h"

#include "Optimizer.h"

#include "ORBmatcher.h"

#include<mutex>
#include<thread>


namespace Planar_SLAM
{

    LoopClosing::LoopClosing(Map *pMap, KeyFrameDatabase *pDB, ORBVocabulary *pVoc, const bool bFixScale):
            mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
            mpKeyFrameDB(pDB), mpORBVocabulary(pVoc), mpMatchedKF(NULL), mLastLoopKFid(0), mbRunningGBA(false), mbFinishedGBA(true),
            mbStopGBA(false), mpThreadGBA(NULL), mbFixScale(bFixScale), mnFullBAIdx(0)
    {
        mnCovisibilityConsistencyTh = 3;
    }

    void LoopClosing::SetTracker(Tracking *pTracker)
    {
        mpTracker=pTracker;
    }

    void LoopClosing::SetLocalMapper(LocalMapping *pLocalMapper)
    {
        mpLocalMapper=pLocalMapper;
    }


    void LoopClosing::Run()
    {
        mbFinished =false;

        while(1)
        {
            // Check if there are keyframes in the queue
            if(CheckNewKeyFrames())
            {
                // Detect loop candidates and check covisibility consistency
                if(DetectLoop())
                {
                    // Compute similarity transformation [sR|t]
                    // In the stereo/RGBD case s=1
                    if(ComputeSim3())
                    {
                        // Perform loop fusion and pose graph optimization
                        CorrectLoop();
                    }
                }
            }

            ResetIfRequested();

            if(CheckFinish())
                break;

            usleep(5000);
        }

        SetFinish();
    }

    void LoopClosing::InsertKeyFrame(KeyFrame *pKF)
    {
        unique_lock<mutex> lock(mMutexLoopQueue);
        if(pKF->mnId!=0)
            mlpLoopKeyFrameQueue.push_back(pKF);
    }

    bool LoopClosing::CheckNewKeyFrames()
    {
        unique_lock<mutex> lock(mMutexLoopQueue);
        return(!mlpLoopKeyFrameQueue.empty());
    }
/**
 * @brief 闭环检测
 *
 * @return true             成功检测到闭环
 * @return false            未检测到闭环
 */
    bool LoopClosing::DetectLoop()
    {
        {
            // Step 1 从队列中取出一个关键帧,作为当前检测闭环关键帧
            unique_lock<mutex> lock(mMutexLoopQueue);
            // 从队列头开始取，也就是先取早进来的关键帧
            mpCurrentKF = mlpLoopKeyFrameQueue.front();
            // 取出关键帧后从队列里弹出该关键帧
            mlpLoopKeyFrameQueue.pop_front();
            // Avoid that a keyframe can be erased while it is being process by this thread
            // 设置当前关键帧不要在优化的过程中被删除
            mpCurrentKF->SetNotErase();
        }

        //If the map contains less than 10 KF or less than 10 KF have passed from last loop detection
        // Step 2：如果距离上次闭环没多久（小于10帧），或者map中关键帧总共还没有10帧，则不进行闭环检测
        // 后者的体现是当mLastLoopKFid为0的时候
        if(mpCurrentKF->mnId<mLastLoopKFid+10)
        {
            mpKeyFrameDB->add(mpCurrentKF);
            mpCurrentKF->SetErase();
            return false;
        }

        // Compute reference BoW similarity score
        // This is the lowest score to a connected keyframe in the covisibility graph
        // We will impose loop candidates to have a higher similarity than this
        // Step 3：遍历当前回环关键帧所有连接（>15个共视地图点）关键帧,也就是共视关键帧，
        // 计算当前关键帧与每个共视关键的bow相似度得分，并得到最低得分minScore
        const vector<KeyFrame*> vpConnectedKeyFrames = mpCurrentKF->GetVectorCovisibleKeyFrames();
        const DBoW2::BowVector &CurrentBowVec = mpCurrentKF->mBowVec;
        float minScore = 1;
        for(size_t i=0; i<vpConnectedKeyFrames.size(); i++)
        {
            KeyFrame* pKF = vpConnectedKeyFrames[i];
            if(pKF->isBad())
                continue;
            const DBoW2::BowVector &BowVec = pKF->mBowVec;
            // 计算两个关键帧的相似度得分；得分越低,相似度越低
            float score = mpORBVocabulary->score(CurrentBowVec, BowVec);
            // 更新最低得分
            if(score<minScore)
                minScore = score;
        }

        // Query the database imposing the minimum score
        // Step 4：在所有关键帧中找出闭环候选帧（注意不和当前帧连接）
        // minScore的作用：认为和当前关键帧具有回环关系的关键帧,不应该低于当前关键帧的相邻关键帧的最低的相似度minScore
        // 得到的这些关键帧,和当前关键帧具有较多的公共单词,并且相似度评分都挺高
        vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectLoopCandidates(mpCurrentKF, minScore);

        // If there are no loop candidates, just add new keyframe and return false
        // 如果没有闭环候选帧，返回false
        if(vpCandidateKFs.empty())
        {
            mpKeyFrameDB->add(mpCurrentKF);
            mvConsistentGroups.clear();
            mpCurrentKF->SetErase();
            return false;
        }

        // For each loop candidate check consistency with previous loop candidates
        // Each candidate expands a covisibility group (keyframes connected to the loop candidate in the covisibility graph)
        // A group is consistent with a previous group if they share at least a keyframe
        // We must detect a consistent loop in several consecutive keyframes to accept it
        // Step 5：在候选帧中检测具有连续性的候选帧
        // 1、每个候选帧将与自己相连的关键帧构成一个“子候选组spCandidateGroup”， vpCandidateKFs-->spCandidateGroup
        // 2、检测“子候选组”中每一个关键帧是否存在于“连续组”，如果存在 nCurrentConsistency++，则将该“子候选组”放入“当前连续组vCurrentConsistentGroups”
        // 3、如果nCurrentConsistency大于等于3，那么该”子候选组“代表的候选帧过关，进入mvpEnoughConsistentCandidates

        // 相关的概念说明:（为方便理解，见视频里的图示）
        // 组(group): 对于某个关键帧, 其和其具有共视关系的关键帧组成了一个"组";
        // 子候选组(CandidateGroup): 对于某个候选的回环关键帧, 其和其具有共视关系的关键帧组成的一个"组";
        // 连续(Consistent):  不同的组之间如果共同拥有一个及以上的关键帧,那么称这两个组之间具有连续关系
        // 连续性(Consistency):称之为连续长度可能更合适,表示累计的连续的链的长度:A--B 为1, A--B--C--D 为3等;具体反映在数据类型 ConsistentGroup.second上
        // 连续组(Consistent group): mvConsistentGroups存储了上次执行回环检测时, 新的被检测出来的具有连续性的多个组的集合.由于组之间的连续关系是个网状结构,因此可能存在
        //                          一个组因为和不同的连续组链都具有连续关系,而被添加两次的情况(当然连续性度量是不相同的)
        // 连续组链:自造的称呼,类似于菊花链A--B--C--D这样形成了一条连续组链.对于这个例子中,由于可能E,F都和D有连续关系,因此连续组链会产生分叉;为了简化计算,连续组中将只会保存
        //         最后形成连续关系的连续组们(见下面的连续组的更新)
        // 子连续组: 上面的连续组中的一个组
        // 连续组的初始值: 在遍历某个候选帧的过程中,如果该子候选组没有能够和任何一个上次的子连续组产生连续关系,那么就将添加自己组为连续组,并且连续性为0(相当于新开了一个连续链)
        // 连续组的更新: 当前次回环检测过程中,所有被检测到和之前的连续组链有连续的关系的组,都将在对应的连续组链后面+1,这些子候选组(可能有重复,见上)都将会成为新的连续组;
        //              换而言之连续组mvConsistentGroups中只保存连续组链中末尾的组

        // 最终筛选后得到的闭环帧

        mvpEnoughConsistentCandidates.clear();

        // 这个下标是每个"子连续组"的下标,bool表示当前的候选组中是否有和该组相同的一个关键帧
        vector<ConsistentGroup> vCurrentConsistentGroups;
        vector<bool> vbConsistentGroup(mvConsistentGroups.size(),false);
        for(size_t i=0, iend=vpCandidateKFs.size(); i<iend; i++)
        {

            KeyFrame* pCandidateKF = vpCandidateKFs[i];
            // Step 5.2：将自己以及与自己相连的关键帧构成一个“子候选组”
            set<KeyFrame*> spCandidateGroup = pCandidateKF->GetConnectedKeyFrames();
            // 把自己也加进去
            spCandidateGroup.insert(pCandidateKF);

            // 连续性达标的标志
            bool bEnoughConsistent = false;
            bool bConsistentForSomeGroup = false;
            // Step 5.3：遍历前一次闭环检测到的连续组链
            // 上一次闭环的连续组链 std::vector<ConsistentGroup> mvConsistentGroups
            // 其中ConsistentGroup的定义：typedef pair<set<KeyFrame*>,int> ConsistentGroup
            // 其中 ConsistentGroup.first对应每个“连续组”中的关键帧集合，ConsistentGroup.second为每个“连续组”的连续长度
            for(size_t iG=0, iendG=mvConsistentGroups.size(); iG<iendG; iG++)
            {
                set<KeyFrame*> sPreviousGroup = mvConsistentGroups[iG].first;

                bool bConsistent = false;
                for(set<KeyFrame*>::iterator sit=spCandidateGroup.begin(), send=spCandidateGroup.end(); sit!=send;sit++)
                {
                    if(sPreviousGroup.count(*sit))
                    {
                        bConsistent=true;
                        bConsistentForSomeGroup=true;
                        break;
                    }
                }

                if(bConsistent)
                {
                    int nPreviousConsistency = mvConsistentGroups[iG].second;
                    int nCurrentConsistency = nPreviousConsistency + 1;
                    if(!vbConsistentGroup[iG])
                    {
                        ConsistentGroup cg = make_pair(spCandidateGroup,nCurrentConsistency);
                        vCurrentConsistentGroups.push_back(cg);
                        vbConsistentGroup[iG]=true; //this avoid to include the same group more than once
                    }
                    if(nCurrentConsistency>=mnCovisibilityConsistencyTh && !bEnoughConsistent)
                    {
                        mvpEnoughConsistentCandidates.push_back(pCandidateKF);
                        bEnoughConsistent=true; //this avoid to insert the same candidate more than once
                    }
                }
            }

            // If the group is not consistent with any previous group insert with consistency counter set to zero
            if(!bConsistentForSomeGroup)
            {
                ConsistentGroup cg = make_pair(spCandidateGroup,0);
                vCurrentConsistentGroups.push_back(cg);
            }
        }

        // Update Covisibility Consistent Groups
        mvConsistentGroups = vCurrentConsistentGroups;


        // Add Current Keyframe to database
        mpKeyFrameDB->add(mpCurrentKF);

        if(mvpEnoughConsistentCandidates.empty())
        {
            mpCurrentKF->SetErase();
            return false;
        }
        else
        {
            return true;
        }

        mpCurrentKF->SetErase();
        return false;
    }

    bool LoopClosing::ComputeSim3()
    {
        // For each consistent loop candidate we try to compute a Sim3

        const int nInitialCandidates = mvpEnoughConsistentCandidates.size();

        // We compute first ORB matches for each candidate
        // If enough matches are found, we setup a Sim3Solver
        ORBmatcher matcher(0.75,true);

        vector<Sim3Solver*> vpSim3Solvers;
        vpSim3Solvers.resize(nInitialCandidates);

        vector<vector<MapPoint*> > vvpMapPointMatches;
        vvpMapPointMatches.resize(nInitialCandidates);

        vector<bool> vbDiscarded;
        vbDiscarded.resize(nInitialCandidates);

        int nCandidates=0; //candidates with enough matches

        for(int i=0; i<nInitialCandidates; i++)
        {
            KeyFrame* pKF = mvpEnoughConsistentCandidates[i];

            // avoid that local mapping erase it while it is being processed in this thread
            pKF->SetNotErase();

            if(pKF->isBad())
            {
                vbDiscarded[i] = true;
                continue;
            }

            int nmatches = matcher.SearchByBoW(mpCurrentKF,pKF,vvpMapPointMatches[i]);

            if(nmatches<20)
            {
                vbDiscarded[i] = true;
                continue;
            }
            else
            {
                Sim3Solver* pSolver = new Sim3Solver(mpCurrentKF,pKF,vvpMapPointMatches[i],mbFixScale);
                pSolver->SetRansacParameters(0.99,20,300);
                vpSim3Solvers[i] = pSolver;
            }

            nCandidates++;
        }

        bool bMatch = false;

        // Perform alternatively RANSAC iterations for each candidate
        // until one is succesful or all fail
        while(nCandidates>0 && !bMatch)
        {
            for(int i=0; i<nInitialCandidates; i++)
            {
                if(vbDiscarded[i])
                    continue;

                KeyFrame* pKF = mvpEnoughConsistentCandidates[i];

                // Perform 5 Ransac Iterations
                vector<bool> vbInliers;
                int nInliers;
                bool bNoMore;

                Sim3Solver* pSolver = vpSim3Solvers[i];
                cv::Mat Scm  = pSolver->iterate(5,bNoMore,vbInliers,nInliers);

                // If Ransac reachs max. iterations discard keyframe
                if(bNoMore)
                {
                    vbDiscarded[i]=true;
                    nCandidates--;
                }

                // If RANSAC returns a Sim3, perform a guided matching and optimize with all correspondences
                if(!Scm.empty())
                {
                    vector<MapPoint*> vpMapPointMatches(vvpMapPointMatches[i].size(), static_cast<MapPoint*>(NULL));
                    for(size_t j=0, jend=vbInliers.size(); j<jend; j++)
                    {
                        if(vbInliers[j])
                            vpMapPointMatches[j]=vvpMapPointMatches[i][j];
                    }

                    cv::Mat R = pSolver->GetEstimatedRotation();
                    cv::Mat t = pSolver->GetEstimatedTranslation();
                    const float s = pSolver->GetEstimatedScale();
                    matcher.SearchBySim3(mpCurrentKF,pKF,vpMapPointMatches,s,R,t,7.5);

                    g2o::Sim3 gScm(Converter::toMatrix3d(R),Converter::toVector3d(t),s);
                    const int nInliers = Optimizer::OptimizeSim3(mpCurrentKF, pKF, vpMapPointMatches, gScm, 10, mbFixScale);

                    // If optimization is succesful stop ransacs and continue
                    if(nInliers>=20)
                    {
                        bMatch = true;
                        mpMatchedKF = pKF;
                        g2o::Sim3 gSmw(Converter::toMatrix3d(pKF->GetRotation()),Converter::toVector3d(pKF->GetTranslation()),1.0);
                        mg2oScw = gScm*gSmw;
                        mScw = Converter::toCvMat(mg2oScw);

                        mvpCurrentMatchedPoints = vpMapPointMatches;
                        break;
                    }
                }
            }
        }

        if(!bMatch)
        {
            for(int i=0; i<nInitialCandidates; i++)
                mvpEnoughConsistentCandidates[i]->SetErase();
            mpCurrentKF->SetErase();
            return false;
        }

        // Retrieve MapPoints seen in Loop Keyframe and neighbors
        vector<KeyFrame*> vpLoopConnectedKFs = mpMatchedKF->GetVectorCovisibleKeyFrames();
        vpLoopConnectedKFs.push_back(mpMatchedKF);
        mvpLoopMapPoints.clear();
        for(vector<KeyFrame*>::iterator vit=vpLoopConnectedKFs.begin(); vit!=vpLoopConnectedKFs.end(); vit++)
        {
            KeyFrame* pKF = *vit;
            vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();
            for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
            {
                MapPoint* pMP = vpMapPoints[i];
                if(pMP)
                {
                    if(!pMP->isBad() && pMP->mnLoopPointForKF!=mpCurrentKF->mnId)
                    {
                        mvpLoopMapPoints.push_back(pMP);
                        pMP->mnLoopPointForKF=mpCurrentKF->mnId;
                    }
                }
            }
        }

        // Find more matches projecting with the computed Sim3
        matcher.SearchByProjection(mpCurrentKF, mScw, mvpLoopMapPoints, mvpCurrentMatchedPoints,10);

        // If enough matches accept Loop
        int nTotalMatches = 0;
        for(size_t i=0; i<mvpCurrentMatchedPoints.size(); i++)
        {
            if(mvpCurrentMatchedPoints[i])
                nTotalMatches++;
        }

        if(nTotalMatches>=40)
        {
            for(int i=0; i<nInitialCandidates; i++)
                if(mvpEnoughConsistentCandidates[i]!=mpMatchedKF)
                    mvpEnoughConsistentCandidates[i]->SetErase();
            return true;
        }
        else
        {
            for(int i=0; i<nInitialCandidates; i++)
                mvpEnoughConsistentCandidates[i]->SetErase();
            mpCurrentKF->SetErase();
            return false;
        }

    }

    void LoopClosing::CorrectLoop()
    {
        cout << "Loop detected!" << endl;

        // Send a stop signal to Local Mapping
        // Avoid new keyframes are inserted while correcting the loop
        mpLocalMapper->RequestStop();

        // If a Global Bundle Adjustment is running, abort it
        if(isRunningGBA())
        {
            unique_lock<mutex> lock(mMutexGBA);
            mbStopGBA = true;

            mnFullBAIdx++;

            if(mpThreadGBA)
            {
                mpThreadGBA->detach();
                delete mpThreadGBA;
            }
        }

        // Wait until Local Mapping has effectively stopped
        while(!mpLocalMapper->isStopped())
        {
            usleep(1000);
        }

        // Ensure current keyframe is updated
        mpCurrentKF->UpdateConnections();

        // Retrive keyframes connected to the current keyframe and compute corrected Sim3 pose by propagation
        mvpCurrentConnectedKFs = mpCurrentKF->GetVectorCovisibleKeyFrames();
        mvpCurrentConnectedKFs.push_back(mpCurrentKF);

        KeyFrameAndPose CorrectedSim3, NonCorrectedSim3;
        CorrectedSim3[mpCurrentKF]=mg2oScw;
        cv::Mat Twc = mpCurrentKF->GetPoseInverse();


        {
            // Get Map Mutex
            unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

            for(vector<KeyFrame*>::iterator vit=mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
            {
                KeyFrame* pKFi = *vit;

                cv::Mat Tiw = pKFi->GetPose();

                if(pKFi!=mpCurrentKF)
                {
                    cv::Mat Tic = Tiw*Twc;
                    cv::Mat Ric = Tic.rowRange(0,3).colRange(0,3);
                    cv::Mat tic = Tic.rowRange(0,3).col(3);
                    g2o::Sim3 g2oSic(Converter::toMatrix3d(Ric),Converter::toVector3d(tic),1.0);
                    g2o::Sim3 g2oCorrectedSiw = g2oSic*mg2oScw;
                    //Pose corrected with the Sim3 of the loop closure
                    CorrectedSim3[pKFi]=g2oCorrectedSiw;
                }

                cv::Mat Riw = Tiw.rowRange(0,3).colRange(0,3);
                cv::Mat tiw = Tiw.rowRange(0,3).col(3);
                g2o::Sim3 g2oSiw(Converter::toMatrix3d(Riw),Converter::toVector3d(tiw),1.0);
                //Pose without correction
                NonCorrectedSim3[pKFi]=g2oSiw;
            }

            // Correct all MapPoints obsrved by current keyframe and neighbors, so that they align with the other side of the loop
            for(KeyFrameAndPose::iterator mit=CorrectedSim3.begin(), mend=CorrectedSim3.end(); mit!=mend; mit++)
            {
                KeyFrame* pKFi = mit->first;
                g2o::Sim3 g2oCorrectedSiw = mit->second;
                g2o::Sim3 g2oCorrectedSwi = g2oCorrectedSiw.inverse();

                g2o::Sim3 g2oSiw =NonCorrectedSim3[pKFi];

                vector<MapPoint*> vpMPsi = pKFi->GetMapPointMatches();
                for(size_t iMP=0, endMPi = vpMPsi.size(); iMP<endMPi; iMP++)
                {
                    MapPoint* pMPi = vpMPsi[iMP];
                    if(!pMPi)
                        continue;
                    if(pMPi->isBad())
                        continue;
                    if(pMPi->mnCorrectedByKF==mpCurrentKF->mnId)
                        continue;

                    // Project with non-corrected pose and project back with corrected pose
                    cv::Mat P3Dw = pMPi->GetWorldPos();
                    Eigen::Matrix<double,3,1> eigP3Dw = Converter::toVector3d(P3Dw);
                    Eigen::Matrix<double,3,1> eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oSiw.map(eigP3Dw));

                    cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
                    pMPi->SetWorldPos(cvCorrectedP3Dw);
                    pMPi->mnCorrectedByKF = mpCurrentKF->mnId;
                    pMPi->mnCorrectedReference = pKFi->mnId;
                    pMPi->UpdateNormalAndDepth();
                }

                // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
                Eigen::Matrix3d eigR = g2oCorrectedSiw.rotation().toRotationMatrix();
                Eigen::Vector3d eigt = g2oCorrectedSiw.translation();
                double s = g2oCorrectedSiw.scale();

                eigt *=(1./s); //[R t/s;0 1]

                cv::Mat correctedTiw = Converter::toCvSE3(eigR,eigt);

                pKFi->SetPose(correctedTiw);

                // Make sure connections are updated
                pKFi->UpdateConnections();
            }

            // Start Loop Fusion
            // Update matched map points and replace if duplicated
            for(size_t i=0; i<mvpCurrentMatchedPoints.size(); i++)
            {
                if(mvpCurrentMatchedPoints[i])
                {
                    MapPoint* pLoopMP = mvpCurrentMatchedPoints[i];
                    MapPoint* pCurMP = mpCurrentKF->GetMapPoint(i);
                    if(pCurMP)
                        pCurMP->Replace(pLoopMP);
                    else
                    {
                        mpCurrentKF->AddMapPoint(pLoopMP,i);
                        pLoopMP->AddObservation(mpCurrentKF,i);
                        pLoopMP->ComputeDistinctiveDescriptors();
                    }
                }
            }

        }

        // Project MapPoints observed in the neighborhood of the loop keyframe
        // into the current keyframe and neighbors using corrected poses.
        // Fuse duplications.
        SearchAndFuse(CorrectedSim3);


        // After the MapPoint fusion, new links in the covisibility graph will appear attaching both sides of the loop
        map<KeyFrame*, set<KeyFrame*> > LoopConnections;

        for(vector<KeyFrame*>::iterator vit=mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKFi = *vit;
            vector<KeyFrame*> vpPreviousNeighbors = pKFi->GetVectorCovisibleKeyFrames();

            // Update connections. Detect new links.
            pKFi->UpdateConnections();
            LoopConnections[pKFi]=pKFi->GetConnectedKeyFrames();
            for(vector<KeyFrame*>::iterator vit_prev=vpPreviousNeighbors.begin(), vend_prev=vpPreviousNeighbors.end(); vit_prev!=vend_prev; vit_prev++)
            {
                LoopConnections[pKFi].erase(*vit_prev);
            }
            for(vector<KeyFrame*>::iterator vit2=mvpCurrentConnectedKFs.begin(), vend2=mvpCurrentConnectedKFs.end(); vit2!=vend2; vit2++)
            {
                LoopConnections[pKFi].erase(*vit2);
            }
        }

        // Optimize graph
        Optimizer::OptimizeEssentialGraph(mpMap, mpMatchedKF, mpCurrentKF, NonCorrectedSim3, CorrectedSim3, LoopConnections, mbFixScale);

        mpMap->InformNewBigChange();

        // Add loop edge
        mpMatchedKF->AddLoopEdge(mpCurrentKF);
        mpCurrentKF->AddLoopEdge(mpMatchedKF);

        // Launch a new thread to perform Global Bundle Adjustment
        mbRunningGBA = true;
        mbFinishedGBA = false;
        mbStopGBA = false;
        mpThreadGBA = new thread(&LoopClosing::RunGlobalBundleAdjustment,this,mpCurrentKF->mnId);

        // Loop closed. Release Local Mapping.
        mpLocalMapper->Release();

        mLastLoopKFid = mpCurrentKF->mnId;
    }

    void LoopClosing::SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap)
    {
        ORBmatcher matcher(0.8);

        for(KeyFrameAndPose::const_iterator mit=CorrectedPosesMap.begin(), mend=CorrectedPosesMap.end(); mit!=mend;mit++)
        {
            KeyFrame* pKF = mit->first;

            g2o::Sim3 g2oScw = mit->second;
            cv::Mat cvScw = Converter::toCvMat(g2oScw);

            vector<MapPoint*> vpReplacePoints(mvpLoopMapPoints.size(),static_cast<MapPoint*>(NULL));
            matcher.Fuse(pKF,cvScw,mvpLoopMapPoints,4,vpReplacePoints);

            // Get Map Mutex
            unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
            const int nLP = mvpLoopMapPoints.size();
            for(int i=0; i<nLP;i++)
            {
                MapPoint* pRep = vpReplacePoints[i];
                if(pRep)
                {
                    pRep->Replace(mvpLoopMapPoints[i]);
                }
            }
        }
    }


    void LoopClosing::RequestReset()
    {
        {
            unique_lock<mutex> lock(mMutexReset);
            mbResetRequested = true;
        }

        while(1)
        {
            {
                unique_lock<mutex> lock2(mMutexReset);
                if(!mbResetRequested)
                    break;
            }
            usleep(5000);
        }
    }

    void LoopClosing::ResetIfRequested()
    {
        unique_lock<mutex> lock(mMutexReset);
        if(mbResetRequested)
        {
            mlpLoopKeyFrameQueue.clear();
            mLastLoopKFid=0;
            mbResetRequested=false;
        }
    }

    void LoopClosing::RunGlobalBundleAdjustment(unsigned long nLoopKF)
    {
        cout << "Starting Global Bundle Adjustment" << endl;

        int idx =  mnFullBAIdx;
        Optimizer::GlobalBundleAdjustemnt(mpMap,10,&mbStopGBA,nLoopKF,false);

        // Update all MapPoints and KeyFrames
        // Local Mapping was active during BA, that means that there might be new keyframes
        // not included in the Global BA and they are not consistent with the updated map.
        // We need to propagate the correction through the spanning tree
        {
            unique_lock<mutex> lock(mMutexGBA);
            if(idx!=mnFullBAIdx)
                return;

            if(!mbStopGBA)
            {
                cout << "Global Bundle Adjustment finished" << endl;
                cout << "Updating map ..." << endl;
                mpLocalMapper->RequestStop();
                // Wait until Local Mapping has effectively stopped

                while(!mpLocalMapper->isStopped() && !mpLocalMapper->isFinished())
                {
                    usleep(1000);
                }

                // Get Map Mutex
                unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

                // Correct keyframes starting at map first keyframe
                list<KeyFrame*> lpKFtoCheck(mpMap->mvpKeyFrameOrigins.begin(),mpMap->mvpKeyFrameOrigins.end());

                while(!lpKFtoCheck.empty())
                {
                    KeyFrame* pKF = lpKFtoCheck.front();
                    const set<KeyFrame*> sChilds = pKF->GetChilds();
                    cv::Mat Twc = pKF->GetPoseInverse();
                    for(set<KeyFrame*>::const_iterator sit=sChilds.begin();sit!=sChilds.end();sit++)
                    {
                        KeyFrame* pChild = *sit;
                        if(pChild->mnBAGlobalForKF!=nLoopKF)
                        {
                            cv::Mat Tchildc = pChild->GetPose()*Twc;
                            pChild->mTcwGBA = Tchildc*pKF->mTcwGBA;//*Tcorc*pKF->mTcwGBA;
                            pChild->mnBAGlobalForKF=nLoopKF;

                        }
                        lpKFtoCheck.push_back(pChild);
                    }

                    pKF->mTcwBefGBA = pKF->GetPose();
                    pKF->SetPose(pKF->mTcwGBA);
                    lpKFtoCheck.pop_front();
                }

                // Correct MapPoints
                const vector<MapPoint*> vpMPs = mpMap->GetAllMapPoints();

                for(size_t i=0; i<vpMPs.size(); i++)
                {
                    MapPoint* pMP = vpMPs[i];

                    if(pMP->isBad())
                        continue;

                    if(pMP->mnBAGlobalForKF==nLoopKF)
                    {
                        // If optimized by Global BA, just update
                        pMP->SetWorldPos(pMP->mPosGBA);
                    }
                    else
                    {
                        // Update according to the correction of its reference keyframe
                        KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();

                        if(pRefKF->mnBAGlobalForKF!=nLoopKF)
                            continue;

                        // Map to non-corrected camera
                        cv::Mat Rcw = pRefKF->mTcwBefGBA.rowRange(0,3).colRange(0,3);
                        cv::Mat tcw = pRefKF->mTcwBefGBA.rowRange(0,3).col(3);
                        cv::Mat Xc = Rcw*pMP->GetWorldPos()+tcw;

                        // Backproject using corrected camera
                        cv::Mat Twc = pRefKF->GetPoseInverse();
                        cv::Mat Rwc = Twc.rowRange(0,3).colRange(0,3);
                        cv::Mat twc = Twc.rowRange(0,3).col(3);

                        pMP->SetWorldPos(Rwc*Xc+twc);
                    }
                }

                mpMap->InformNewBigChange();

                mpLocalMapper->Release();

                cout << "Map updated!" << endl;


            }

            mbFinishedGBA = true;
            mbRunningGBA = false;
        }
    }

    void LoopClosing::RequestFinish()
    {
        unique_lock<mutex> lock(mMutexFinish);
        mbFinishRequested = true;
    }

    bool LoopClosing::CheckFinish()
    {
        unique_lock<mutex> lock(mMutexFinish);
        return mbFinishRequested;
    }

    void LoopClosing::SetFinish()
    {
        unique_lock<mutex> lock(mMutexFinish);
        mbFinished = true;
    }

    bool LoopClosing::isFinished()
    {
        unique_lock<mutex> lock(mMutexFinish);
        return mbFinished;
    }


} //namespace Planar_SLAM
