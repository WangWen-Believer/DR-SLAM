#include "MapDrawer.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include <pangolin/pangolin.h>
#include <mutex>

using namespace std;
using namespace cv;
using namespace cv::line_descriptor;
using namespace Eigen;

namespace Planar_SLAM
{


    MapDrawer::MapDrawer(Map* pMap, const string &strSettingPath):mpMap(pMap),
                                                                  m_octree(NULL),
                                                                  m_maxRange(-1.0),
                                                                  m_useHeightMap(true),
                                                                  m_res(0.10),
                                                                  m_colorFactor(0.8),
                                                                  m_treeDepth(0),
                                                                  m_maxTreeDepth(0),
                                                                  nGBA(0)
    {
        cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

        mKeyFrameSize = fSettings["Viewer.KeyFrameSize"];
        mKeyFrameLineWidth = fSettings["Viewer.KeyFrameLineWidth"];
        mGraphLineWidth = fSettings["Viewer.GraphLineWidth"];
        mPointSize = fSettings["Viewer.PointSize"];
        mCameraSize = fSettings["Viewer.CameraSize"];
        mCameraLineWidth = fSettings["Viewer.CameraLineWidth"];
        mLineWidth = fSettings["Viewer.LineWidth"];

        fx = fSettings["Camera.fx"];
        fy = fSettings["Camera.fy"];
        cx = fSettings["Camera.cx"];
        cy = fSettings["Camera.cy"];

        m_octree = new octomap::ColorOcTree(m_res);//分辨率
        // initialize octomap
        m_octree->setClampingThresMin(0.0);
        m_octree->setClampingThresMax(1.0);
        m_octree->setProbHit(0.7);
        m_octree->setProbMiss(0.4);

        m_treeDepth = m_octree->getTreeDepth();
        m_maxTreeDepth = m_treeDepth;

        string filepath = fSettings["Octo.mapPath"];
        boost::filesystem::path octmapPath(filepath.c_str());

        if(boost::filesystem::exists(octmapPath))
        {
            bIsLocalization = true;
            LoadOctoMap(filepath);
        }
        else
        {
            bIsLocalization = false;
        }
    }

    void MapDrawer::DrawMapPoints()
    {
        const vector<MapPoint*> &vpMPs = mpMap->GetAllMapPoints();
        const vector<MapPoint*> &vpRefMPs = mpMap->GetReferenceMapPoints();

        set<MapPoint*> spRefMPs(vpRefMPs.begin(), vpRefMPs.end());

        if(vpMPs.empty())
            return;

        glPointSize(mPointSize);
        glBegin(GL_POINTS);
        glColor3f(0.0,0.0,0.0);// 黑色

        for(size_t i=0, iend=vpMPs.size(); i<iend;i++)
        {
            if(vpMPs[i]->isBad() || spRefMPs.count(vpMPs[i]))
                continue;
            cv::Mat pos = vpMPs[i]->GetWorldPos();
            glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
        }
        glEnd();

        glPointSize(mPointSize);
        glBegin(GL_POINTS);
        glColor3f(0.0,0.0,0.0);     //红色

        for(set<MapPoint*>::iterator sit=spRefMPs.begin(), send=spRefMPs.end(); sit!=send; sit++)
        {
            if((*sit)->isBad())
                continue;
            cv::Mat pos = (*sit)->GetWorldPos();
            glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
        }

        glEnd();
    }

    void MapDrawer::DrawMapLines()
    {
        const vector<MapLine*> &vpMLs = mpMap->GetAllMapLines();
        const vector<MapLine*> &vpRefMLs = mpMap->GetReferenceMapLines();

        set<MapLine*> spRefMLs(vpRefMLs.begin(), vpRefMLs.end());

        if(vpMLs.empty())
            return;

        glLineWidth(mLineWidth);
        glBegin ( GL_LINES );
        glColor3f(0.0,0.0,0.0);

        for(size_t i=0, iend=vpMLs.size(); i<iend; i++)
        {
            if(vpMLs[i]->isBad() || spRefMLs.count(vpMLs[i]))
                continue;
            Vector6d pos = vpMLs[i]->GetWorldPos();
            glVertex3f(pos(0), pos(1), pos(2));
            glVertex3f(pos(3), pos(4), pos(5));

        }
        glEnd();

        glLineWidth(mLineWidth);
        glBegin ( GL_LINES );
        glColor3f(0.0,0.0,0.0); //红色

        for(set<MapLine*>::iterator sit=spRefMLs.begin(), send=spRefMLs.end(); sit!=send; sit++)
        {
            if((*sit)->isBad())
                continue;
            Vector6d pos = (*sit)->GetWorldPos();
            glVertex3f(pos(0), pos(1), pos(2));
            glVertex3f(pos(3), pos(4), pos(5));
        }
        glEnd();
    }

    void MapDrawer::DrawMapPlanes() {
        const vector<MapPlane*> &vpMPs = mpMap->GetAllMapPlanes();
        if(vpMPs.empty())
            return;
        glPointSize(mPointSize*2);
        glBegin(GL_POINTS);

        for(auto pMP : vpMPs){
            float ir = pMP->mRed;
            float ig = pMP->mGreen;
            float ib = pMP->mBlue;
            float norm = sqrt(ir*ir + ig*ig + ib*ib);
            glColor3f(ir/norm, ig/norm, ib/norm);

            cv::Mat pos(4,1,CV_32F);
            for(auto& p : pMP->mvPlanePoints.get()->points){
                pos.at<float>(0) = p.x;
                pos.at<float>(1) = p.y;
                pos.at<float>(2) = p.z;
                pos.at<float>(3) = 1;

                glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
            }
        }
        glEnd();
    }

    void MapDrawer::DrawMapNormal(){
        SurfaceNormal_M vpSNMs = mpMap->GetSurfaceNormal_Manhattan();

        if(vpSNMs.mvpsurfacenormal.empty())
            return;
        glPointSize(mPointSize);
        glBegin(GL_POINTS);

        for(size_t i=0,iend=vpSNMs.mvpsurfacenormal.size();i<iend;i++)
        {
            if(vpSNMs.bsurfacenormal_inline[i] == false)
                glColor3f(0.0,0.0,0.0);// 黑色
            else
                glColor3f(1.0,0.0,0.7);

            cv::Point3f pos = vpSNMs.mvpsurfacenormal[i].normal;
            glVertex3f(pos.x,pos.y,pos.z);
        }
        glEnd();
    }
    void MapDrawer::DrawManhattan(){
        cv::Mat R = mpMap->GetRmc();
        static cv::Mat R_init;
        static bool Initalized = false;
        if(R.data)
        {

            if(!Initalized)
            {
                Initalized = true;
                cout << "wangwen"<< endl;
                R_init = mpMap->GetRmc();
                cout << "Rinit" << R_init<<endl;
            }
//                glLineWidth(2);//设置线宽
//                glBegin(GL_LINES);//2点为1条线，级数点无效
//                glColor3f(0.0,0.0,1.0);
//                glVertex3f(0,0,0);glVertex3f(R_init.at<float>(0,0),R_init.at<float>(1,0),R_init.at<float>(2,0));
//                glColor3f(0.0,1.0,0.0);
//                glVertex3f(0,0,0);glVertex3f(R_init.at<float>(0,1),R_init.at<float>(1,1),R_init.at<float>(2,1));
//                glColor3f(1.0,0.0,0.0);
//                glVertex3f(0,0,0);glVertex3f(R_init.at<float>(0,2),R_init.at<float>(1,2),R_init.at<float>(2,2));
//                glEnd();
                glLineWidth(5);//设置线宽
                glBegin(GL_LINES);//2点为1条线，级数点无效
                glColor3f(0.8,0.2,1);
                glColor3f(0.0,0.0,1.0);
                glVertex3f(0,0,0);glVertex3f(R.at<float>(0,0),R.at<float>(1,0),R.at<float>(2,0));
                glColor3f(0.0,1.0,0.0);
                glVertex3f(0,0,0);glVertex3f(R.at<float>(0,1),R.at<float>(1,1),R.at<float>(2,1));
                glColor3f(1.0,0.0,0.0);
                glVertex3f(0,0,0);glVertex3f(R.at<float>(0,2),R.at<float>(1,2),R.at<float>(2,2));
                glEnd();

        }

    }

    void MapDrawer::DrawSphere() {
        int du = 30,dv=30;
        float gu = 3.1415926535 / du , gv = 2.0*3.1415926535 / (dv-1);
        float x, y, z ,r ;
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
        glPushMatrix();
        #define MAKE_R(z2) sqrt(1.0 - (z2))
        #define MAKE_Z(z_step) cos((z_step) *gu)
        #define MAKE_Y(x_step,r) (r)*sin((x_step)* gv)
        #define MAKE_X(x_step,r) (r)*cos((x_step)* gv)
        glBegin(GL_QUADS);
        glColor3f(.6f, .05f, .95f);
        for (int z_step = 0 ; z_step < du; ++z_step) {
            for (int x_step = 0; x_step < dv; ++x_step) {

                z = MAKE_Z(z_step);
                r = MAKE_R(z * z);
                x = MAKE_X(x_step, r);
                y = MAKE_Y(x_step, r);
                glVertex3f(x, z, y);

                z = MAKE_Z(z_step + 1);
                r = MAKE_R(z * z);
                x = MAKE_X(x_step, r);
                y = MAKE_Y(x_step, r);
                glVertex3f(x, z, y);

                z = MAKE_Z(z_step + 1);
                r = MAKE_R(z * z);
                x = MAKE_X(x_step + 1, r);
                y = MAKE_Y(x_step + 1, r);
                glVertex3f(x, z, y);

                z = MAKE_Z(z_step);
                r = MAKE_R(z * z);
                x = MAKE_X(x_step + 1, r);
                y = MAKE_Y(x_step + 1, r);
                glVertex3f(x, z, y);

            }
        }
        glEnd();
        #undef MAKE_Y
        #undef MAKE_Z
        #undef MAKE_X
        #undef MAKE_R
        glPopMatrix();
    }


    void MapDrawer::DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph)
    {
        const float &w = mKeyFrameSize;
        const float h = w*0.75;
        const float z = w*0.6;

        const vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();

        if(bDrawKF)
        {
            for(size_t i=0; i<vpKFs.size(); i++)
            {
                KeyFrame* pKF = vpKFs[i];
                cv::Mat Twc = pKF->GetPoseInverse().t();

                glPushMatrix();

                glMultMatrixf(Twc.ptr<GLfloat>(0));

                glLineWidth(mKeyFrameLineWidth);
                glColor3f(0.0f,0.0f,1.0f);
                glBegin(GL_LINES);
                glVertex3f(0,0,0);
                glVertex3f(w,h,z);
                glVertex3f(0,0,0);
                glVertex3f(w,-h,z);
                glVertex3f(0,0,0);
                glVertex3f(-w,-h,z);
                glVertex3f(0,0,0);
                glVertex3f(-w,h,z);

                glVertex3f(w,h,z);
                glVertex3f(w,-h,z);

                glVertex3f(-w,h,z);
                glVertex3f(-w,-h,z);

                glVertex3f(-w,h,z);
                glVertex3f(w,h,z);

                glVertex3f(-w,-h,z);
                glVertex3f(w,-h,z);
                glEnd();

                glPopMatrix();
            }
        }

        if(bDrawGraph)
        {
            glLineWidth(mGraphLineWidth);
            glColor4f(0.0f,1.0f,0.0f,0.6f);
            glBegin(GL_LINES);

            for(size_t i=0; i<vpKFs.size(); i++)
            {
                // Covisibility Graph
                const vector<KeyFrame*> vCovKFs = vpKFs[i]->GetCovisiblesByWeight(100);
                cv::Mat Ow = vpKFs[i]->GetCameraCenter();
                if(!vCovKFs.empty())
                {
                    for(vector<KeyFrame*>::const_iterator vit=vCovKFs.begin(), vend=vCovKFs.end(); vit!=vend; vit++)
                    {
                        if((*vit)->mnId<vpKFs[i]->mnId)
                            continue;
                        cv::Mat Ow2 = (*vit)->GetCameraCenter();
                        glVertex3f(Ow.at<float>(0),Ow.at<float>(1),Ow.at<float>(2));
                        glVertex3f(Ow2.at<float>(0),Ow2.at<float>(1),Ow2.at<float>(2));
                    }
                }

                // Spanning tree
                KeyFrame* pParent = vpKFs[i]->GetParent();
                if(pParent)
                {
                    cv::Mat Owp = pParent->GetCameraCenter();
                    glVertex3f(Ow.at<float>(0),Ow.at<float>(1),Ow.at<float>(2));
                    glVertex3f(Owp.at<float>(0),Owp.at<float>(1),Owp.at<float>(2));
                }

                // Loops
                set<KeyFrame*> sLoopKFs = vpKFs[i]->GetLoopEdges();
                for(set<KeyFrame*>::iterator sit=sLoopKFs.begin(), send=sLoopKFs.end(); sit!=send; sit++)
                {
                    if((*sit)->mnId<vpKFs[i]->mnId)
                        continue;
                    cv::Mat Owl = (*sit)->GetCameraCenter();
                    glVertex3f(Ow.at<float>(0),Ow.at<float>(1),Ow.at<float>(2));
                    glVertex3f(Owl.at<float>(0),Owl.at<float>(1),Owl.at<float>(2));
                }
            }

            glEnd();
        }
    }

    void MapDrawer::DrawCurrentCamera(pangolin::OpenGlMatrix &Twc)
    {
        const float &w = mCameraSize;
        const float h = w*0.75;
        const float z = w*0.6;

        glPushMatrix();

#ifdef HAVE_GLES
        glMultMatrixf(Twc.m);
#else
        glMultMatrixd(Twc.m);
#endif

        glLineWidth(mCameraLineWidth);
        glColor3f(0.0f,1.0f,0.0f);
        glBegin(GL_LINES);
        glVertex3f(0,0,0);
        glVertex3f(w,h,z);
        glVertex3f(0,0,0);
        glVertex3f(w,-h,z);
        glVertex3f(0,0,0);
        glVertex3f(-w,-h,z);
        glVertex3f(0,0,0);
        glVertex3f(-w,h,z);

        glVertex3f(w,h,z);
        glVertex3f(w,-h,z);

        glVertex3f(-w,h,z);
        glVertex3f(-w,-h,z);

        glVertex3f(-w,h,z);
        glVertex3f(w,h,z);

        glVertex3f(-w,-h,z);
        glVertex3f(w,-h,z);
        glEnd();

        glPopMatrix();
    }


    void MapDrawer::SetCurrentCameraPose(const cv::Mat &Tcw)
    {
        unique_lock<mutex> lock(mMutexCamera);
        mCameraPose = Tcw.clone();
    }

    void MapDrawer::GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M)
    {
        if(!mCameraPose.empty())
        {
            cv::Mat Rwc(3,3,CV_32F);
            cv::Mat twc(3,1,CV_32F);
            {
                unique_lock<mutex> lock(mMutexCamera);
                Rwc = mCameraPose.rowRange(0,3).colRange(0,3).t();
                twc = -Rwc*mCameraPose.rowRange(0,3).col(3);
            }

            M.m[0] = Rwc.at<float>(0,0);
            M.m[1] = Rwc.at<float>(1,0);
            M.m[2] = Rwc.at<float>(2,0);
            M.m[3]  = 0.0;

            M.m[4] = Rwc.at<float>(0,1);
            M.m[5] = Rwc.at<float>(1,1);
            M.m[6] = Rwc.at<float>(2,1);
            M.m[7]  = 0.0;

            M.m[8] = Rwc.at<float>(0,2);
            M.m[9] = Rwc.at<float>(1,2);
            M.m[10] = Rwc.at<float>(2,2);
            M.m[11]  = 0.0;

            M.m[12] = twc.at<float>(0);
            M.m[13] = twc.at<float>(1);
            M.m[14] = twc.at<float>(2);
            M.m[15]  = 1.0;
        }
        else
            M.SetIdentity();
    }
    /**************************/
    void MapDrawer::DrawOctoMap()
    {
        vector<KeyFrame*> vKFs= mpMap->GetAllKeyFrames();
        int N = vKFs.size();

        if(N==0)
        {
//            m_octree->clear();
            lastKeyframeSize = 0;
            return;
        }
        if(bIsLocalization == false)
        {
            int bigChange = mpMap->GetLastBigChangeIdx();
            if( nGBA!= bigChange)
            {
                nGBA = bigChange;
                m_octree->clear();
                lastKeyframeSize = 0;
            }

            UpdateOctomap(vKFs);
        }


        octomap::ColorOcTree::tree_iterator it  = m_octree->begin_tree();
        octomap::ColorOcTree::tree_iterator end = m_octree->end_tree();
        int counter = 0;
        double occ_thresh = 0.9;
        int level = 16;
        glClearColor(1.0f,1.0f,1.0f,1.0f);

        glDisable(GL_LIGHTING);
        glEnable (GL_BLEND);
        ////DRAW OCTOMAP BEGIN//////
        double stretch_factor = 128/(1 - occ_thresh); //occupancy range in which the displayed cubes can be
        for(; it != end; ++counter, ++it)
        {
            if(level != it.getDepth())
            {
                continue;
            }
            double occ = it->getOccupancy();
            if(occ < occ_thresh)
            {
                continue;
            }

            double minX, minY, minZ, maxX, maxY, maxZ;
            m_octree->getMetricMin(minX, minY, minZ);
            m_octree->getMetricMax(maxX, maxY, maxZ);

            float halfsize = it.getSize()/2.0;
            float x = it.getX();
            float y = it.getY();
            float z = it.getZ();
            double h = ( std::min(std::max((y-minY)/(maxY-minY), 0.0), 1.0))*0.8;
            double r, g, b;
            heightMapColor(h, r,g,b);
            glBegin(GL_TRIANGLES);
            //Front
            glColor3d(r, g, b);
            glVertex3f(x-halfsize,y-halfsize,z-halfsize);// - - - 1
            glVertex3f(x-halfsize,y+halfsize,z-halfsize);// - + - 2
            glVertex3f(x+halfsize,y+halfsize,z-halfsize);// + + -3

            glVertex3f(x-halfsize,y-halfsize,z-halfsize); // - - -
            glVertex3f(x+halfsize,y+halfsize,z-halfsize); // + + -
            glVertex3f(x+halfsize,y-halfsize,z-halfsize); // + - -4

            //Back
            glVertex3f(x-halfsize,y-halfsize,z+halfsize); // - - + 1
            glVertex3f(x+halfsize,y-halfsize,z+halfsize); // + - + 2
            glVertex3f(x+halfsize,y+halfsize,z+halfsize); // + + + 3

            glVertex3f(x-halfsize,y-halfsize,z+halfsize); // - - +
            glVertex3f(x+halfsize,y+halfsize,z+halfsize); // + + +
            glVertex3f(x-halfsize,y+halfsize,z+halfsize); // - + + 4

            //Left
            glVertex3f(x-halfsize,y-halfsize,z-halfsize); // - - - 1
            glVertex3f(x-halfsize,y-halfsize,z+halfsize); // - - + 2
            glVertex3f(x-halfsize,y+halfsize,z+halfsize); // - + + 3

            glVertex3f(x-halfsize,y-halfsize,z-halfsize); // - - -
            glVertex3f(x-halfsize,y+halfsize,z+halfsize); // - + +
            glVertex3f(x-halfsize,y+halfsize,z-halfsize); // - + - 4

            //Right
            glVertex3f(x+halfsize,y-halfsize,z-halfsize);
            glVertex3f(x+halfsize,y+halfsize,z-halfsize);
            glVertex3f(x+halfsize,y+halfsize,z+halfsize);

            glVertex3f(x+halfsize,y-halfsize,z-halfsize);
            glVertex3f(x+halfsize,y+halfsize,z+halfsize);
            glVertex3f(x+halfsize,y-halfsize,z+halfsize);

            //top
            glVertex3f(x-halfsize,y-halfsize,z-halfsize);
            glVertex3f(x+halfsize,y-halfsize,z-halfsize);
            glVertex3f(x+halfsize,y-halfsize,z+halfsize);

            glVertex3f(x-halfsize,y-halfsize,z-halfsize);
            glVertex3f(x+halfsize,y-halfsize,z+halfsize);
            glVertex3f(x-halfsize,y-halfsize,z+halfsize);

            //bottom
            glVertex3f(x-halfsize,y+halfsize,z-halfsize);
            glVertex3f(x-halfsize,y+halfsize,z+halfsize);
            glVertex3f(x+halfsize,y+halfsize,z+halfsize);

            glVertex3f(x-halfsize,y+halfsize,z-halfsize);
            glVertex3f(x+halfsize,y+halfsize,z+halfsize);
            glVertex3f(x+halfsize,y+halfsize,z-halfsize);
            glEnd();

            glBegin(GL_LINES);
            glColor3f(0,0,0);
            //
            glVertex3f(x-halfsize,y-halfsize,z-halfsize);// - - - 1
            glVertex3f(x-halfsize,y+halfsize,z-halfsize);

            glVertex3f(x-halfsize,y+halfsize,z-halfsize);// - + - 2
            glVertex3f(x+halfsize,y+halfsize,z-halfsize);// + + -3

            glVertex3f(x+halfsize,y+halfsize,z-halfsize);// + + -3
            glVertex3f(x+halfsize,y-halfsize,z-halfsize); // + - -4

            glVertex3f(x+halfsize,y-halfsize,z-halfsize); // + - -4
            glVertex3f(x-halfsize,y-halfsize,z-halfsize);// - - - 1


            // back

            glVertex3f(x-halfsize,y-halfsize,z+halfsize); // - - + 1
            glVertex3f(x+halfsize,y-halfsize,z+halfsize); // + - + 2

            glVertex3f(x+halfsize,y-halfsize,z+halfsize); // + - + 2
            glVertex3f(x+halfsize,y+halfsize,z+halfsize); // + + + 3

            glVertex3f(x+halfsize,y+halfsize,z+halfsize); // + + + 3
            glVertex3f(x-halfsize,y+halfsize,z+halfsize); // - + + 4

            glVertex3f(x-halfsize,y+halfsize,z+halfsize); // - + + 4
            glVertex3f(x-halfsize,y-halfsize,z+halfsize); // - - + 1

            // top
            glVertex3f(x+halfsize,y-halfsize,z-halfsize);
            glVertex3f(x+halfsize,y-halfsize,z+halfsize);

            glVertex3f(x-halfsize,y-halfsize,z+halfsize);
            glVertex3f(x-halfsize,y-halfsize,z-halfsize);

            // bottom

            glVertex3f(x-halfsize,y+halfsize,z+halfsize);
            glVertex3f(x+halfsize,y+halfsize,z+halfsize);

            glVertex3f(x-halfsize,y+halfsize,z-halfsize);
            glVertex3f(x+halfsize,y+halfsize,z-halfsize);
            glEnd();
        }
    }
    void MapDrawer::SaveOctoMap(const char *filename)
    {
        std::ofstream outfile(filename, std::ios_base::out | std::ios_base::binary);
        if (outfile.is_open())
        {
            m_octree->write(outfile);
            outfile.close();
        }
    }
    void MapDrawer::ReadOctoMap(string filename)
    {

        octomap::AbstractOcTree* tree = octomap::AbstractOcTree::read(filename);
        m_octree = dynamic_cast<octomap::ColorOcTree*>(tree);
    }
    void MapDrawer::DrawObs(pcl::PointCloud<pcl::PointXYZRGB> observation)
    {
        glPointSize(mPointSize);
        glBegin(GL_POINTS);
        glColor3f(1.0,0.0,0.0);

        for(int i=0; i<observation.points.size(); i++)
        {
            glVertex3f(observation.points[i].x,observation.points[i].y,observation.points[i].z);
        }
        glEnd();
    }
    void MapDrawer::GeneratePointCloud(KeyFrame *kf, pcl::PointCloud<pcl::PointXYZRGB> &ground, pcl::PointCloud<pcl::PointXYZRGB> &nonground)
    {
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
        // note: 如此遍历相当耗时的一件事情，不能这么做
        for ( int m=0; m<(kf->mImDep.rows); m+=4 )
        {
            for ( int n=0; n<(kf->mImDep.cols); n+=4 )
            {
                float d = kf->mImDep.ptr<float>(m)[n];///1000.0;
                if (d < 0.01 || d>5.0)
                    continue;
                pcl::PointXYZRGB p;
                p.z = d;
                p.x = ( n - cx) * p.z / fx;
                p.y = ( m - cy) * p.z / fy;
//            if(p.y<-3.0 || p.y>3.0) continue;
                p.b = kf->mImRGB.ptr<uchar>(m)[n*3];
                p.g = kf->mImRGB.ptr<uchar>(m)[n*3+1];
                p.r = kf->mImRGB.ptr<uchar>(m)[n*3+2];
                cloud->points.push_back( p );
            }
        }
        pcl::VoxelGrid<pcl::PointXYZRGB>vg;
        vg.setInputCloud(cloud);
        vg.setLeafSize(0.10,0.10, 0.10);
        vg.filter(*cloud);
        //STEP2: transform camera to word
        Eigen::Isometry3d T = Planar_SLAM::Converter::toSE3Quat( kf->GetPose() );
        pcl::PointCloud<pcl::PointXYZRGB> temp;
        pcl::transformPointCloud( *cloud, temp, T.inverse().matrix());

    // filter ground plane
    if(temp.size()<50)
    {
        printf("pointcloud too small skip ground plane extraction\n;");
        ground = temp;
    }
    else
    {
        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);

        pcl::SACSegmentation<pcl::PointCloud<pcl::PointXYZRGB>::PointType> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setMaxIterations(200);
        seg.setDistanceThreshold(0.04);
        seg.setAxis(Eigen::Vector3f(0, 1 ,0));
        seg.setEpsAngle(0.5);

        pcl::PointCloud<pcl::PointXYZRGB> cloud_filtered(temp);
        pcl::ExtractIndices<pcl::PointCloud<pcl::PointXYZRGB>::PointType> extract;
        bool groundPlaneFound = false;

//        while(cloud_filtered.size()>10 && !groundPlaneFound)
//        {
//            seg.setInputCloud(cloud_filtered.makeShared());
//            seg.segment(*inliers, *coefficients);
//            if(inliers->indices.size()==0)
//            {
//                break;
//            }
//            extract.setInputCloud(cloud_filtered.makeShared());
//            extract.setIndices(inliers);
//            // note: 这是怎么就将地面给检测出来了
//            if (std::abs(coefficients->values.at(3)) >0.07)
//            {
//                printf("Ground plane found: %zu/%zu inliers. Coeff: %f %f %f %f", inliers->indices.size(),
//                       cloud_filtered.size(),
//                       coefficients->values.at(0),
//                       coefficients->values.at(1),
//                       coefficients->values.at(2),
//                       coefficients->values.at(3));
//                extract.setNegative (false);
//                extract.filter (ground);
//                // remove ground points from full pointcloud:
//                // workaround for PCL bug:
//                if(inliers->indices.size() != cloud_filtered.size())
//                {
//                    extract.setNegative(true);
//                    pcl::PointCloud<pcl::PointXYZRGB> cloud_out;
//                    extract.filter(cloud_out);
//                    nonground += cloud_out;
//                    cloud_filtered = cloud_out;
//                }
//
//                groundPlaneFound = true;
//            }
//            else
//            {
//                printf("Horizontal plane (not ground) found: %zu/%zu inliers. Coeff: %f %f %f %f", inliers->indices.size(),
//                       cloud_filtered.size(),
//                       coefficients->values.at(0),
//                       coefficients->values.at(1),
//                       coefficients->values.at(2),
//                       coefficients->values.at(3));
//                pcl::PointCloud<pcl::PointXYZRGB> cloud_out;
//                extract.setNegative (false);
//                extract.filter(cloud_out);
//                nonground +=cloud_out;
//                if(inliers->indices.size() != cloud_filtered.size())
//                {
//                    extract.setNegative(true);
//                    cloud_out.points.clear();
//                    extract.filter(cloud_out);
//                    cloud_filtered = cloud_out;
//                }
//                else
//                {
//                    cloud_filtered.points.clear();
//
//                }
//            }
//
//        }//while

        if(!groundPlaneFound)
        {
        nonground = temp;
        }
        }
    }
    void MapDrawer::InsertScan(octomap::point3d sensorOrigin, pcl::PointCloud<pcl::PointXYZRGB> &ground, pcl::PointCloud<pcl::PointXYZRGB> &nonground)
    {

        if(!m_octree->coordToKeyChecked(sensorOrigin, m_updateBBXMin)|| !m_octree->coordToKeyChecked(sensorOrigin, m_updateBBXMax))
        {
            printf("coulde not generate key for origin\n");
        }
        octomap::KeySet free_cells, occupied_cells;
        for(auto p:ground.points)
        {
            octomap::point3d point(p.x, p.y, p.z);
            // only clear space (ground points)
            if(m_octree->computeRayKeys(sensorOrigin, point, m_keyRay))
            {
                free_cells.insert(m_keyRay.begin(), m_keyRay.end());
                m_octree->averageNodeColor(p.x, p.y, p.z, p.r,p.g, p.b);
            }
            octomap::OcTreeKey endKey;
            if(m_octree->coordToKeyChecked(point, endKey))
            {
                updateMinKey(endKey, m_updateBBXMin);
                updateMaxKey(endKey, m_updateBBXMax);
            }
            else
            {
                printf("could not generator key for endpoint");
            }
        }

        // all other points : free on ray, occupied on endpoings:
        for(auto p:nonground.points)
        {
            octomap::point3d point(p.x, p.y, p.z);
            //free cell
            if(m_octree->computeRayKeys(sensorOrigin, point, m_keyRay))
            {
                // free_cells.insert(m_keyRay.begin(),m_keyRay.end());
            }
            //occupided endpoint
            octomap::OcTreeKey key;
            if(m_octree->coordToKeyChecked(point, key))
            {
                occupied_cells.insert(key);
                updateMinKey(key, m_updateBBXMin);
                updateMaxKey(key, m_updateBBXMax);
                m_octree->averageNodeColor(p.x, p.y, p.z, p.r,p.g, p.b);
            }

        }pcl::PointCloud<pcl::PointXYZRGB>observation;
        for(octomap::KeySet::iterator it = free_cells.begin(), end= free_cells.end(); it!=end; ++it)
        {
            if(occupied_cells.find(*it) == occupied_cells.end())
            {
                m_octree->updateNode(*it, false);
            }
        }

        for(octomap::KeySet::iterator it = occupied_cells.begin(), end= occupied_cells.end(); it!=end; ++it)
        {
            m_octree->updateNode(*it, true);
        }

        m_octree->prune();
    }
    void MapDrawer::UpdateOctomap(vector<KeyFrame*> vKFs)
    {
        int N = vKFs.size();
        for ( size_t i=lastKeyframeSize; i<N ; i++ )
        {
            Eigen::Isometry3d pose = Planar_SLAM::Converter::toSE3Quat( vKFs[i]->GetPose());

            pcl::PointCloud<pcl::PointXYZRGB>  ground;
            pcl::PointCloud<pcl::PointXYZRGB>  nonground;
            // 区分地面与非地面
            GeneratePointCloud( vKFs[i], ground, nonground);
            octomap::point3d sensorOrigin = octomap::point3d( pose(0,3), pose(1,3), pose(2,3));
            InsertScan(sensorOrigin, ground, nonground);
        }

        lastKeyframeSize = N;
    }
    void MapDrawer::heightMapColor(double h, double& r, double &g, double& b)
    {

        double s = 1.0;
        double v = 1.0;

        h -= floor(h);
        h *= 6;

        int i;
        double m, n, f;

        i = floor(h);
        f = h - i;

        if(!(i & 1))
        {
            f = 1 - f;
        }
        m = v * (1-s);
        n = v * (1- s*f);

        switch(i)
        {
            case 6:
            case 0:
                r = v; g = n; b = m;
                break;
            case 1:
                r = n; g = v; b = m;
                break;
            case 2:
                r = m; g = v; b = n;
                break;
            case 3:
                r = m; g = n; b = v;
                break;
            case 4:
                r = n; g = m; b = v;
                break;
            case 5:
                r = v; g = m; b = n;
                break;
            default:
                r = 1; g = 0.5; b = 0.5;
                break;

        }

    }
    void MapDrawer::LoadOctoMap(const string octofile)
    {
        octomap::AbstractOcTree* tree = octomap::AbstractOcTree::read(octofile);
        m_octree= dynamic_cast<octomap::ColorOcTree*> (tree);
    }



} //namespace Planar_SLAM