#include "PlaneExtractor.h"
#include "Config.h"
using namespace std;
using namespace cv;
using namespace Eigen;
namespace Planar_SLAM {
    PlaneDetection::PlaneDetection() {
        cloud.vertices.resize(kDepthHeight * kDepthWidth);
        cloud.w = kDepthWidth;
        cloud.h = kDepthHeight;
    }

    PlaneDetection::~PlaneDetection() {
        cloud.vertices.clear();
        seg_img_.release();
        color_img_.release();
    }

    bool PlaneDetection::readColorImage(cv::Mat RGBImg) {
        color_img_ = RGBImg;
        if (color_img_.empty() || color_img_.depth() != CV_8U) {
            cout << "ERROR: cannot read color image. No such a file, or the image format is not 8UC3" << endl;
            return false;
        }
        return true;
    }

    bool PlaneDetection::readDepthImage(cv::Mat depthImg, cv::Mat &K,const float depthfactor) {
        cv::Mat depth_img = depthImg;
        if (depth_img.empty() || depth_img.depth() != CV_16U) {
            cout << "WARNING: cannot read depth image. No such a file, or the image format is not 16UC1" << endl;
            return false;
        }

        int rows = depth_img.rows, cols = depth_img.cols;
        int vertex_idx = 0;
        for (int i = 0; i < rows; i += 1) {
            for (int j = 0; j < cols; j += 1) {
                double z = (double) (depth_img.at<unsigned short>(i, j)) * depthfactor;
                if (_isnan(z)) {
                    cloud.vertices[vertex_idx++] = VertexType(0, 0, z);
                    continue;
                }
                else if(z > 5.0)
                {
                    cloud.vertices[vertex_idx++] = VertexType(0, 0, 0);
                    continue;
                }
                double x = ((double) j - K.at<float>(0, 2)) * z / K.at<float>(0, 0);
                double y = ((double) i - K.at<float>(1, 2)) * z / K.at<float>(1, 1);
                cloud.vertices[vertex_idx++] = VertexType(x, y, z);
            }
        }
        return true;
    }

    void PlaneDetection::runPlaneDetection() {
        seg_img_ = cv::Mat(kDepthHeight, kDepthWidth, CV_8UC3);
        seg_output = cv::Mat_<uchar>(kDepthHeight, kDepthWidth,CV_8UC1);
        plane_filter.run(&cloud, &plane_vertices_, &seg_img_, &seg_output);

        plane_num_ = (int) plane_vertices_.size();
    }

/*-----------------------------------------------*/
    PlaneDetection_CAPE::~PlaneDetection_CAPE() {
        color_img_.release();
        depth_img.release();
    }

    bool PlaneDetection_CAPE::readColorImage(cv::Mat RGBImg) {
        color_img_ = RGBImg;
        if (color_img_.empty() || color_img_.depth() != CV_8U) {
            cout << "ERROR: cannot read color image. No such a file, or the image format is not 8UC3" << endl;
            return false;
        }
        return true;
    }

    void organizePointCloudByCell(Eigen::MatrixXf &cloud_in, Eigen::MatrixXf &cloud_out, cv::Mat &cell_map) {

        int width = cell_map.cols;
        int height = cell_map.rows;
        int mxn = width * height;
        int mxn2 = 2 * mxn;

        int id, it(0);
        int *cell_map_ptr;
        for (int r = 0; r < height; r++) {
            cell_map_ptr = cell_map.ptr<int>(r);
            for (int c = 0; c < width; c++) {
                id = cell_map_ptr[c];
                *(cloud_out.data() + id) = *(cloud_in.data() + it);
                *(cloud_out.data() + mxn + id) = *(cloud_in.data() + mxn + it);
                *(cloud_out.data() + mxn2 + id) = *(cloud_in.data() + mxn2 + it);
                it++;
            }
        }
    }

    bool PlaneDetection_CAPE::readDepthImage(cv::Mat depthImg, cv::Mat &K) {
        depth_img = depthImg;
        K_ = K;
        if (depth_img.empty() || depth_img.depth() != CV_32F) {
            cout << "WARNING: cannot read depth image. No such a file, or the image format is not 16UC1" << endl;
            return false;
        }
        return true;
    }

    void PlaneDetection_CAPE::runPlaneDetection() {
        int rows = depth_img.rows, cols = depth_img.cols;
        Eigen::MatrixXf cloud_array(cols * rows, 3);
        Eigen::MatrixXf cloud_array_organized(cols * rows, 3);

        // 生成点云
        for (int i = 0; i < rows; i += 1) {
            for (int j = 0; j < cols; j += 1) {
                double z = (double) (depth_img.at<float>(i, j));
                double x = ((double) j - K_.at<float>(0, 2)) * z / K_.at<float>(0, 0);
                double y = ((double) i - K_.at<float>(1, 2)) * z / K_.at<float>(1, 1);
                int index = i * cols + j;
                cloud_array(index, 0) = x;
                cloud_array(index, 1) = y;
                cloud_array(index, 2) = z;
            }
        }
        cv::Mat_<cv::Vec3b> seg_rz = cv::Mat_<cv::Vec3b>(rows, cols, cv::Vec3b(0, 0, 0));
        seg_output = cv::Mat_<uchar>(rows, cols, uchar(0));// 对应点云的平面属性

        // Run CAPE
        int height = rows;
        int width = cols;
        // Pre-computations for maping an image point cloud to a cache-friendly array where cell's local point clouds are contiguous
        cv::Mat_<int> cell_map(height, width);
        int nr_horizontal_cells = width / PATCH_SIZE;
        // 网格划分 recommand: 20
        for (int r = 0; r < height; r++) {
            int cell_r = r / PATCH_SIZE;
            int local_r = r % PATCH_SIZE;
            for (int c = 0; c < width; c++) {
                int cell_c = c / PATCH_SIZE;
                int local_c = c % PATCH_SIZE;
                cell_map.at<int>(r, c) =
                        (cell_r * nr_horizontal_cells + cell_c) * PATCH_SIZE * PATCH_SIZE + local_r * PATCH_SIZE +
                        local_c;
            }
        }
        plane_detector = new CAPE(height, width, PATCH_SIZE, PATCH_SIZE, cylinder_detection, COS_ANGLE_MAX,
                                  MAX_MERGE_DIST);
//    double t1 = cv::getTickCount();
        organizePointCloudByCell(cloud_array, cloud_array_organized, cell_map);
        plane_detector->process(cloud_array_organized, nr_planes, nr_cylinders, seg_output, plane_params,
                                cylinder_params);
//    double t2 = cv::getTickCount();
//    double time_elapsed = (t2-t1)/(double)cv::getTickFrequency();
//    cout<<"Total time elapsed: "<<time_elapsed<<endl;


//    for(int p_id=0; p_id<nr_planes;p_id++){
//        cout<<"[Plane #"<<p_id<<"] with ";
//        cout<<"normal: ("<<plane_params[p_id].normal[0]<<" "<<plane_params[p_id].normal[1]<<" "<<plane_params[p_id].normal[2]<<"), ";
//        cout<<"d: "<<plane_params[p_id].d<<endl;
//    }
        uchar *sCode;
        int code;
        //便利获取各自面的属性
        // 每个平面的索引
        // note: 实例化
        for (std::size_t i = 0; i < nr_planes; ++i) {
            PointCloud::Ptr cloud_in(new PointCloud);
            plane_cloud.push_back(std::move(cloud_in));
        }

        for (int i = 0; i < height; i++) {
            sCode = seg_output.ptr<uchar>(i);
            for (int j = 0; j < width; j++) {
                code = *sCode;
                if (code > 0) {
                    PointT p;
                    int index = i * width + j;
                    p.x = cloud_array(index, 0);
                    p.y = cloud_array(index, 1);
                    p.z = cloud_array(index, 2);
                    //cloud_array
                    plane_cloud[code - 1]->points.push_back(p);
                }
                sCode++;
            }
        }
    }
}