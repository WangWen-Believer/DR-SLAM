# DR-SLAM

DR-SLAM is a drift rejection SLAM method for indoor scenarios. It uses multiple feature primitives and geometric information (parallel or perpendicular) restraint in man-made environments. Under some satisfy Manhattan world assumption scene, such as corridors, we can get absolutely and drift-free rotation estimation using a Gaussian sphere. By fully utilizing drift-free rotation estimation under Manhattan world assumption and the local stability of purely track restricted by point, line, and plane features, our drift rejection SLAM method becomes more accurate and robust.

![图片1](README.assets/%E5%9B%BE%E7%89%871.svg)

updated on **1.5.2022**: Add octomap for navigation and Vanishing point with geometry restrain in 2d image

updated on **2.20.2022**: Add Object detection

updated on **3.17.2022** Add Map Relocalization

![](README.assets/2022-03-29%2021-55-47%20%E7%9A%84%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png)

## 1. Prerequisites

1.1 **Ubuntu** and **ROS** Ubuntu 18.04.  ROS Melodic.

1.2 **Pangolin**

we use Pangolin for visualization and user interface. Download and install instructions can be found at:  https://github.com/stevenlovegrove/Pangolin.

1.3 **OpenCV**

We use [OpenCV](http://opencv.org/) to manipulate images and features. Dowload and install instructions can be found at: [http://opencv.org](http://opencv.org/). **Required at leat 2.4.3. Tested with OpenCV 3.4.4**.

1.4 **Eigen3**

Required by g2o (see below). Download and install instructions can be found at: [http://eigen.tuxfamily.org](http://eigen.tuxfamily.org/). **Required at least 3.1.0. Tested with Eigen 3.3.7**. 

1.5 **DBoW2** and g2o (Included in Thirdparty folder)

We use modified versions of the DBoW2 library to perform place recognition and g2o library to perform non-linear optimizations. Both modified libraries (which are BSD) are included in the *Thirdparty* folder.

1.6 **Ocotmap**

We use octomap to generation navigation map. Download and install instructions can be found at :https://github.com/OctoMap/octomap.

1.7 **YOLOX**

We merge YOLOX into our system for object detection. Download and install instructions can be found at:  https://github.com/Megvii-BaseDetection/YOLOX. (cuda 10.2, cudnn 8.2.1.32, TensorRT 7.2.1.6)

## 2. Building DR-SLAM on ROS

Clone the repository and catkin_make:

    cd ~/catkin_dr/src
    git clone https://github.com/WangWen-Believer/DR-SLAM.git
    cd DR-SLAM && ./build.sh
    cd ../../
    catkin_make -j4
    source ~/catkin_ws/devel/setup.bash
## 3. Testing DR-SLAM on Datasets

### 3.1 ICL-NUIM Dataset

1. Download a sequence from http://vision.in.tum.de/data/datasets/rgbd-dataset/download and uncompress it.

2.  Execute the following command. Change `PATH_TO_SEQUENCE_FOLDER`to the uncompressed sequence folder.

     ```
     ./Examples/RGB-D/DR_SLAM Vocabulary/ORBvoc.txt Examples/RGB-D/ICL.yaml PATH_TO_SEQUENCE_FOLDER  PATH_TO_SEQUENCE_FOLDER/ASSOCIATIONS_FILE
     ```

### 3.2 TUM Dataset

1. Download a sequence from http://vision.in.tum.de/data/datasets/rgbd-dataset/download and uncompress it.

2. Associate RGB images and depth images using the python script [associate.py](http://vision.in.tum.de/data/datasets/rgbd-dataset/tools).

     ```
     python associate.py PATH_TO_SEQUENCE/rgb.txt PATH_TO_SEQUENCE/depth.txt > associations.txt
     ```

3. Execute the following command. Change `TUMX.yaml` to TUM1.yaml,TUM2.yaml or TUM3.yaml for freiburg1, freiburg2 and freiburg3 sequences respectively. Change `PATH_TO_SEQUENCE_FOLDER`to the uncompressed sequence folder.

     ```
     ./Examples/RGB-D/DR_SLAM Vocabulary/ORBvoc.txt Examples/RGB-D/TUMX.yaml PATH_TO_SEQUENCE_FOLDER ASSOCIATIONS_FILE
     ```

### 3.3 Author-collected RGB-D Dataset

Record our datasets  in office and corridor environments by using RealSense D435i.

- office
- [corridor](https://drive.google.com/file/d/1HPyzFBHa8Wc2QSSPkfxUWAEnVNLhhUQV/view?usp=sharing)

Topics:

- /camera/aligned_depth_to_color/image_raw
- /camera/color/image_raw
- /camera/imu 

```
roslaunch DRSLAM realsense_Planar.launch
```

![](README.assets/2022-03-29%2015-25-14%20%E7%9A%84%E5%B1%8F%E5%B9%95%E6%88%AA%E5%9B%BE.png)

## 4. Licence

The source code is released under [GPLv3](https://github.com/WangWen-Believer/DR-SLAM/blob/main/LICENSE) license.

We are still working on improving the code reliability. For any technical issues, please contact us.

