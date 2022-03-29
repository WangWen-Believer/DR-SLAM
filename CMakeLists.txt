cmake_minimum_required(VERSION 2.8)
project(DRSLAM)

IF (NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
ENDIF ()

MESSAGE("Build type: " ${CMAKE_BUILD_TYPE})

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
#ADD_DEFINITIONS("-w -O2 -pthread")


set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall  -O3 -march=native ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall   -O3 -march=native")
add_definitions(-w)
# Check C++11 or C++0x support
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if (COMPILER_SUPPORTS_CXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    add_definitions(-DCOMPILEDWITHC11)
    message(STATUS "Using flag -std=c++11.")
elseif (COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
    add_definitions(-DCOMPILEDWITHC0X)
    message(STATUS "Using flag -std=c++0x.")
else ()
    message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
endif ()

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

find_package(catkin REQUIRED COMPONENTS
        roscpp
        std_msgs
        sensor_msgs
        cv_bridge
        #camera_model
        message_filters
        )
include_directories(
        ${catkin_INCLUDE_DIRS}
)

find_package(OpenCV 3.0 QUIET)
if (NOT OpenCV_FOUND)
    find_package(OpenCV 2.4.3 QUIET)
    if (NOT OpenCV_FOUND)
        message(FATAL_ERROR "OpenCV > 2.4.3 not found.")
    endif ()
endif ()

message(OPENCV_VERSION)

find_package(Eigen3 3.1.0 REQUIRED)
find_package(Pangolin REQUIRED)
find_package(PCL 1.9 REQUIRED)
FIND_PACKAGE(octomap REQUIRED )
find_package(CUDA  REQUIRED)
include_directories(${PROJECT_SOURCE_DIR}/include)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda-10.2/include)
link_directories(/usr/local/cuda-10.2/lib64)
# cudnn
include_directories(/home/wangwen/Documents/cudnn-10.2-linux-x64-v8.2.1.32/cuda/include)
link_directories(/home/wangwen/Documents/cudnn-10.2-linux-x64-v8.2.1.32/cuda/lib64)
# tensorrt
include_directories(/home/wangwen/Documents/TensorRT-7.2.1.6.Ubuntu-18.04.x86_64-gnu.cuda-10.2.cudnn8.0/TensorRT-7.2.1.6/include)
link_directories(/home/wangwen/Documents/TensorRT-7.2.1.6.Ubuntu-18.04.x86_64-gnu.cuda-10.2.cudnn8.0/TensorRT-7.2.1.6/lib)


include_directories(
        ${PROJECT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/g2oAddition
        ${EIGEN3_INCLUDE_DIR}
        ${Pangolin_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        ${PROJECT_SOURCE_DIR}/src/CAPE
        ${OCTOMAP_INCLUDE_DIRS}
)

include_directories(include/peac)
add_definitions(${PCL_DEFINITIONS})
link_directories(${PCL_LIBRARY_DIRS})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_library(${PROJECT_NAME} SHARED
        src/System.cc
        src/Tracking.cc
        src/LocalMapping.cc
        src/LoopClosing.cc
        src/ORBextractor.cc
        src/ORBmatcher.cc
        src/FrameDrawer.cc
        src/Converter.cc
        src/MapPoint.cc
        src/MapLine.cpp
        src/KeyFrame.cc
        src/Map.cc
        src/MapDrawer.cc
        src/Optimizer.cc
        src/PnPsolver.cc
        src/Frame.cc
        src/KeyFrameDatabase.cc
        src/Sim3Solver.cc
        src/Initializer.cc
        src/Viewer.cc
        src/LineExtractor.cpp
        src/LSDextractor.cpp
        src/LSDmatcher.cpp
        src/PlaneExtractor.cpp
        src/Config.cc
        src/MapPlane.cc
        src/MeshViewer.cc
        src/PlaneMatcher.cpp
        src/PangolinViewer.cpp
        src/get_char_input.cpp
        src/CAPE/CAPE.cpp
        src/CAPE/PlaneSeg.cpp
        src/CAPE/CylinderSeg.cpp
        src/CAPE/Histogram.cpp
        src/YOLOX.cpp
        src/SystemSetting.cpp
        src/InitKeyFrame.cpp
        )
file(GLOB sources "*.cpp")
target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS}
        ${EIGEN3_LIBS}
        ${Pangolin_LIBRARIES}
        ${OCTOMAP_LIBRARIES}
        ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
        ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
        ${PCL_LIBRARIES}
        nvinfer
        cudart
        )

# Build examples

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/RGB-D)
add_executable(DR_SLAM Examples/RGB-D/main.cc)
target_link_libraries(DR_SLAM ${PROJECT_NAME} ${catkin_LIBRARIES} nvinfer)
add_executable(DR_SLAM_ros Examples/RGB-D/main_ros.cc)
target_link_libraries(DR_SLAM_ros ${PROJECT_NAME} ${catkin_LIBRARIES} nvinfer)
