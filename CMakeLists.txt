cmake_minimum_required (VERSION 3.6)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}" CACHE PATH "Installation directory" FORCE)
message(STATUS "CMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}")
set(CMAKE_CXX_STANDARD 17)
enable_language(CXX)

if(WIN32)
    add_definitions(/MP) # 相当于make -j
    add_compile_options(/utf-8)
endif()

project (vins)

set(SELF_LIBS_OUT ${CMAKE_SYSTEM_NAME}.out)

include_directories(src/vins)

# 顺序不能乱，eigen、glog必须在Ceres前面，gflags必须在glog前面，诸如此类
find_package(
        Boost REQUIRED
        PATHS 3rd/Boost/install_mac_m1_dir/lib/cmake/Boost-1.76.0
        NO_DEFAULT_PATH
)
find_package(
        Eigen3 REQUIRED
        PATHS 3rd/Eigen3/install_mac_m1_dir/share/eigen3/cmake
        NO_DEFAULT_PATH
)
find_package(
        gflags REQUIRED
        PATHS 3rd/gflags/install_mac_m1_dir/lib/cmake/gflags
        NO_DEFAULT_PATH
)
find_package(
        glog REQUIRED
        PATHS 3rd/glog/install_mac_m1_dir/lib/cmake/glog
        NO_DEFAULT_PATH
)
find_package(
        Ceres REQUIRED
        PATHS 3rd/Ceres/install_mac_m1_dir/lib/cmake/Ceres
        NO_DEFAULT_PATH
)
find_package(
        DBoW2 REQUIRED
        PATHS 3rd/DBoW2/install_mac_m1_dir/lib/cmake/DBoW2
        NO_DEFAULT_PATH
)
find_package(
        OpenCV REQUIRED
        PATHS 3rd/opencv/install_mac_m1_dir/lib/cmake/opencv4
        NO_DEFAULT_PATH
)
include_directories(
        ${Boost_INCLUDE_DIRS}
        ${Eigen3_INCLUDE_DIRS}
        ${GLOG_INCLUDE_DIRS}
        ${GFLAGS_INCLUDE_DIR}
        ${Ceres_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIR}
)

file(GLOB SELF_TEMP_SRC_FILES RELATIVE ${PROJECT_SOURCE_DIR} src/*.cc src/*.h)

file(GLOB_RECURSE SRC RELATIVE ${PROJECT_SOURCE_DIR} src/*.cpp)
add_library(vins STATIC ${SRC})

add_executable(kitti_demo kitti_demo/main.cpp)
target_link_libraries(kitti_demo vins)