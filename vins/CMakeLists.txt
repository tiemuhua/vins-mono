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

# 顺序不能乱，eigen、glog必须在Ceres前面，gflags必须在glog前面，诸如此类
find_package(Boost REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(gflags REQUIRED)
find_package(glog REQUIRED)
find_package(Ceres REQUIRED)
find_package(DLib REQUIRED)
find_package(DBoW2 REQUIRED)
find_package(CAMODOCAL REQUIRED)

include_directories(
        ${Boost_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${gflags_INCLUDE_DIRS}
        ${DLib_INCLUDE_DIRS}
        ${DBoW2_INCLUDE_DIRS}
        src
)

file(GLOB_RECURSE SRC RELATIVE ${PROJECT_SOURCE_DIR} src/*.cpp)
add_library(vins STATIC ${SRC})

target_link_libraries(vins
        Eigen3::Eigen
        Ceres::ceres
        glog::glog
        camodocal_camera_models
        ${DBoW2_LIBS}
        ${DLib_LIBS}
        ${OpenCV_LIBS}
        )
