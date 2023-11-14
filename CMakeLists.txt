
# 本文件仅仅是用来给clion生成代码高亮与索引，实际编译过程中并不会用到
cmake_minimum_required(VERSION 3.21)
project(vins_mono)

set(CMAKE_CXX_STANDARD 14)

# install_${platform}在首次编译完成后才会生成，请用户自行修改相应的install目录
include_directories(install_macOS-13.4.1-arm64-arm-64bit/include)
include_directories(install_macOS-13.4.1-arm64-arm-64bit/include/eigen3)
include_directories(install_macOS-13.4.1-arm64-arm-64bit/include/opencv4)

include_directories(vins)
include_directories(vins/impl)

add_executable(vins_mono
        kitti_demo/main.cpp
        vins/impl/front_end_optimize/cost_functions/imu_cost.cpp
        vins/impl/front_end_optimize/cost_functions/imu_cost.h
        vins/impl/front_end_optimize/cost_functions/marginal_cost.cpp
        vins/impl/front_end_optimize/cost_functions/marginal_cost.h
        vins/impl/front_end_optimize/cost_functions/project_cost.cpp
        vins/impl/front_end_optimize/cost_functions/project_cost.h
        vins/impl/front_end_optimize/cost_functions/project_td_cost.cpp
        vins/impl/front_end_optimize/cost_functions/project_td_cost.h
        vins/impl/front_end_optimize/front_end_optimize.cpp
        vins/impl/front_end_optimize/front_end_optimize.h
        vins/impl/initial/impl/visual_inertial_aligner.cpp
        vins/impl/initial/impl/visual_inertial_aligner.h
        vins/impl/initial/impl/visual_initiator.cpp
        vins/impl/initial/impl/visual_initiator.h
        vins/impl/initial/initiate.cpp
        vins/impl/initial/initiate.h
        vins/impl/loop_closer/impl/loop_detector.cpp
        vins/impl/loop_closer/impl/loop_detector.h
        vins/impl/loop_closer/impl/loop_relative_pos.cpp
        vins/impl/loop_closer/impl/loop_relative_pos.h
        vins/impl/loop_closer/keyframe.cpp
        vins/impl/loop_closer/keyframe.h
        vins/impl/loop_closer/loop_closer.cpp
        vins/impl/loop_closer/loop_closer.h
        vins/impl/camera_wrapper.h
        vins/impl/feature_helper.cpp
        vins/impl/feature_helper.h
        vins/impl/feature_tracker.cpp
        vins/impl/feature_tracker.h
        vins/impl/imu_integrator.cpp
        vins/impl/imu_integrator.h
        vins/impl/vins_controller.cpp
        vins/impl/vins_controller.h
        vins/impl/vins_model.h
        vins/impl/vins_utils.h
        vins/param.h
        vins/vins_logic.cpp
        vins/vins_logic.h
        )
