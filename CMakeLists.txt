cmake_minimum_required (VERSION 3.6)

set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}" CACHE PATH "Installation directory" FORCE)
message(STATUS "CMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}")

enable_language(CXX)

project (vins)

#set(CMAKE_VERBOSE_MAKEFILE on)
set(SELF_LIBS_OUT ${CMAKE_SYSTEM_NAME}.out)

include_directories(~/vins/Sophus/)
include_directories(~/vins/eigen-3.4.0)
include_directories(~/vins)

add_subdirectory(camera_model camera_model)
add_subdirectory(feature_tracker feature_tracker)
add_subdirectory(pose_graph pose_graph)
add_subdirectory(vins_estimator vins_estimator)
