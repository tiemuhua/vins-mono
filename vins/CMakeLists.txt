cmake_minimum_required(VERSION 2.8.3)
project(vins)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

set (CMAKE_CXX_STANDARD 17)

add_library(vins
        src/vins_core.cpp
        )

