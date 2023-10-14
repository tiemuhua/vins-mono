import platform
import os

import build_library
import cmake_options

PLATFORM: str = platform.platform()
THIRD_PATH: str = os.path.split(os.path.realpath(__file__))[0] + '/../3rd/'
INSTALL_FOLDER: str = "install_" + PLATFORM
BUILD_FOLDER: str = "build_" + PLATFORM


def compile_third_libs():
    my_env: dict[str, str] = os.environ.copy()
    os.chdir(THIRD_PATH)
    os.chdir("Boost")
    build_library.b2_build_library(my_env, BUILD_FOLDER, INSTALL_FOLDER)
    my_env["Boost_DIR"] = THIRD_PATH + "Boost/" + INSTALL_FOLDER + "/lib/cmake/" + "/Boost-1.82.0/"

    os.chdir("../Eigen3")
    build_library.cmake_build_library(my_env, BUILD_FOLDER, INSTALL_FOLDER)
    my_env["Eigen3_DIR"] = THIRD_PATH + "Eigen3/" + INSTALL_FOLDER + "/share/eigen3/cmake/"

    os.chdir("../opencv")
    build_library.cmake_build_library(my_env, BUILD_FOLDER, INSTALL_FOLDER, cmake_options.OPENCV_CMAKE_OPTIONS)
    my_env["OpenCV_DIR"] = THIRD_PATH + "opencv/" + INSTALL_FOLDER + "/lib/cmake/" + "/opencv4"

    os.chdir("../gflags")
    build_library.cmake_build_library(my_env, BUILD_FOLDER, INSTALL_FOLDER)
    my_env["gflags_DIR"] = THIRD_PATH + "gflags/" + INSTALL_FOLDER + "/lib/cmake/" + "/gflags"

    os.chdir("../glog")
    build_library.cmake_build_library(my_env, BUILD_FOLDER, INSTALL_FOLDER)
    my_env["glog_DIR"] = THIRD_PATH + "glog/" + INSTALL_FOLDER + "/lib/cmake/" + "/glog"

    os.chdir("../Ceres")
    build_library.cmake_build_library(my_env, BUILD_FOLDER, INSTALL_FOLDER)
    my_env["Ceres_DIR"] = THIRD_PATH + "Ceres/" + INSTALL_FOLDER + "/lib/cmake/" + "/Ceres"

    os.chdir("../DLib")
    build_library.cmake_build_library(my_env, BUILD_FOLDER, INSTALL_FOLDER)
    my_env["DLib_DIR"] = THIRD_PATH + "DLib/" + INSTALL_FOLDER + "/lib/cmake/" + "/DLib"

    os.chdir("../DBoW2")
    build_library.cmake_build_library(my_env, BUILD_FOLDER, INSTALL_FOLDER)
    my_env["DBoW2_DIR"] = THIRD_PATH + "DBoW2/" + INSTALL_FOLDER + "/lib/cmake/" + "/DBoW2"

    os.chdir("../camodocal")
    build_library.cmake_build_library(my_env, BUILD_FOLDER, INSTALL_FOLDER)
    my_env["camodocal_DIR"] = THIRD_PATH + "camodocal/" + INSTALL_FOLDER + "/lib/cmake/" + "/camodocal"
    os.chdir("..")

    print("-------------------------------------------------------------")
    print("please export xxx_DIR in your ~/.zshrc or ~/.bashrc manually")
    libs: list[str] = ["Boost", "Eigen3", "OpenCV", "gflags", "glog", "Ceres", "DLib", "DBoW2", "camodocal"]
    for lib in libs:
        export_expression: str = "export {lib}_DIR={path}".format(lib=lib, path=my_env[lib + "_DIR"])
        print(export_expression)
    print("-------------------------------------------------------------")


if __name__ == '__main__':
    compile_third_libs()
