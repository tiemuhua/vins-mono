import platform
import os
import subprocess
from typing import List, Any

import cmake_options

PLATFORM: str = platform.platform()
PROJECT_ROOT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
THIRD_PATH: str = os.path.join(PROJECT_ROOT_PATH, "3rd")
INSTALL_PATH: str = os.path.abspath(os.path.join(PROJECT_ROOT_PATH, "install_" + PLATFORM))
BUILD_FOLDER: str = "build_" + PLATFORM


def call_shell(cmd: str) -> subprocess.CompletedProcess[Any] | subprocess.CompletedProcess[bytes]:
    word_list_of_cmd: List[str] = [word for word in cmd.split(" ") if word != ""]
    return subprocess.run(word_list_of_cmd)


# 执行本函数时应当保证工作目录位于 CMakeLists.txt 目录中
# 本函数结束时保证仍然位于 CMakeLists.txt 目录中
def cmake_build_library(options: str = ""):
    assert os.path.exists("CMakeLists.txt")
    project_path: str = os.path.abspath('.')
    build_path: str = os.path.join(project_path, BUILD_FOLDER)
    if not os.path.exists(build_path):
        os.makedirs(build_path)
    if not os.path.exists(INSTALL_PATH):
        os.makedirs(INSTALL_PATH)
    os.chdir(build_path)
    cmake_cmd: str = "cmake .. " + options + \
                     " -DCMAKE_POSITION_INDEPENDENT_CODE=ON" + \
                     " -DCMAKE_INSTALL_PREFIX=" + INSTALL_PATH + \
                     " -DCMAKE_PREFIX_PATH=" + INSTALL_PATH
    print(cmake_cmd)
    call_shell(cmake_cmd)
    call_shell("make -j16")
    call_shell("make install")
    os.chdir("..")
    assert os.path.exists("CMakeLists.txt")


def b2_build_library():
    call_shell("./bootstrap.sh --with-python-version=3.10 --prefix=" + INSTALL_PATH)
    call_shell("./b2 install --build-dir=" + BUILD_FOLDER)


def compile_third_libs():
    os.chdir(THIRD_PATH)
    # 这个顺序已经考虑了三方库之间的依赖关系。如果三方库有几十个的话，有必要通过拓扑排序来自动根据依赖关系求解编译顺序。
    # macOS对文件夹大小写不敏感，但是Linux对大小写敏感
    for lib in ["boost", "Eigen3", "gflags", "glog", "opencv", "Ceres", "DLib", "DBoW2", "camodocal"]:
        os.chdir(lib)
        if lib == "boost":
            b2_build_library()
        elif lib == "opencv":
            cmake_build_library(cmake_options.OPENCV_CMAKE_OPTIONS)
        else:
            cmake_build_library()
        os.chdir("..")


if __name__ == '__main__':
    # compile_third_libs()
    os.chdir(os.path.join(PROJECT_ROOT_PATH, "vins"))
    cmake_build_library()
    os.chdir("..")
    os.chdir(os.path.join(PROJECT_ROOT_PATH, "kitti_demo"))
    os.chdir("..")
