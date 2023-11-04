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
def cmake_build_library(build_folder: str,
                        install_path: str,
                        cmake_options: str = ""):
    assert os.path.exists("CMakeLists.txt")
    project_path: str = os.path.abspath('.')
    build_path: str = os.path.join(project_path, build_folder)
    if not os.path.exists(build_path):
        os.makedirs(build_path)
    if not os.path.exists(install_path):
        os.makedirs(install_path)
    os.chdir(build_path)
    cmake_cmd: str = "cmake .. " + cmake_options + \
                     " -DCMAKE_POSITION_INDEPENDENT_CODE=ON" + \
                     " -DCMAKE_INSTALL_PREFIX=" + install_path + \
                     " -DCMAKE_PREFIX_PATH=" + install_path
    print(cmake_cmd)
    call_shell(cmake_cmd)
    call_shell("make -j16")
    call_shell("make install")
    os.chdir("..")
    assert os.path.exists("CMakeLists.txt")


def b2_build_library(build_folder: str, install_folder: str):
    call_shell("./bootstrap.sh --with-python-version=3.10 --prefix=" + install_folder)
    call_shell("./b2 install --build-dir=" + build_folder)


def compile_third_libs():
    os.chdir(THIRD_PATH)
    for _, dirs, _ in os.walk("."):
        for lib in dirs:
            if lib.lower() == "patch" or lib.lower() == "opencv_contrib":
                continue
            os.chdir(lib)
            if lib.lower() == "boost":
                b2_build_library(BUILD_FOLDER, INSTALL_PATH)
            elif lib.lower() == "opencv":
                cmake_build_library(BUILD_FOLDER, INSTALL_PATH,
                                    cmake_options.OPENCV_CMAKE_OPTIONS)
            else:
                cmake_build_library(BUILD_FOLDER, INSTALL_PATH)
            os.chdir("..")


if __name__ == '__main__':
    compile_third_libs()
