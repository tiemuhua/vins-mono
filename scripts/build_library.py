import os
import subprocess
from subprocess import CompletedProcess
from typing import Any


def call_shell_with_env(cmd: str, env: dict[str, str]) -> CompletedProcess[Any] | CompletedProcess[bytes]:
    return subprocess.run(cmd.split(" "), env=env)


def cmake_build_library(env: dict[str, str], build_folder: str, install_folder: str, cmake_options: str = ""):
    project_path: str = os.path.abspath('.')
    build_path: str = os.path.join(project_path, build_folder)
    install_path: str = os.path.join(project_path, install_folder)
    if not os.path.exists(build_path):
        os.makedirs(build_path)
    if not os.path.exists(install_path):
        os.makedirs(install_path)
    os.chdir(build_path)
    cmake_cmd: str = "cmake .. " + cmake_options + "-DCMAKE_INSTALL_PREFIX=" + install_path
    print(cmake_cmd)
    call_shell_with_env(cmake_cmd, env)
    call_shell_with_env("make -j16", env)
    call_shell_with_env("make install", env)
    os.chdir("..")


def b2_build_library(env: dict[str, str], build_folder: str, install_folder: str):
    call_shell_with_env("./bootstrap.sh --with-python-version=3.10 --prefix=" + install_folder, env)
    call_shell_with_env("./b2 install --build-dir=" + build_folder, env)
