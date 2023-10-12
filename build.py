
import platform
import os
import subprocess

PLATFORM:str = platform.platform()
THIRD_PATH:str = os.path.split(os.path.realpath(__file__))[0] + '/3rd/'
INSTALL_FOLDER:str = "install_" + PLATFORM

my_env:dict[str, str] = os.environ.copy()
os.chdir("3rd")

def call_shell_with_env(cmd:str, env:dict[str, str])->int:
    return subprocess.run(cmd.split(" "), env=env)


os.chdir("Boost")
call_shell_with_env("python3.10 build.py", my_env)
my_env["Boost_DIR"] = THIRD_PATH + "Boost/" + INSTALL_FOLDER + "/lib/cmake/" + "/Boost-1.82.0/"

os.chdir("../Eigen")
call_shell_with_env("python3.10 build.py", my_env)
my_env["Eigen_DIR"] = THIRD_PATH + "Eigen/" + INSTALL_FOLDER + "/share/eigen3/cmake/"

os.chdir("../opencv")
call_shell_with_env("python3.10 build.py", my_env)
my_env["OpenCV_DIR"] = THIRD_PATH + "opencv/" + INSTALL_FOLDER + "/lib/cmake/" + "/opencv4"

os.chdir("../gflags")
call_shell_with_env("python3.10 build.py", my_env)
my_env["gflags_DIR"] = THIRD_PATH + "gflags/" + INSTALL_FOLDER + "/lib/cmake/" + "/gflags"

os.chdir("../glog")
call_shell_with_env("python3.10 build.py", my_env)
my_env["glog_DIR"] = THIRD_PATH + "glog/" + INSTALL_FOLDER + "/lib/cmake/" + "/glog"

os.chdir("../Ceres")
call_shell_with_env("python3.10 build.py", my_env)
my_env["Ceres_DIR"] = THIRD_PATH + "Ceres/" + INSTALL_FOLDER + "/lib/cmake/" + "/Ceres"

os.chdir("../DLib")
call_shell_with_env("python3.10 build.py", my_env)
my_env["DLib_DIR"] = THIRD_PATH + "DLib/" + INSTALL_FOLDER + "/lib/cmake/" + "/DLib"

os.chdir("../DBoW2")
call_shell_with_env("python3.10 build.py", my_env)
my_env["DBoW2_DIR"] = THIRD_PATH + "DBoW2/" + INSTALL_FOLDER + "/lib/cmake/" + "/DBoW2"

os.chdir("../camodocal")
call_shell_with_env("python3.10 build.py", my_env)
my_env["camodocal_DIR"] = THIRD_PATH + "camodocal/" + INSTALL_FOLDER + "/lib/cmake/" + "/camodocal"
os.chdir("..")


