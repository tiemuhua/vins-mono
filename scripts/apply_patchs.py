
import platform
import os
import subprocess

PLATFORM:str = platform.platform()
THIRD_PATH:str = os.path.split(os.path.realpath(__file__))[0] + '/../3rd/'
INSTALL_FOLDER:str = "install_" + PLATFORM

def call_shell(cmd:str)->int:
    return subprocess.run(cmd.split(" "))

def apply_patchs():
    os.chdir(THIRD_PATH)
    directories:list[str] = os.listdir(".")
    for dir in directories:
        if os.path.isfile(dir):
            continue
        os.chdir(dir)
        if not os.path.exists("../patch/" + dir):
            print("warning: there is no directory " + dir + "in patch folder")
        else:
            call_shell("git reset --hard")
            call_shell("git clean -xfd")
            call_shell("git apply ../patch/" + dir + "/0001-build.patch")
        os.chdir("..")

if __name__ == '__main__':
    apply_patchs()