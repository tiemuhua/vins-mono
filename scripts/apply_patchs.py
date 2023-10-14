import platform
import os
import subprocess
from subprocess import CompletedProcess
from typing import Any

PLATFORM: str = platform.platform()
THIRD_PATH: str = os.path.split(os.path.realpath(__file__))[0] + '/../3rd/'


def call_shell(cmd: str) -> CompletedProcess[Any] | CompletedProcess[bytes]:
    return subprocess.run(cmd.split(" "))


def apply_patches():
    os.chdir(THIRD_PATH)
    directories: list[str] = os.listdir(".")
    for directory in directories:
        if os.path.isfile(directory):
            continue
        if not os.path.exists("patch/" + directory):
            print("warning: there is no directory " + directory + "in patch folder")
            continue
        os.chdir(directory)
        call_shell("git reset --hard")
        call_shell("git clean -xfd")
        call_shell("git apply ../patch/" + directory + "/0001-build.patch")
        os.chdir("..")


if __name__ == '__main__':
    apply_patches()
