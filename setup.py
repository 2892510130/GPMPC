import platform
import os
import subprocess
import argparse

parser = argparse.ArgumentParser(description="CMake Configuration Script")
parser.add_argument(
    "--config_type",
    default="Release",
    choices=["Release", "Debug", "RelWithDebInfo", "MinSizeRel"],
    help="Specify the build type (default: Release)"
)

args = parser.parse_args()
config_type = args.config_type

system_name = platform.machine()

folder_path = f"build/{system_name}/{config_type}"

os.makedirs(folder_path, exist_ok=True)

os.chdir(folder_path)

subprocess.run(["cmake", "-G MinGW Makefiles", f"-DCMAKE_BUILD_TYPE={config_type}", "../../.."], check=True)

subprocess.run(["cmake", f"--build", "."], check=True)