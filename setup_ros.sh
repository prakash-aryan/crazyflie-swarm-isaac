#!/usr/bin/env bash
# ROS 2 side: uv environment with the Python dependencies of the Crazyswarm2 simulator, and the Python bindings of
# the Crazyflie firmware (cffirmware), built from crazyflie-firmware 2025.12.1.
# Prerequisites: ROS 2 Humble with
#   sudo apt install ros-humble-crazyflie ros-humble-crazyflie-sim ros-humble-crazyflie-py swig build-essential
set -euo pipefail
ROOT=$(cd "$(dirname "$0")" && pwd)
uv venv --allow-existing --python /usr/bin/python3 --system-site-packages "$ROOT/.venv-ros"
uv pip install --python "$ROOT/.venv-ros/bin/python" "numpy==1.26.4" rowan transforms3d kconfiglib setuptools
if [ ! -d "$ROOT/ext/crazyflie-firmware" ]; then
    git clone --depth 1 --branch 2025.12.1 --recursive --shallow-submodules \
        https://github.com/bitcraze/crazyflie-firmware.git "$ROOT/ext/crazyflie-firmware"
fi
cd "$ROOT/ext/crazyflie-firmware"
export PATH="$ROOT/.venv-ros/bin:$PATH"
LOG="$ROOT/ext/firmware_build.log"
if ! { make cf2_defconfig && make bindings_python; } > "$LOG" 2>&1; then
    tail -20 "$LOG"
    exit 1
fi
echo "cffirmware built in $ROOT/ext/crazyflie-firmware/build (log: $LOG)"
