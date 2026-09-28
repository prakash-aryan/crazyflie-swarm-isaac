# Source before running the ROS 2 side: ROS 2 Humble, the cffirmware bindings and the uv environment (setup_ros.sh).
source /opt/ros/humble/setup.bash
CF_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH=$CF_ROOT/ext/crazyflie-firmware/build:$CF_ROOT/ros:$PYTHONPATH
export PATH=$CF_ROOT/.venv-ros/bin:$PATH
export ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-42}
export ROS_LOCALHOST_ONLY=1
