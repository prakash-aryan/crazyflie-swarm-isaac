"""Start the Crazyswarm2 simulation server (crazyflie_sim) for a given drone list and physics backend.

The server runs the Crazyflie firmware's high-level commander, trajectory planner and controller in the loop
(software in the loop, via the cffirmware bindings) and exposes the same ROS 2 services and topics as the
hardware server. Parameters are merged from the drone list and the stock server.yaml, as in crazyflie/launch.py.

    python run_server.py --crazyflies ../config/office6.yaml --backend isaac --controller mellinger
"""
import argparse
import os
import sys
import tempfile

from ament_index_python.packages import get_package_share_directory
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--crazyflies", required=True, help="crazyflies.yaml with the drone list")
ap.add_argument("--backend", default="np", help="np (numpy dynamics) or isaac (Isaac Sim physics)")
ap.add_argument("--controller", default="mellinger", choices=["pid", "mellinger", "brescianini"])
ap.add_argument("--record", default=None, help="record_states output directory (optional)")
args, ros_args = ap.parse_known_args()

share = get_package_share_directory("crazyflie")
with open(os.path.join(share, "config", "server.yaml")) as f:
    server = yaml.safe_load(f)["/crazyflie_server"]["ros__parameters"]
sim = server["sim"]
sim["backend"] = args.backend
sim["controller"] = args.controller
for vis in sim["visualizations"].values():
    vis["enabled"] = False
if args.record:
    sim["visualizations"]["record_states"].update(enabled=True, output_dir=args.record, file_formats=["csv"])
urdf = os.path.join(get_package_share_directory("crazyflie_description"), "urdf", "crazyflie_description.urdf")
with open(urdf) as f:
    server["robot_description"] = f.read()
with open(args.crazyflies) as f:
    params = {**yaml.safe_load(f), **server}

params_file = tempfile.NamedTemporaryFile("w", suffix=".yaml", prefix="cf_server_", delete=False)
yaml.safe_dump({"crazyflie_server": {"ros__parameters": params}}, params_file)
params_file.close()

# crazyflie_sim 1.0.7 calls plan_start_trajectory(planner, traj, reversed, relative, start_from); firmware 2025.12
# added relative_yaw and start_yaw arguments. Keep the 1.0.7 behaviour (absolute yaw).
import cffirmware  # noqa: E402

_plan_start_trajectory = cffirmware.plan_start_trajectory


def _plan_start_trajectory_compat(planner, traj, reversed_, relative, *rest):
    if len(rest) == 1:
        return _plan_start_trajectory(planner, traj, reversed_, relative, False, rest[0], 0.0)
    return _plan_start_trajectory(planner, traj, reversed_, relative, *rest)


cffirmware.plan_start_trajectory = _plan_start_trajectory_compat

if args.backend == "isaac":
    import crazyflie_sim.backend  # noqa: F401
    import isaac_backend
    sys.modules["crazyflie_sim.backend.isaac"] = isaac_backend

from crazyflie_sim.crazyflie_server import main  # noqa: E402

main(["--ros-args", "--params-file", params_file.name] + [a for a in ros_args if a != "--ros-args"])
