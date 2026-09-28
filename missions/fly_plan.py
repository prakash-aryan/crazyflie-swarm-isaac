"""Crazyswarm2 mission: fly a planned swarm choreography.

Uploads each drone's piecewise polynomial trajectory (plans/<show>/<name>.csv), takes off (one drone after
another, or the whole swarm with a broadcast command when --stagger is 0), starts all trajectories with a single
broadcast (absolute coordinates) and lands. The same script flies real Crazyflies when the Crazyswarm2 server runs
with the hardware backend.

    python fly_plan.py --plan ../plans/office6 --ros-args -p use_sim_time:=True
"""
import argparse
import json
import os

from crazyflie_py import Crazyswarm
from crazyflie_py.uav_trajectory import Trajectory

ap = argparse.ArgumentParser()
ap.add_argument("--plan", required=True)
ap.add_argument("--idle", type=float, default=1.5, help="time on the pads before the first takeoff [s]")
ap.add_argument("--stagger", type=float, default=0.3,
                help="delay between individual takeoffs and landings [s]; 0 takes off and lands the whole swarm "
                     "together with broadcast commands")
args, _ = ap.parse_known_args()
with open(os.path.join(args.plan, "plan.json")) as f:
    plan = json.load(f)

swarm = Crazyswarm()
th, allcfs = swarm.timeHelper, swarm.allcfs
cfs = {cf.prefix.strip("/"): cf for cf in allcfs.crazyflies}
names = plan["names"]
for name in names:
    traj = Trajectory()
    traj.loadcsv(os.path.join(args.plan, f"{name}.csv"))
    cfs[name].uploadTrajectory(0, 0, traj)

th.sleep(args.idle)
if args.stagger > 0:
    for name in names:
        cfs[name].takeoff(targetHeight=plan["hover_height"], duration=2.5)
        th.sleep(args.stagger)
else:
    allcfs.takeoff(targetHeight=plan["hover_height"], duration=3.0)
th.sleep(3.0 if args.stagger > 0 else 3.5)
t0 = th.time()
allcfs.startTrajectory(0, timescale=1.0, relative=False)
th.sleep(plan["duration"] + 0.5)
if args.stagger > 0:
    for name in names:
        cfs[name].land(targetHeight=0.02, duration=2.5)
        th.sleep(args.stagger)
else:
    allcfs.land(targetHeight=0.02, duration=3.0)
th.sleep(3.5)
print(f"show started at sim time {t0:.2f} s, mission finished at {th.time():.2f} s", flush=True)
