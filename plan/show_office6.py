"""Six-Crazyflie show in the open-plan office (GRScenes MWF4WLIKTIFZIAABAAAAAEA8).

The drones take off from pads in the east aisle and fly in the corridor between the first two desk rows
(y = -3.4 m), which is clear from 1.35 m to 1.9 m above the floor along its full 8.5 m length.

    python show_office6.py --voxels ../maps/office_voxels.npz
"""
import argparse
import json
import os

import numpy as np

import choreo as C

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(HERE, "..", "plans", "office6"))
ap.add_argument("--config", default=os.path.join(HERE, "..", "config", "office6.yaml"))
ap.add_argument("--voxels", default=os.path.join(HERE, "..", "maps", "office_voxels.npz"),
                help="clearance map from scene_voxels.py")
args = ap.parse_args()

N = 6
CY = -3.4  # corridor centre line
HOVER = 0.8
names = [f"cf{i + 1}" for i in range(N)]
pads = C.grid((4.35, CY, 0.0), rows=3, cols=2, dx=0.6, dy=0.6)
start = pads + [0.0, 0.0, HOVER]

show = C.Show(start)
show.hold(0.5, "hover over the pads")
show.animate(3.0, lambda s, p0: p0 + [0.0, 0.0, 0.8 * C.minjerk(s)], "climb to corridor height in the aisle")
show.morph(5.0, C.line((2.0, CY, 1.6), N, 0.6), "into the corridor as a line")


def glide(s, p0):
    p = p0 + [-2.2 * C.minjerk(s), 0.0, 0.0]
    p[:, 2] += 0.18 * np.sin(2 * np.pi * p0[:, 0] / 1.8 - 3 * np.pi * s) * C.bump(s)
    return p


show.animate(7.0, glide, "line glides west with a travelling wave")
RC = np.array([-0.6, CY, 1.6])
show.morph(4.0, C.ring(RC, N, 0.75, phase=np.pi / 6), "line to ring")


def spin(s, p0):
    rel = p0 - RC
    th = np.arctan2(rel[:, 1], rel[:, 0]) + 2 * np.pi * C.minjerk(s)
    r = np.hypot(rel[:, 0], rel[:, 1])
    return np.stack([RC[0] + r * np.cos(th), RC[1] + r * np.sin(th), RC[2] + 0.2 * np.sin(2 * th) * C.bump(s)], 1)


show.animate(10.0, spin, "ring spins one turn with a crown wave")
lower = C.ring((RC[0], CY, 1.4), 3, 0.6, np.pi / 2)
upper = C.ring((RC[0], CY, 1.85), 3, 0.6, np.pi / 2 + np.pi / 3)
show.morph(4.0, np.vstack([lower, upper]), "ring to two stacked triangles")


def counter(s, p0):
    out = p0.copy()
    shift = 2.0 * C.minjerk(s)
    for i in range(N):
        rel = p0[i, :2] - RC[:2]
        th = np.arctan2(rel[1], rel[0]) + (1 if p0[i, 2] < 1.6 else -1) * (2 * np.pi / 3) * C.minjerk(s)
        out[i, :2] = RC[:2] + [shift, 0.0] + np.hypot(*rel) * np.array([np.cos(th), np.sin(th)])
    return out


show.animate(8.0, counter, "triangles counter-rotate while moving east")
show.morph(4.0, C.arrow((2.4, CY, 1.6), N, 0.45), "arrowhead")
show.animate(4.0, lambda s, p0: p0 + [0.9 * C.minjerk(s), 0.0, 0.0], "arrowhead flies east")
show.morph(5.0, start + [0.0, 0.0, 0.8], "back to the aisle at corridor height")
show.animate(3.0, lambda s, p0: p0 - [0.0, 0.0, 0.8 * C.minjerk(s)], "descend over the pads")
show.hold(0.5, "hover over the pads")

paths = show.paths()
report = C.validate(paths, C.Clearance(args.voxels))
plan = C.save_plan(args.out, names, pads, HOVER, paths, show.events, report)
C.write_crazyflies_yaml(args.config, names, pads)
print(json.dumps({k: plan[k] for k in ("duration", "pieces", "fit_error", "validation")}, indent=1))
for name, a, b in plan["events"]:
    print(f"  {a:6.2f}-{b:6.2f}  {name}")
