"""Twenty-Crazyflie formation show in the NVIDIA SimReady warehouse, in the style of the Crazyswarm demos.

The swarm takes off together from a 4 x 5 grid on the open floor between the loading and sorting areas
(centre (-35.5, -58.6); clear from the floor to about 6 m, 4 m to the nearest object at 4 m height) and flies
3D formations: grid wave, spinning ring, helix, rotating dodecahedron (20 vertices), a figure-8 chase,
then returns to the grid and lands together.

    python show_warehouse20.py --voxels ../maps/warehouse_voxels.npz
"""
import argparse
import json
import os

import numpy as np

import choreo as C

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--out", default=os.path.join(HERE, "..", "plans", "warehouse20"))
ap.add_argument("--config", default=os.path.join(HERE, "..", "config", "warehouse20.yaml"))
ap.add_argument("--voxels", default=os.path.join(HERE, "..", "maps", "warehouse_voxels.npz"),
                help="clearance map from scene_voxels.py")
args = ap.parse_args()

N = 20
CX, CY = -35.5, -58.6
HOVER = 1.0
names = [f"cf{i + 1}" for i in range(N)]
pads = C.grid((CX, CY, 0.0), rows=4, cols=5, dx=0.6, dy=0.6)
start = pads + [0.0, 0.0, HOVER]


def ramp(s, r=0.2):
    """Phase law from 0 to 1: raised-cosine acceleration over the first and last fraction r, constant speed in
    between (peak speed 1 / (1 - r) times the average, against 1.875 for a minimum-jerk law)."""
    s = float(np.clip(s, 0.0, 1.0))
    v_peak = 1.0 / (1.0 - r)
    if s < r:
        return v_peak * (s / 2 - r / (2 * np.pi) * np.sin(np.pi * s / r))
    if s > 1 - r:
        return 1.0 - ramp(1.0 - s, r)
    return v_peak * (r / 2 + (s - r))


def rotation(axis, angle):
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def helix(center, n, radius, z0, z1, turns):
    k = np.arange(n)
    a = 2 * np.pi * turns * k / n
    return np.column_stack([center[0] + radius * np.cos(a), center[1] + radius * np.sin(a),
                            z0 + (z1 - z0) * k / (n - 1)])


def dodecahedron(center, radius):
    phi = (1 + 5 ** 0.5) / 2
    v = [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    for a in (-1, 1):
        for b in (-1, 1):
            v += [(0, a / phi, b * phi), (a / phi, b * phi, 0), (a * phi, 0, b / phi)]
    return np.asarray(center, float) + np.array(v) * radius / 3 ** 0.5


def figure8(center, n, a, z_twist, phase):
    """Horizontal figure-8 (lemniscate of Gerono) with the two branches at different heights where they cross."""
    t = phase + 2 * np.pi * np.arange(n) / n
    return np.column_stack([center[0] + a * np.sin(t), center[1] + 0.5 * a * np.sin(2 * t),
                            center[2] + z_twist * np.cos(t)])


show = C.Show(start)
show.hold(0.5, "hover over the pads")
show.animate(2.5, lambda s, p0: p0 + [0.0, 0.0, 0.4 * C.minjerk(s)], "grid rises to 1.4 m")


def grid_wave(s, p0):
    p = p0.copy()
    phase = 2 * np.pi * (p0[:, 0] - CX) / 2.4 + 2 * np.pi * (p0[:, 1] - CY) / 4.0 - 4 * np.pi * s
    p[:, 2] += 0.35 * np.sin(phase) * C.bump(s)
    return p


show.animate(8.0, grid_wave, "wave rolls across the grid")
RING = np.array([CX, CY, 2.1])
show.morph(5.0, C.ring(RING, N, 1.9), "grid expands into a 3.8 m ring")


def ring_spin(s, p0):
    rel = p0 - RING
    th = np.arctan2(rel[:, 1], rel[:, 0]) + np.pi * C.minjerk(s)
    r = np.hypot(rel[:, 0], rel[:, 1])
    return np.column_stack([RING[0] + r * np.cos(th), RING[1] + r * np.sin(th),
                            RING[2] + 0.35 * np.sin(3 * th) * C.bump(s)])


show.animate(8.0, ring_spin, "ring spins half a turn with a triple wave")
HELIX = np.array([CX, CY, 0.0])
show.morph(5.0, helix(HELIX, N, 1.1, 1.3, 3.5, 2.0), "ring winds into a 2.2 m helix")


def helix_spin(s, p0):
    rel = p0 - HELIX
    th = np.arctan2(rel[:, 1], rel[:, 0]) + 2 * np.pi * C.minjerk(s)
    r = np.hypot(rel[:, 0], rel[:, 1])
    return np.column_stack([HELIX[0] + r * np.cos(th), HELIX[1] + r * np.sin(th), p0[:, 2]])


show.animate(10.0, helix_spin, "helix turns once")
DODE = np.array([CX, CY, 2.4])
show.morph(5.0, dodecahedron(DODE, 1.3), "helix folds into a dodecahedron")
AXIS = (0.35, 0.2, 1.0)


def dode_spin(s, p0):
    R = rotation(AXIS, 2 * np.pi * C.minjerk(s))
    scale = 1.0 + 0.3 * C.bump(s)  # breathes out and back in
    return DODE + ((p0 - DODE) @ R.T) * scale


show.animate(12.0, dode_spin, "dodecahedron turns once about a tilted axis and breathes")
F8 = np.array([CX, CY, 2.0])
show.morph(5.0, figure8(F8, N, 1.9, 0.35, 0.0), "dodecahedron unrolls onto a figure-8")


def chase(s, p0):
    slot = np.argmin(np.linalg.norm(p0[:, None, :] - figure8(F8, N, 1.9, 0.35, 0.0)[None], axis=-1), axis=1)
    return figure8(F8, N, 1.9, 0.35, 2 * np.pi * ramp(s))[slot]


show.animate(14.0, chase, "figure-8 chase, one lap")
show.morph(5.0, start + [0.0, 0.0, 0.4], "back to the grid")
show.animate(2.0, lambda s, p0: p0 - [0.0, 0.0, 0.4 * C.minjerk(s)], "grid settles to 1 m")
show.hold(0.5, "hover over the pads")

paths = show.paths()
report = C.validate(paths, C.Clearance(args.voxels), min_clear=0.5, min_sep=0.4)
plan = C.save_plan(args.out, names, pads, HOVER, paths, show.events, report)
C.write_crazyflies_yaml(args.config, names, pads)
print(json.dumps({k: plan[k] for k in ("duration", "pieces", "fit_error", "validation")}, indent=1))
for name, a, b in plan["events"]:
    print(f"  {a:6.2f}-{b:6.2f}  {name}")
