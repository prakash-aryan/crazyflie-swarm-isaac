"""Swarm choreography planning for Crazyswarm2: formations, transitions, polynomial trajectories and validation.

A show is a sequence of segments that all start and end at rest, so segment boundaries are continuous in position,
velocity and acceleration. Formation changes use concurrent assignment and planning (CAPT): drones are assigned
to target slots by minimum sum of squared distances and move on synchronised straight lines with a minimum-jerk
time law. Each drone's path is fitted with 7th-order polynomial pieces (Crazyflie high-level commander format,
C2-continuous across pieces) and checked against a clearance map of the room and against the other drones.
"""
import json
import os

import numpy as np
from scipy.optimize import linear_sum_assignment

RATE = 100.0  # sampling rate of the reference paths [Hz]


def minjerk(s):
    s = np.clip(s, 0.0, 1.0)
    return s ** 3 * (10 - 15 * s + 6 * s ** 2)


def bump(s):
    """Smooth envelope that is zero with zero first and second derivatives at s = 0 and s = 1, peak 1 at s = 0.5."""
    s = np.clip(s, 0.0, 1.0)
    return 64.0 * s ** 3 * (1 - s) ** 3


# ---- formations: functions returning (N, 3) positions


def line(center, n, spacing, axis=(1.0, 0.0, 0.0)):
    u = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.arange(n) - (n - 1) / 2
    return np.asarray(center, float) + k[:, None] * spacing * u


def ring(center, n, radius, phase=0.0):
    a = phase + 2 * np.pi * np.arange(n) / n
    return np.asarray(center, float) + np.stack([radius * np.cos(a), radius * np.sin(a), np.zeros(n)], axis=1)


def grid(center, rows, cols, dx, dy):
    ix, iy = np.meshgrid(np.arange(cols) - (cols - 1) / 2, np.arange(rows) - (rows - 1) / 2)
    return np.asarray(center, float) + np.stack([ix.ravel() * dx, iy.ravel() * dy, np.zeros(rows * cols)], axis=1)


def arrow(center, n, spacing, heading=0.0):
    """Arrowhead with the apex at the centre, pointing along heading: symmetric wings, and a leftover drone
    (even n) on the centre line behind the apex."""
    pts, k = [(0.0, 0.0)], 1
    while n - len(pts) >= 2:
        pts += [(-k * spacing * 0.8, k * spacing * 0.75), (-k * spacing * 0.8, -k * spacing * 0.75)]
        k += 1
    if len(pts) < n:
        pts.append((-(k - 1) * spacing * 0.8, 0.0))
    p = np.array(pts)
    c, s = np.cos(heading), np.sin(heading)
    xy = p @ np.array([[c, s], [-s, c]])
    return np.asarray(center, float) + np.column_stack([xy, np.zeros(n)])


# ---- show construction


class Show:
    def __init__(self, start):
        self.pos = np.asarray(start, float)  # current positions, indexed by drone
        self.samples = [self.pos.copy()]
        self.events = []  # (name, t_start, t_end)

    @property
    def t(self):
        return (len(self.samples) - 1) / RATE

    def _append(self, name, duration, fn):
        t0, n = self.t, int(round(duration * RATE))
        for k in range(1, n + 1):
            self.samples.append(fn(k / n))
        self.pos = self.samples[-1].copy()
        self.events.append((name, t0, self.t))

    def hold(self, duration, name="hold"):
        p = self.pos.copy()
        self._append(name, duration, lambda s: p)

    def morph(self, duration, target, name="morph", lift=0.0):
        """Move to the target formation (drone-to-slot assignment by minimum sum of squared distances). lift adds
        a height offset [m], per drone or common, that peaks halfway through the transition."""
        start, target = self.pos.copy(), np.asarray(target, float)
        cost = ((start[:, None, :] - target[None, :, :]) ** 2).sum(-1)
        _, slot = linear_sum_assignment(cost)
        goal = target[slot]
        lift = np.outer(np.broadcast_to(np.asarray(lift, float), len(start)), [0.0, 0.0, 1.0])
        self._append(name, duration, lambda s: start + (goal - start) * minjerk(s) + bump(s) * lift)
        return slot

    def animate(self, duration, fn, name="animate"):
        """fn(s, p0) -> positions, with fn(0, p0) == p0 and zero velocity at s = 0 and s = 1."""
        p0 = self.pos.copy()
        self._append(name, duration, lambda s: fn(s, p0))

    def paths(self):
        return np.array(self.samples)  # (T, N, 3)


# ---- polynomial fitting (Crazyflie piecewise polynomial format)


def _fit_piece(t, y, start, end=None):
    """Least-squares degree-7 fit on [0, T] with position, velocity and acceleration constraints at the start (and
    optionally at the end)."""
    T = t[-1]
    A = np.vander(t, 8, increasing=True)

    def rows(tt):
        p = np.array([tt ** i for i in range(8)])
        v = np.array([i * tt ** (i - 1) if i > 0 else 0.0 for i in range(8)])
        a = np.array([i * (i - 1) * tt ** (i - 2) if i > 1 else 0.0 for i in range(8)])
        return np.stack([p, v, a])

    C, d = [rows(0.0)], [np.asarray(start)]
    if end is not None:
        C.append(rows(T))
        d.append(np.asarray(end))
    C, d = np.vstack(C), np.concatenate(d)
    K = np.block([[2 * A.T @ A, C.T], [C, np.zeros((len(C), len(C)))]])
    rhs = np.concatenate([2 * A.T @ y, d])
    return np.linalg.solve(K, rhs)[:8]


def fit_path(path, max_pieces=31, min_piece=0.5):
    """Fit one drone's (T, 3) path; returns a list of (duration, px, py, pz).

    Both ends of every piece match the reference position, velocity and acceleration at the piece boundary, so the
    trajectory is C2-continuous and fitting errors do not accumulate from piece to piece.
    """
    total = (len(path) - 1) / RATE
    n_pieces = int(min(max_pieces, max(1, np.ceil(total / 2.0)), np.floor(total / min_piece)))
    bounds = np.round(np.linspace(0, len(path) - 1, n_pieces + 1)).astype(int)
    vel = np.gradient(path, 1.0 / RATE, axis=0)
    acc = np.gradient(vel, 1.0 / RATE, axis=0)
    vel[[0, -1]], acc[[0, -1]] = 0.0, 0.0  # shows start and end at rest
    pieces = []
    for j in range(n_pieces):
        i0, i1 = bounds[j], bounds[j + 1]
        t = np.arange(i1 - i0 + 1) / RATE
        coeffs = [_fit_piece(t, path[i0:i1 + 1, a], (path[i0, a], vel[i0, a], acc[i0, a]),
                             (path[i1, a], vel[i1, a], acc[i1, a])) for a in range(3)]
        pieces.append((t[-1], *coeffs))
    return pieces


def eval_pieces(pieces, rate=RATE):
    out = []
    for T, px, py, pz in pieces:
        t = np.arange(int(round(T * rate))) / rate
        out.append(np.stack([np.polyval(px[::-1], t), np.polyval(py[::-1], t), np.polyval(pz[::-1], t)], axis=1))
    T, px, py, pz = pieces[-1]
    out.append(np.array([[np.polyval(px[::-1], T), np.polyval(py[::-1], T), np.polyval(pz[::-1], T)]]))
    return np.concatenate(out)


def write_csv(pieces, path):
    header = "duration," + ",".join(f"{ax}^{i}" for ax in ("x", "y", "z", "yaw") for i in range(8)) + ","
    with open(path, "w") as f:
        f.write(header + "\n")
        for T, px, py, pz in pieces:
            row = [T, *px, *py, *pz, *np.zeros(8)]
            f.write(",".join(f"{v:.9f}" for v in row) + ",\n")


# ---- validation


class Clearance:
    def __init__(self, npz):
        d = np.load(npz)
        self.c, self.o, self.r = d["clearance"].astype(np.float32), d["origin"], float(d["res"])

    def __call__(self, p):
        i = np.clip(((p - self.o) / self.r).astype(int), 0, np.array(self.c.shape) - 1)
        return self.c[i[..., 0], i[..., 1], i[..., 2]]


def validate(paths, clearance, min_clear=0.3, min_sep=0.35, above=0.5):
    """paths: (T, N, 3). Clearance is checked for samples above `above` m (takeoff and landing are near the floor)."""
    c = clearance(paths)
    high = paths[..., 2] > above
    worst_clear = float(c[high].min()) if high.any() else float("inf")
    d = np.linalg.norm(paths[:, :, None, :] - paths[:, None, :, :], axis=-1)
    n = paths.shape[1]
    d[:, np.arange(n), np.arange(n)] = np.inf
    worst_sep = float(d.min())
    k, i, j = np.unravel_index(int(np.argmin(d)), d.shape)
    ki, di = np.unravel_index(int(np.argmin(np.where(high, c, np.inf))), c.shape)
    speed = np.linalg.norm(np.diff(paths, axis=0), axis=-1) * RATE
    return {"min_clearance": round(worst_clear, 3), "min_clearance_at": [round(ki / RATE, 2), int(di)],
            "min_separation": round(worst_sep, 3), "min_separation_at": [round(k / RATE, 2), int(i), int(j)],
            "max_speed": round(float(speed.max()), 2), "ok": worst_clear >= min_clear and worst_sep >= min_sep}


def save_plan(out_dir, names, pads, hover, paths, events, report):
    os.makedirs(out_dir, exist_ok=True)
    max_err = 0.0
    for i, name in enumerate(names):
        pieces = fit_path(paths[:, i, :])
        write_csv(pieces, os.path.join(out_dir, f"{name}.csv"))
        ev = eval_pieces(pieces)
        m = min(len(ev), len(paths))
        max_err = max(max_err, float(np.linalg.norm(ev[:m] - paths[:m, i, :], axis=1).max()))
    plan = {"names": names, "pads": np.asarray(pads).round(4).tolist(), "hover_height": hover,
            "duration": round((len(paths) - 1) / RATE, 3), "pieces": len(pieces), "fit_error": round(max_err, 5),
            "events": [(n, round(a, 2), round(b, 2)) for n, a, b in events], "validation": report}
    with open(os.path.join(out_dir, "plan.json"), "w") as f:
        json.dump(plan, f, indent=1)
    np.save(os.path.join(out_dir, "reference.npy"), paths.astype(np.float32))
    return plan


def write_crazyflies_yaml(path, names, pads):
    lines = ["fileversion: 3", "robots:"]
    for i, (name, p) in enumerate(zip(names, pads)):
        lines += [f"  {name}:", "    enabled: true", f"    uri: radio://0/80/2M/E7E7E7E7{i + 1:02X}",
                  f"    initial_position: [{p[0]:.4f}, {p[1]:.4f}, 0.0]", "    type: cf21"]
    lines += ["robot_types:", "  cf21:", "    motion_capture:", "      tracking: \"librigidbodytracker\"",
              "      marker: default_single_marker", "      dynamics: default", "    big_quad: false",
              "    battery:", "      voltage_warning: 3.8", "      voltage_critical: 3.7",
              "all:", "  firmware_logging:", "    enabled: false", "  firmware_params:", "    commander:",
              "      enHighLevel: 1", "    stabilizer:", "      estimator: 2", "      controller: 2",
              "  reference_frame: \"world\"", "  broadcasts:", "    num_repeats: 15", "    delay_between_repeats_ms: 1"]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
