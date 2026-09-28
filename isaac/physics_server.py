"""Isaac Sim physics backend for the Crazyswarm2 simulator (crazyflie_sim).

Listens on a local TCP port for the crazyflie_sim "isaac" backend (ros/isaac_backend.py). After a JSON handshake
with the drone names and initial positions it spawns one rigid body per Crazyflie on a ground plane, then runs in
lockstep with the firmware-in-the-loop server: every step it receives the four commanded motor speeds [rpm] and the
current setpoint of each drone, passes the commands through a first-order motor model (spin-up and spin-down time
constants of the Crazyflie 2 RotorS model), converts the motor speeds to rotor thrusts with the Crazyswarm2
system-identification fit, applies collective thrust and body torques, advances PhysX by dt and returns the drone states
(position, velocity, quaternion wxyz, body angular velocity). States, setpoints and motor speeds are recorded
to an .npz for rendering.

    python physics_server.py --port 47800 --record run.npz
"""
import argparse
import json
import os
import socket
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=47800)
ap.add_argument("--body", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "cf21",
                                               "cf21_body.usda"))
ap.add_argument("--record", required=True, help="output .npz with the recorded flight")
ap.add_argument("--record-every", type=int, default=4, help="record every k-th physics step")
ap.add_argument("--floor-z", type=float, default=0.0)
args = ap.parse_args()

from isaaclab.app import AppLauncher

app = AppLauncher(headless=True).app

import numpy as np
import torch
from pxr import Gf, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationCfg, SimulationContext

# Crazyswarm2 crazyflie_sim model (backend/np.py): motor layout and thrust-to-torque ratio
ARM = 0.707106781 * 0.046
T2T = 0.006
B0 = torch.tensor([[1.0, 1.0, 1.0, 1.0],
                   [-ARM, -ARM, ARM, ARM],
                   [-ARM, ARM, ARM, -ARM],
                   [-T2T, T2T, -T2T, T2T]], dtype=torch.float64)
RPM_TO_GRAMS = (2.55077341e-08, -4.92422570e-05, -1.51910248e-01)
FOOT = -0.0125  # motor mount feet below the centre of mass [m]
MOTOR_TAU_UP, MOTOR_TAU_DOWN = 0.0125, 0.025  # rotor speed time constants [s] (RotorS crazyflie2 model)


def rpm_to_force(rpm):
    grams = (RPM_TO_GRAMS[0] * rpm + RPM_TO_GRAMS[1]) * rpm + RPM_TO_GRAMS[2]
    return torch.clamp(grams * 9.81 / 1000.0, min=0.0)


def recv_exact(conn, n):
    buf = bytearray(n)
    view, got = memoryview(buf), 0
    while got < n:
        k = conn.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("backend disconnected")
        got += k
    return bytes(buf)


def readline(conn):
    data = b""
    while not data.endswith(b"\n"):
        chunk = conn.recv(1)
        if not chunk:
            raise ConnectionError("backend disconnected during handshake")
        data += chunk
    return json.loads(data)


server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("127.0.0.1", args.port))
server.listen(1)
print(f"[physics] listening on 127.0.0.1:{args.port}", flush=True)
conn, _ = server.accept()
conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
hello = readline(conn)
names, init = hello["names"], np.array(hello["initial_positions"], dtype=np.float64)
dt, n = float(hello["dt"]), len(hello["names"])
print(f"[physics] {n} drones, dt {dt}", flush=True)

sim = SimulationContext(SimulationCfg(dt=dt, device="cpu", gravity=(0.0, 0.0, -9.81)))
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg(), translation=(0.0, 0.0, args.floor_z))
stage = sim.stage
for i, name in enumerate(names):
    prim = stage.DefinePrim(f"/World/Drones/cf_{i:02d}", "Xform")
    prim.GetReferences().AddReference(os.path.abspath(args.body))
    xf = UsdGeom.Xformable(prim)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(init[i, 0], init[i, 1], args.floor_z + init[i, 2] - FOOT))
drones = RigidObject(RigidObjectCfg(prim_path="/World/Drones/cf_.*", spawn=None))
sim.reset()
assert drones.num_instances == n, f"expected {n} drone bodies, found {drones.num_instances}"

motor_rpm = torch.zeros(n, 4, dtype=torch.float64)
alpha_up, alpha_down = 1.0 - np.exp(-dt / MOTOR_TAU_UP), 1.0 - np.exp(-dt / MOTOR_TAU_DOWN)
force = torch.zeros(n, 1, 3)
torque = torch.zeros(n, 1, 3)
body_ids = [0]
rec = {k: [] for k in ("t", "pos", "quat", "vel", "omega", "rpm", "rpm_cmd", "setpoint")}


def states():
    d = drones.data
    pos = d.root_link_pos_w.torch.double()
    q = d.root_link_quat_w.torch.double()  # xyzw
    vel = d.root_link_lin_vel_w.torch.double()
    omega = d.root_link_ang_vel_b.torch.double()
    return torch.cat([pos, vel, q[:, [3, 0, 1, 2]], omega], dim=1)


conn.sendall((json.dumps({"ready": True, "states": states().tolist()}) + "\n").encode())
step, t, wall0 = 0, 0.0, time.time()
try:
    while True:
        msg = np.frombuffer(recv_exact(conn, n * 7 * 8), dtype=np.float64).reshape(n, 7)
        cmd = torch.from_numpy(msg[:, :4].copy())
        motor_rpm += (cmd - motor_rpm) * torch.where(cmd > motor_rpm, alpha_up, alpha_down)
        eta = rpm_to_force(motor_rpm) @ B0.T  # (n, 4): collective thrust, roll, pitch, yaw torques in the body frame
        force[:, 0, 2] = eta[:, 0].float()
        torque[:, 0, :] = eta[:, 1:].float()
        drones.permanent_wrench_composer.set_forces_and_torques_index(forces=force, torques=torque,
                                                                      body_ids=body_ids, is_global=False)
        drones.write_data_to_sim()
        sim.step(render=False)
        drones.update(dt)
        t += dt
        s = states()
        conn.sendall(s.numpy().tobytes())
        if step % args.record_every == 0:
            sn = s.numpy()
            rec["t"].append(t)
            rec["pos"].append(sn[:, 0:3].astype(np.float32))
            rec["vel"].append(sn[:, 3:6].astype(np.float32))
            rec["quat"].append(sn[:, 6:10].astype(np.float32))
            rec["omega"].append(sn[:, 10:13].astype(np.float32))
            rec["rpm"].append(motor_rpm.numpy().astype(np.float32))
            rec["rpm_cmd"].append(msg[:, :4].astype(np.float32))
            rec["setpoint"].append(msg[:, 4:7].astype(np.float32))
        step += 1
        if step % 5000 == 0:
            print(f"[physics] t {t:.1f} s, {step / (time.time() - wall0):.0f} steps/s", flush=True)
except ConnectionError as e:
    print(f"[physics] {e}", flush=True)
finally:
    out = {k: np.asarray(v) for k, v in rec.items()}
    np.savez_compressed(args.record, names=np.array(names), dt=dt, floor_z=args.floor_z, **out)
    print(f"[physics] saved {len(rec['t'])} samples ({t:.1f} s) to {args.record}", flush=True)
    os._exit(0)
