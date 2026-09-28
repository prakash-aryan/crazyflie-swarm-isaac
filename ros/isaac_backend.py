"""crazyflie_sim backend that delegates the rigid-body dynamics to Isaac Sim (isaac/physics_server.py).

Runs in lockstep with the physics server over a local TCP socket: every step sends the four motor speeds and the
setpoint position of each Crazyflie, receives the next states and publishes /clock, so ROS 2 nodes started with
use_sim_time follow simulation time even when Isaac Sim runs slower than real time.

Environment: CF_ISAAC_ADDR (default 127.0.0.1:47800), CF_ISAAC_DT (default 0.001 s).
"""
from __future__ import annotations

import json
import os
import socket

import numpy as np
from rclpy.node import Node
from rclpy.time import Time
from rosgraph_msgs.msg import Clock

from crazyflie_sim.sim_data_types import Action, State


def _recv_exact(sock, n):
    buf = bytearray(n)
    view, got = memoryview(buf), 0
    while got < n:
        k = sock.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("Isaac Sim physics server closed the connection")
        got += k
    return bytes(buf)


class Backend:
    """Backend that steps Isaac Sim PhysX in lockstep with the firmware-in-the-loop controllers."""

    def __init__(self, node: Node, names: list[str], states: list[State]):
        self.node = node
        self.names = names
        self.n = len(names)
        self.clock_publisher = node.create_publisher(Clock, 'clock', 10)
        self.t = 0.0
        self.dt = float(os.environ.get('CF_ISAAC_DT', '0.001'))
        host, port = os.environ.get('CF_ISAAC_ADDR', '127.0.0.1:47800').rsplit(':', 1)
        self.sock = socket.create_connection((host, int(port)), timeout=600.0)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        hello = {'names': names, 'initial_positions': [np.asarray(s.pos).tolist() for s in states], 'dt': self.dt}
        self.sock.sendall((json.dumps(hello) + '\n').encode())
        line = b''
        while not line.endswith(b'\n'):
            line += _recv_exact(self.sock, 1)
        reply = json.loads(line)
        node.get_logger().info(f'Isaac Sim physics ready for {self.n} Crazyflies (dt {self.dt} s)')
        self._states = [State() for _ in range(self.n)]
        self._set_states(np.array(reply['states'], dtype=np.float64))

    def _set_states(self, arr):
        for s, a in zip(self._states, arr):
            s.pos, s.vel, s.quat, s.omega = a[0:3], a[3:6], a[6:10], a[10:13]

    def time(self) -> float:
        return self.t

    def step(self, states_desired: list[State], actions: list[Action]) -> list[State]:
        msg = np.zeros((self.n, 7), dtype=np.float64)
        for i, (desired, action) in enumerate(zip(states_desired, actions)):
            if action is not None:
                msg[i, :4] = action.rpm
            if desired is not None:
                msg[i, 4:7] = desired.pos
        self.sock.sendall(msg.tobytes())
        arr = np.frombuffer(_recv_exact(self.sock, self.n * 13 * 8), dtype=np.float64).reshape(self.n, 13)
        self._set_states(arr)
        self.t += self.dt
        clock_message = Clock()
        clock_message.clock = Time(seconds=self.time()).to_msg()
        self.clock_publisher.publish(clock_message)
        return self._states

    def shutdown(self):
        self.sock.close()
