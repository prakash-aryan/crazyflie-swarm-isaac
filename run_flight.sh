#!/usr/bin/env bash
# Fly a planned show with Crazyswarm2 (Crazyflie firmware in the loop) and Isaac Sim physics, and record the flight.
#   ./run_flight.sh <crazyflies.yaml> <plan directory> <record.npz> [fly_plan.py options]
# Environment: ISAAC_PYTHON (Isaac Lab Python, default ~/IsaacLab/.venv/bin/python), CF_ISAAC_PORT (default 47800),
# CF_RECORD_EVERY (record every k-th physics step, default 4).
CF_ROOT=$(cd "$(dirname "$0")" && pwd)
YAML=$(realpath "$1"); PLAN=$(realpath "$2"); RECORD=$(realpath -m "$3"); shift 3
ISAAC_PYTHON=${ISAAC_PYTHON:-$HOME/IsaacLab/.venv/bin/python}
LOGDIR=$(dirname "$RECORD"); NAME=$(basename "$RECORD" .npz)
mkdir -p "$LOGDIR"
export CF_ISAAC_ADDR=127.0.0.1:${CF_ISAAC_PORT:-47800}

OMNI_KIT_ACCEPT_EULA=YES PYTHONNOUSERSITE=1 "$ISAAC_PYTHON" "$CF_ROOT/isaac/physics_server.py" \
    --port "${CF_ISAAC_ADDR##*:}" --record "$RECORD" --record-every "${CF_RECORD_EVERY:-4}" \
    > "$LOGDIR/$NAME.physics.log" 2>&1 &
PHYS=$!
until grep -q "listening" "$LOGDIR/$NAME.physics.log" 2>/dev/null; do
    kill -0 $PHYS 2>/dev/null || { echo "physics server exited, see $LOGDIR/$NAME.physics.log"; exit 1; }
    sleep 1
done

source "$CF_ROOT/ros/env.sh"
python "$CF_ROOT/ros/run_server.py" --crazyflies "$YAML" --backend isaac > "$LOGDIR/$NAME.server.log" 2>&1 &
SERVER=$!
until grep -q "Isaac Sim physics ready" "$LOGDIR/$NAME.server.log" 2>/dev/null; do
    if ! kill -0 $SERVER 2>/dev/null; then
        echo "Crazyswarm2 server exited, see $LOGDIR/$NAME.server.log"
        kill $PHYS
        exit 1
    fi
    sleep 1
done

python "$CF_ROOT/missions/fly_plan.py" --plan "$PLAN" "$@" --ros-args -p use_sim_time:=True 2>&1 \
    | grep -v "No joystick" | tee "$LOGDIR/$NAME.mission.log"
kill -INT $SERVER
sleep 2
kill -9 $SERVER 2>/dev/null
wait $PHYS
tail -1 "$LOGDIR/$NAME.physics.log"
