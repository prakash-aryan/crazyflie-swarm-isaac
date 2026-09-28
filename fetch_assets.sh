#!/usr/bin/env bash
# Download assets into ./assets.
#   ./fetch_assets.sh crazyflie   Crazyflie 2 meshes of MuJoCo Menagerie (MIT), 1 MB
#   ./fetch_assets.sh warehouse   NVIDIA PhysicalAI SimReady Warehouse 01 (CC-BY-4.0), ~14 GB
#   ./fetch_assets.sh office      open-plan office of GRScenes-100 (CC BY-NC-SA 4.0), ~24 GB download, ~3 GB kept
set -euo pipefail
DEST="$(cd "$(dirname "$0")" && pwd)/assets"
MENAGERIE=https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main/bitcraze_crazyflie_2
case "${1:-}" in
crazyflie)
    mkdir -p "$DEST/crazyflie_2"
    for i in 0 1 2 3 4 5 6; do
        curl -fsSL -o "$DEST/crazyflie_2/cf2_$i.obj" "$MENAGERIE/assets/cf2_$i.obj"
    done
    curl -fsSL -o "$DEST/crazyflie_2/LICENSE" "$MENAGERIE/LICENSE"
    ;;
warehouse)
    hf download nvidia/PhysicalAI-SimReady-Warehouse-01 --repo-type dataset --local-dir "$DEST/simready_warehouse"
    ;;
office)
    hf download InternRobotics/GRScenes --repo-type dataset \
        --include "scenes/GRScenes-100/commercial_scenes.zip" --local-dir "$DEST/grscenes"
    cd "$DEST/grscenes/scenes/GRScenes-100"
    unzip -q -o commercial_scenes.zip "*MWF4WLIKTIFZIAABAAAAAEA8_usd/*"
    rm -f commercial_scenes.zip
    ;;
*)
    echo "usage: $0 crazyflie|warehouse|office"
    exit 1
    ;;
esac
