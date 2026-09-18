#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS_DIR="${SCRIPT_DIR}/ros2_ws"

set +u
source /opt/ros/jazzy/setup.bash
set -u

cd "${WS_DIR}"
colcon build --symlink-install

set +u
source "${WS_DIR}/install/setup.bash"
set -u

exec ros2 run ros2_bridge_pkg bridge_node
