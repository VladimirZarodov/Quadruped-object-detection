#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/compose.local.yml"
COMPOSE_VIZ_FILE="${SCRIPT_DIR}/compose.viz.yml"
COMPOSE_ROS2_FILE="${SCRIPT_DIR}/compose.ros2.yml"

usage() {
  cat <<'EOF'
Usage:
  docker/ctl.sh build   # build local docker layers and the competition image
  docker/ctl.sh up      # build then start the competition container with visualization
  docker/ctl.sh down    # stop and remove the competition container
  docker/ctl.sh exec    # open a shell inside the running container
  docker/ctl.sh ros2-build  # build the ROS 2 Jazzy layer (desktop-full + rqt + tools)
  docker/ctl.sh ros2-up     # start ROS 2 Jazzy container with X11 support
  docker/ctl.sh ros2-down   # stop and remove ROS 2 Jazzy container
  docker/ctl.sh ros2-exec   # open a shell inside the ROS 2 Jazzy container
EOF
}

retry_command() {
  local max_attempts="$1"
  shift

  local attempt=1
  while true; do
    if "$@"; then
      return 0
    fi

    if (( attempt >= max_attempts )); then
      echo "Command failed after ${attempt} attempts: $*" >&2
      return 1
    fi

    echo "Command failed (attempt ${attempt}/${max_attempts}): $*" >&2
    echo "Retrying in 5 seconds..." >&2
    attempt=$((attempt + 1))
    sleep 5
  done
}

build_layers() {
  bash "${SCRIPT_DIR}/build.sh"
}

compose() {
  docker compose -p aliengo-sim -f "${COMPOSE_FILE}" -f "${COMPOSE_VIZ_FILE}" "$@"
}

compose_ros2() {
  docker compose -p aliengo-ros2 -f "${COMPOSE_ROS2_FILE}" "$@"
}

ensure_x11_access() {
  if [[ -z "${DISPLAY:-}" ]]; then
    echo "DISPLAY is not set. Visualization is required; export DISPLAY first (example: export DISPLAY=:0)." >&2
    exit 1
  fi
  if ! command -v xhost >/dev/null 2>&1; then
    echo "xhost is not installed. Install xhost to grant X11 access for Docker visualization." >&2
    exit 1
  fi

  # Allow local Docker clients to connect to the host X server.
  if ! xhost +local:docker >/dev/null 2>&1 && ! xhost +SI:localuser:root >/dev/null 2>&1; then
    echo "Failed to grant X11 access via xhost. Run xhost manually and retry." >&2
    exit 1
  fi
}

cmd="${1:-}"
case "${cmd}" in
  build)
    build_layers
    retry_command 3 compose build
    ;;
  up)
    build_layers
    ensure_x11_access
    compose up -d
    ;;
  down)
    compose down
    ;;
  exec)
    ensure_x11_access
    compose exec aliengo-competition bash
    ;;
  ros2-build)
    retry_command 3 compose_ros2 build
    ;;
  ros2-up)
    ensure_x11_access
    compose_ros2 up -d
    ;;
  ros2-down)
    compose_ros2 down
    ;;
  ros2-exec)
    ensure_x11_access
    compose_ros2 exec ros2-jazzy bash
    ;;
  *)
    usage
    exit 1
    ;;
esac
