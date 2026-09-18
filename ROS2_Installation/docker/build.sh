#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

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

cd "${ROOT_DIR}"

retry_command 3 docker build -t aliengo-base docker/base
retry_command 3 docker build -t aliengo-isaac-gym docker/isaac-gym
retry_command 3 docker build -t aliengo-competition -f docker/Dockerfile .
