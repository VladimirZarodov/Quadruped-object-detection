FROM osrf/ros:jazzy-desktop-full

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=jazzy
ENV ROS_WS=/workspace/aliengo_competition/ros2_isaac_bridge/ros2_ws

RUN set -eux; \
    success=0; \
    for attempt in 1 2 3; do \
        apt-get update && \
        apt-get install -y --no-install-recommends \
            python3-colcon-common-extensions \
            python3-opencv \
            python3-numpy \
            ros-jazzy-rviz2 \
            ros-jazzy-image-transport-plugins \
            ros-jazzy-rqt \
            ros-jazzy-rqt-common-plugins \
        && success=1 && break; \
        echo "apt install attempt ${attempt} failed; retrying..." >&2; \
        rm -rf /var/lib/apt/lists/*; \
        sleep 5; \
    done; \
    test "${success}" = "1"; \
    apt-get clean; \
    rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    printf '%s\n' \
        '#!/usr/bin/env bash' \
        'set +u' \
        'source /opt/ros/jazzy/setup.bash' \
        'if [[ -f /workspace/aliengo_competition/ros2_isaac_bridge/ros2_ws/install/setup.bash ]]; then' \
        '    source /workspace/aliengo_competition/ros2_isaac_bridge/ros2_ws/install/setup.bash' \
        'fi' \
        'set -u' \
        > /etc/profile.d/ros2_isaac_bridge.sh; \
    chmod 0644 /etc/profile.d/ros2_isaac_bridge.sh; \
    printf '\n# Auto-source ROS 2 Jazzy and the bridge workspace\nif [ -f /etc/profile.d/ros2_isaac_bridge.sh ]; then\n  . /etc/profile.d/ros2_isaac_bridge.sh\nfi\n' >> /root/.bashrc

WORKDIR /workspace/aliengo_competition

CMD ["bash", "-lc", "tail -f /dev/null"]
