import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/ammar/roboschool/ros2_isaac_bridge/ros2_ws/install/ros2_bridge_pkg'
