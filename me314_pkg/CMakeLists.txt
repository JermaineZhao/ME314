cmake_minimum_required(VERSION 3.5)
project(me314_pkg)

# ==============================
# Dependencies
# ==============================
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(xarm_planner REQUIRED)
find_package(xarm_msgs REQUIRED)
find_package(xarm_moveit_config REQUIRED)
find_package(me314_msgs REQUIRED)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

# ==============================
# Install Python package
# ==============================
ament_python_install_package(me314_py)

# ==============================
# Install Python executables
# ==============================
install(PROGRAMS
  me314_py/xarm_api_scripts/move_A_to_B.py
  me314_py/teleoperation/xarm_spacemouse_ros2.py
  me314_py/teleoperation/spacemouse_perms.py
  me314_py/drivers/install_spacemouse.py
  me314_py/xarm_planner/xarm_a2b_example.py
  me314_py/xarm_planner/xarm_commander.py
  me314_py/xarm_planner/xarm_pick_n_place.py
  me314_py/xarm_planner/xarm_peg_in_hole.py
  DESTINATION lib/${PROJECT_NAME}
)

# ==============================
# Install launch files
# ==============================
install(
  DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
  PATTERN "__pycache__" EXCLUDE
)

# ==============================
# Install Gazebo models
# ==============================
install(
  DIRECTORY gazebo_models
  DESTINATION share/${PROJECT_NAME}
  PATTERN "__pycache__" EXCLUDE
)

# ==============================
# Install package.xml
# ==============================
install(
  FILES package.xml
  DESTINATION share/${PROJECT_NAME}
)

# ==============================
# Export dependencies
# ==============================
ament_export_dependencies(me314_msgs)

ament_package()
