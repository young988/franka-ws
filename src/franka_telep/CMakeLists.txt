cmake_minimum_required(VERSION 3.8)
project(franka_telep)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(THIS_PACKAGE_INCLUDE_DEPENDS
  controller_interface
  Eigen3
  pluginlib
  rclcpp
  rclcpp_lifecycle
  realtime_tools
  sensor_msgs
)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
foreach(Dependency IN ITEMS ${THIS_PACKAGE_INCLUDE_DEPENDS})
  find_package(${Dependency} REQUIRED)
endforeach()

add_library(franka_telep_controllers SHARED
  src/uarm_follower_controller.cpp
)
target_include_directories(franka_telep_controllers PUBLIC
  include
  ${EIGEN3_INCLUDE_DIRS}
)
ament_target_dependencies(franka_telep_controllers
  ${THIS_PACKAGE_INCLUDE_DEPENDS}
)

pluginlib_export_plugin_description_file(
  controller_interface franka_telep_controllers.xml
)

ament_python_install_package(${PROJECT_NAME})

install(TARGETS franka_telep_controllers
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(PROGRAMS
  scripts/franka_home_initializer
  scripts/franka_teleop
  scripts/openvla_dataset_recorder
  scripts/uarm_leader_publisher
  scripts/urdf_joint_state
  scripts/zhonglin_servo_reader
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  config
  launch
  openvla_rlds
  rviz
  urdf
  DESTINATION share/${PROJECT_NAME}
)

install(FILES franka_telep_controllers.xml
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(franka_telep_pytests test)
endif()

ament_export_include_directories(include)
ament_export_libraries(franka_telep_controllers)
ament_export_dependencies(${THIS_PACKAGE_INCLUDE_DEPENDS})

ament_package()
