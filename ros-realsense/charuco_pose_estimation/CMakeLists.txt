cmake_minimum_required(VERSION 3.0.2)
project(charuco_pose_estimation)

find_package(catkin REQUIRED COMPONENTS
  rospy
  sensor_msgs
  geometry_msgs
  std_msgs
  cv_bridge
  message_generation
  tf
  image_transport
  roscpp
)

add_service_files(
  FILES
  CharUcoPose.srv
  TestResponse.srv
  ImageSaver.srv
  SpacecraftDetection.srv
)

generate_messages(
  DEPENDENCIES
  sensor_msgs geometry_msgs std_msgs
)

catkin_package(
  CATKIN_DEPENDS
  cv_bridge
  geometry_msgs
  image_transport
  message_runtime
  roscpp
  sensor_msgs
  std_msgs
)

catkin_install_python(PROGRAMS
  src/charuco_pose_server_stream.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)