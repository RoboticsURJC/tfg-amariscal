cmake_minimum_required(VERSION 3.0.2)
project(circuit_simulation)

find_package(catkin REQUIRED COMPONENTS
  catkin
	std_msgs
	rospy
	sensor_msgs
	gazebo_msgs
	cv_bridge
	gazebo
)

catkin_package(
	CATKIN_DEPENDS
	catkin
	std_msgs
	rospy
	sensor_msgs
	gazebo_msgs
	cv_bridge
)

catkin_install_python(PROGRAMS src/simulatorController.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
