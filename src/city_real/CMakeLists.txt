cmake_minimum_required(VERSION 3.0.2)
project(city_real)

find_package(catkin REQUIRED COMPONENTS
	std_msgs
	rospy
	sensor_msgs
	cv_bridge
	darknet_ros_msgs
)

catkin_package(
	CATKIN_DEPENDS
	catkin
	std_msgs
	rospy
	sensor_msgs
	cv_bridge
	darknet_ros_msgs
)

catkin_install_python(PROGRAMS src/autonomous_vehicle.py src/camera_interface.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
