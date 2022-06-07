cmake_minimum_required(VERSION 3.0.2)
project(city_simulation)

find_package(catkin REQUIRED COMPONENTS
  catkin
	std_msgs
	rospy
	sensor_msgs
	gazebo_msgs
	cv_bridge
	darknet_ros_msgs
	gazebo_traffic_light
)

catkin_package(
	CATKIN_DEPENDS
	catkin
	std_msgs
	rospy
	sensor_msgs
	gazebo_msgs
	cv_bridge
	darknet_ros_msgs
	gazebo_traffic_light
)

catkin_install_python(PROGRAMS src/simulator_controller.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
