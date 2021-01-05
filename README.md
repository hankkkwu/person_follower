## Person Follower

The goal of this project is to build a robot that can follow people. Here we use a mono camera feed to get a robot car following a person.

The [cv_command]() script in this repo contains everything needed to run the person tracker on the robot. Before running on the commander machine, you need to setup a ROS catkin workspace on both the robot and the commander machine. This script should be run from within that workspace. See the [repo](https://github.com/goromal/lab_turtlebot) for help on getting ROS and the robot set up.

The 'MobileNetSSD' files define a trained deep learning model that can be used for a variety of object detection implementations. Here we use person detection.


### Environment
The robot car environment:
- Jetson nano
- STM32F103RCT6 motor controller
- Ubuntu 18.04
- ROS Melodic

The commander machine environment:
- Ubuntu 16.04
- ROS Kinetic
- OpenCV 3.4.4