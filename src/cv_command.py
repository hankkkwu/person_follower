#!/usr/bin/env python

# This script contains everything needed to run the person tracker on the TurtleBot.
# Before running on the commander machine, you need to setup a ROS catkin workspace
# on both the TurtleBot and the commander machine.  This script should be run from
# within that workspace.  See the following repo for help on getting ROS and the 
# TurtleBot set up: https://github.com/goromal/lab_turtlebot
# The repo for the image processing part of the project is located here:
# https://github.com/jakelarsen17/TurtleBot-Follow-Person
# The 'MobileNetSSD' files in the repo are needed to run and define a trained machine learning model 
# that can be used for a variety of object detection implementations. Here we use person detection.

import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import Twist, Vector3

# print(cv2.__version__)


class CVControl:
    def __init__(self):
    	# load the serialized machine learning model for person detection
		self.net = cv2.dnn.readNetFromCaffe('/home/riki/catkin_ws/src/lab_turtlebot/turtlebot_cv/src/MobileNetSSD_deploy.prototxt.txt', 
									   		'/home/riki/catkin_ws/src/lab_turtlebot/turtlebot_cv/src/MobileNetSSD_deploy.caffemodel')

		# set counter for counting frames
		self.count_max = 400
		self.counter = 0

		# video writer setup
		fourcc = cv2.VideoWriter_fourcc('M','J','P','G') #Define the codec and create VideoWriter object
		self.out = cv2.VideoWriter('/home/riki/output_dnn.avi',fourcc, 10.0, (640,480))

		# ROS Setup
		# Image subscriber
		self.image_sub = rospy.Subscriber("/decompressed_img", Image, self.img_callback)

        # Turtlebot command publisher
		self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=3)


    # Main function that repeatedly gets called
    def img_callback(self, data):
		self.counter += 1

		# initialize the list of class labels MobileNet SSD was trained to
		# detect, then generate a set of bounding box colors for each class
		"""
		CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
			"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
			"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
			"sofa", "train", "tvmonitor"]
		COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
		"""
		
		try:
			bridge = CvBridge()
			cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
			print e
		
		frame = cv_image

		(h, w) = frame.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

		# pass the blob through the network and obtain the detections and predictions
		self.net.setInput(blob)
		detections = self.net.forward()

		big_area = 0
		big_center = 320
		detected = 0
		# loop over the detections
		for i in np.arange(0, detections.shape[2]):
			object_type = detections[0,0,i,1]
			confidence = detections[0, 0, i, 2]
			if object_type == 15 and confidence > 0.2: # execute if confidence is high for person detection
				
				# extract the index of the class label from the
				# `detections`, then compute the (x, y)-coordinates of
				# the bounding box for the object
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				# draw the prediction on the frame
				label = "{}: {:.2f}%".format('person',confidence * 100)
				cv2.rectangle(frame, (startX, startY), (endX, endY),[0,0,255], 2)
				y = startY - 15 if startY - 15 > 15 else startY + 15
				cv2.putText(frame, label, (startX, y),cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0,0,255], 2)

				# get bounding box center and area used for commanding the TurtleBot to follow a person
				rect_center = int((startX+endX)/2)
				rect_area = (endX-startX)*(endY-startY)
				detected = 1
				if rect_area > big_area: # big_area and big_center are used so that the TurtleBot always tracks the closest person
					big_area = rect_area
					big_center = rect_center

	    # if a person is detected, send commands to approach or move away from them depending on proximity
		if detected:
			#print(rect_center)
			#print(rect_area)
			if big_area > 10000: # Execute if the person is within a reasonable range
				target_center = 320
				target_area = 150000
				# proportional controller for the TurtleBot based on the persons position in the frame and how far they are away
				kr = .002
				angular_vel = -kr*(big_center - target_center)
				maxav = 0.5   # Set a maximum angular velocity for the robot
				angular_vel = np.max([-maxav, angular_vel])
				angular_vel = np.min([maxav, angular_vel])

				kt = 0.0000045
				linear_vel = -kt*(big_area - target_area)
				maxv = 0.5   # Set a maximum velocity for the robot
				linear_vel = np.max([-maxv, linear_vel])
				linear_vel = np.min([maxv, linear_vel])
				#print(linear_vel)
				# Send Velocity command to turtlebot
				self.send_command(linear_vel, angular_vel)	# Publish a motion command to the TurtleBot 
		
		# Write frames to a video file for up to count_max frames
		if self.counter < self.count_max:
			self.out.write(frame)
			print(self.counter)
		if self.counter == self.count_max:
			self.out.release()
			print('made video')
		cv2.imshow("Image window", frame)
		cv2.waitKey(1)

    def send_command(self, linear_vel, angular_vel):
        # Put v, w commands into Twist message
        velocity = Twist()
        # velocity.linear.x = linear_vel
        # velocity.angular.z = angular_vel

        velocity.linear = Vector3(linear_vel, 0., 0.)
        velocity.angular= Vector3(0., 0., angular_vel)

        # Publish Twist command
        self.cmd_pub.publish(velocity)

def main():
	ctrl = CVControl()
	rospy.init_node('image_converter')
	try:
		print("getting start")
		rospy.spin()
		
	except KeyboardInterrupt:
		print "Shutting down"
		cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
