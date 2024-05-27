#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSReliabilityPolicy

from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Bool

from geometry_msgs.msg import Twist

from visualization_msgs.msg import Marker

from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import time

from ultralytics import YOLO

# from rclpy.parameter import Parameter
# from rcl_interfaces.msg import SetParametersResult

class detect_faces(Node):

	def __init__(self):
		super().__init__('detect_faces')

		self.declare_parameters(
			namespace='',
			parameters=[
				('device', ''),
		])

		marker_topic = "/people_marker"

		self.detection_color = (0,0,255)
		self.device = self.get_parameter('device').get_parameter_value().string_value

		self.bridge = CvBridge()
		self.scan = None

		self.rgb_image_sub = self.create_subscription(Image, "/oakd/rgb/preview/image_raw", self.rgb_callback, qos_profile_sensor_data)
		self.pointcloud_sub = self.create_subscription(PointCloud2, "/oakd/rgb/preview/depth/points", self.pointcloud_callback, qos_profile_sensor_data)

		self.when_to_detect_sub = self.create_subscription(Bool, "/when_to_detect_faces", self.when_to_detect_callback, 1)
		self.when_to_start_centering_sub = self.create_subscription(Bool, "/center_mona_lisa", self.when_to_center_callback, 1)
		self.when_to_detect_lisas_sub = self.create_subscription(Bool, "/detect_mona_lisas", self.detect_lisas, 1)

		self.marker_pub = self.create_publisher(Marker, marker_topic, QoSReliabilityPolicy.BEST_EFFORT)
		self.mona_lisa_pub = self.create_publisher(Marker, "/mona_lisa", QoSReliabilityPolicy.BEST_EFFORT)
		self.centered_lisa_pub = self.create_publisher(Bool, "/centered_lisa", 1)
		self.do_face_again_pub = self.create_publisher(Bool, "/when_to_detect_faces", 1)


		self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

		self.model = YOLO("yolov8n.pt")

		self.faces = []

		self.detectMonaLisas = False
		self.mona_lisas = []

		## TO DVOJE DEJ NA FALSE - ZA POTREBE ROBBOT KOMANDERJA KASNEJE
		self.readyToDetect = False
		self.center = False
		
		self.angle_tolerance = 5
		self.min_pixels_in_image = 20000
		self.is_at_the_angle = False
		self.close_enough = False

		self.get_logger().info(f"Node has been initialized! Will publish face markers to {marker_topic}.")

	def when_to_center_callback(self, data):
		self.center = True
		self.readyToDetect = True
		return

	def detect_lisas(self, data):
		self.detectMonaLisas = True
		self.readyToDetect = True
		return

	def when_to_detect_callback(self, data):
		self.readyToDetect = True

		self.get_logger().info(f"Face detection is set to {self.readyToDetect}.")
		
	def rgb_callback(self, data):
		self.faces = []
		self.mona_lisas = []
		if not self.readyToDetect:
			return

		try:
			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

			#self.get_logger().info(f"Running inference on image...")

			# run inference
			res = self.model.predict(cv_image, imgsz=(256, 320), show=False, verbose=False, classes=[0], device=self.device)

			detected_faces = []
			not_drawn_image = cv_image.copy()
			for x in res:
				bbox = x.boxes.xyxy
				if bbox.nelement() == 0: # skip if empty
					continue
				bbox = bbox[0]
				detected_faces.append(bbox)

			# Detect rectangles (borders) in the image
			gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
			blurred = cv2.GaussianBlur(gray, (5, 5), 0)
			edged = cv2.Canny(blurred, 50, 150)
			contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

			rectangles = [cv2.boundingRect(contour) for contour in contours if cv2.contourArea(contour) > 1000]

			rectangles = []
			for contour in contours:
				if cv2.contourArea(contour) > 1000:
					# Approximate the contour to a polygon
					epsilon = 0.08 * cv2.arcLength(contour, True)
					approx = cv2.approxPolyDP(contour, epsilon, True)

					# Check if the approximated contour has 4 vertices (is a quadrilateral)
					if len(approx) == 4:
						rectangles.append(cv2.boundingRect(contour))
						

			# Process detected faces
			for bbox in detected_faces:
				cx = int((bbox[0] + bbox[2]) / 2)
				cy = int((bbox[1] + bbox[3]) / 2)

				x_min, y_min, x_max, y_max = bbox
				bottom_border_y = int(y_max.item())
				bottom_border_x_min = int(x_min.item())
				bottom_border_x_max = int(x_max.item())

				# za centriranje lise
				offset_x = cx - cv_image.shape[1] // 2
				offset_y = cy - cv_image.shape[0] // 2
				
				is_painting = False

				cv_image_cutout = cv_image[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
				pixels_in_image = cv_image_cutout.size

				# če so vrednosti tega pod 90 pol je to verjetno slika
				if bottom_border_y >= cv_image.shape[0] or bottom_border_x_min >= cv_image.shape[1] or bottom_border_x_max >= cv_image.shape[1]:
					bottom_border_y = cv_image.shape[0] - 1
					bottom_border_x_min = cv_image.shape[1] - 1
					bottom_border_x_max = cv_image.shape[1] - 1
					
				if (all(not_drawn_image[bottom_border_y, bottom_border_x_min] < 90) and all(not_drawn_image[bottom_border_y, bottom_border_x_max] < 90)):
					is_painting = True

				# Check if the face is within a detected rectangle
				if self.center: ### samo ko self center ko se centrira ker je drugače preveč false positive
					for rect in rectangles:
						x, y, w, h = rect
						if x <= cx <= x + w and y <= cy <= y + h:
							is_painting = True
							break
				

				# if self.center:
				# 	self.move_robot_to_center_the_image(offset_x, offset_y)

				if not is_painting and 90 < cx < 180:
					# Draw rectangle and center of bounding box
					cv_image = cv2.rectangle(cv_image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), self.detection_color, 3)
					cv_image = cv2.circle(cv_image, (cx, cy), 5, self.detection_color, -1)
					self.faces.append((cx, cy))

				if self.detectMonaLisas and is_painting and 90 < cx < 180:
					cv_image = cv2.rectangle(cv_image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 3)
					cv_image = cv2.circle(cv_image, (cx, cy), 5, (0, 255, 0), -1)
					self.mona_lisas.append((cx, cy))
				
					if self.center:
						self.move_robot_to_center_the_image(offset_x, offset_y, pixels_in_image)

			cv2.imshow("image", cv_image)
			key = cv2.waitKey(1)
			if key == 27:
				print("exiting")
				exit()

		except CvBridgeError as e:
			print(e)

	def move_robot_to_center_the_image(self, offset_x, offset_y, pixels_in_image):
		print(f"Offset x: {offset_x}, Offset y: {offset_y}")
		if not self.is_at_the_angle:
			if offset_x > self.angle_tolerance:
				self.move_robot(0, -1)
			elif offset_x < -self.angle_tolerance:
				self.move_robot(0, 1)
			else:
				self.is_at_the_angle = True

		if self.is_at_the_angle and not self.close_enough:
			if pixels_in_image < self.min_pixels_in_image:
				self.move_robot(-1, 0)
			else:
				self.close_enough = True
				print("Close enough!")
				self.centered_lisa_pub.publish(Bool(data=True))
				self.center = False
				# tole na false zato k se bo vrjetno izvaju anomaly detection
				self.detectMonaLisas = False
				self.is_at_the_angle = False
				self.close_enough = False

		

	def move_robot(self, offset_y, offset_x):
		twist = Twist()
		twist.linear.x = 0.1
		twist.angular.z = 0.8 * offset_x
		self.get_logger().info(f"Moving the robot. Offset x: {offset_x}, Offset y: {offset_y}")
		self.cmd_vel_pub.publish(twist)
		self.do_face_again_pub.publish(Bool(data=True))
		

	def pointcloud_callback(self, data):
		# get point cloud attributes
		height = data.height
		width = data.width
		point_step = data.point_step
		row_step = data.row_step		

		# iterate over face coordinates
		if not self.detectMonaLisas:
			for x,y in self.faces:
			
				# get 3-channel representation of the poitn cloud in numpy format
				a = pc2.read_points_numpy(data, field_names= ("x", "y", "z"))
				a = a.reshape((height,width,3))

				# read center coordinates
				d = a[y,x,:]

				# create marker
				marker = Marker()

				marker.header.frame_id = "/base_link"
				marker.header.stamp = data.header.stamp

				marker.type = 2
				marker.id = 0

				# Set the scale of the marker
				scale = 0.25
				marker.scale.x = scale
				marker.scale.y = scale
				marker.scale.z = scale

				# Set the color
				marker.color.r = 1.0
				marker.color.g = 1.0
				marker.color.b = 1.0
				marker.color.a = 1.0

				# Set the pose of the marker
				marker.pose.position.x = float(d[0])
				marker.pose.position.y = float(d[1])
				marker.pose.position.z = float(d[2])

				

				self.marker_pub.publish(marker)
		else:
			for x,y in self.mona_lisas:
				# get 3-channel representation of the poitn cloud in numpy format
				a = pc2.read_points_numpy(data, field_names= ("x", "y", "z"))
				a = a.reshape((height,width,3))

				# read center coordinates
				d = a[y,x,:]

				# create marker
				marker = Marker()

				marker.header.frame_id = "/base_link"
				marker.header.stamp = data.header.stamp

				marker.type = 2
				marker.id = 1

				# Set the scale of the marker
				scale = 0.25
				marker.scale.x = scale
				marker.scale.y = scale
				marker.scale.z = scale

				# Set the color
				marker.color.r = 1.0
				marker.color.g = 0.0
				marker.color.b = 0.0
				marker.color.a = 1.0

				# Set the pose of the marker
				marker.pose.position.x = float(d[0])
				marker.pose.position.y = float(d[1])
				marker.pose.position.z = float(d[2])

				print("PUBLISHING", d[0], d[1], d[2])

				self.mona_lisa_pub.publish(marker)

def main():
	print('Face detection node starting.')

	rclpy.init(args=None)
	node = detect_faces()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()