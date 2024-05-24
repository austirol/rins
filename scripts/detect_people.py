#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSReliabilityPolicy

from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

from visualization_msgs.msg import Marker

from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

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

		self.marker_pub = self.create_publisher(Marker, marker_topic, QoSReliabilityPolicy.BEST_EFFORT)

		self.model = YOLO("yolov8n.pt")

		self.faces = []

		self.get_logger().info(f"Node has been initialized! Will publish face markers to {marker_topic}.")
		
	def rgb_callback(self, data):
		self.faces = []

		try:
			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")

			self.get_logger().info(f"Running inference on image...")

			# run inference
			res = self.model.predict(cv_image, imgsz=(256, 320), show=False, verbose=False, classes=[0], device=self.device)

			detected_faces = []
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
						# Draw the detected rectangle on the image
						cv_image = cv2.drawContours(cv_image, [approx], -1, (0, 255, 0), 3)

			# Process detected faces
			for bbox in detected_faces:
				cx = int((bbox[0] + bbox[2]) / 2)
				cy = int((bbox[1] + bbox[3]) / 2)

				# Check if the face is within a detected rectangle
				is_painting = False
				for rect in rectangles:
					x, y, w, h = rect
					if x <= cx <= x + w and y <= cy <= y + h:
						is_painting = True
						break

				if not is_painting and 90 < cx < 180:
					# Draw rectangle and center of bounding box
					cv_image = cv2.rectangle(cv_image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), self.detection_color, 3)
					cv_image = cv2.circle(cv_image, (cx, cy), 5, self.detection_color, -1)
					self.faces.append((cx, cy))

			cv2.imshow("image", cv_image)
			key = cv2.waitKey(1)
			if key == 27:
				print("exiting")
				exit()

		except CvBridgeError as e:
			print(e)

	def pointcloud_callback(self, data):

		# get point cloud attributes
		height = data.height
		width = data.width
		point_step = data.point_step
		row_step = data.row_step		

		# iterate over face coordinates
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

def main():
	print('Face detection node starting.')

	rclpy.init(args=None)
	node = detect_faces()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()