#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSReliabilityPolicy

from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Bool

from geometry_msgs.msg import Twist

from visualization_msgs.msg import Marker, MarkerArray

from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import time
from sklearn.neighbors import KNeighborsClassifier

from ultralytics import YOLO

from inference import load_models, inference

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
		self.when_to_check_anomalys_sub = self.create_subscription(Bool, "/check_for_anomalys", self.when_to_anomalys_callback, 1)

		self.marker_pub = self.create_publisher(MarkerArray, marker_topic, QoSReliabilityPolicy.BEST_EFFORT)
		self.mona_lisa_pub = self.create_publisher(MarkerArray, "/mona_lisa", QoSReliabilityPolicy.BEST_EFFORT)
		self.centered_lisa_pub = self.create_publisher(Bool, "/centered_lisa", 1)
		self.do_face_again_pub = self.create_publisher(Bool, "/when_to_detect_faces", 1)
		self.anomaly_result_pub = self.create_publisher(Bool, "/anomaly_result", 1)


		self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

		self.model = YOLO("yolov8n.pt")

		self.faces = []
		self.candidates = {}
		self.rgb_values = [
			[39, 68, 66],     # Mona Lisa's average colors
			[36, 86, 61],
			[38, 72, 64],
			[38, 76, 64],
			[37, 85, 60],
			[37, 83, 60],
			[17.9, 23.8, 34],
			[17, 55, 27],
			[17, 60, 28],
			[17, 68, 29],
			[17, 48, 29],
			[18, 37, 31],
			[18, 39, 31],
			[20, 39, 35],     
			[20, 50, 60],
			[40, 70, 65],
			[40, 60, 70],
			[40, 60, 70],
			[35, 90, 60],
			[17, 47, 28],
			[17, 57, 28],
			[18, 40, 31],
			[19, 29, 32],
			[22, 56, 30],
			[22, 64, 28],
			[22, 70, 28],     
			[34, 90, 59],
			[33, 94, 57],
			[36, 80, 63],
			[36, 84, 63],
			[38, 73.5, 65],
			[15, 80, 26],
			[35, 87, 61],
			[102, 99, 140],  # Other detected faces
			[105, 115, 143],
			[108, 120, 147],
			[107, 120, 147],
			[75, 80, 80],   
			[73, 77, 80], 
			[71, 81, 83],
			[56, 58, 64],
			[65, 66, 71],
			[63, 64, 70],
			[67, 69, 74],
			[70, 71, 77],
			[50, 58, 60],
			[52, 60, 68],
			[54, 62, 70],
			[132, 125, 147],
			[123, 138, 153],
			[124, 137, 152],
			[125, 140, 176],
			[122, 135, 140],
			[155, 165, 171],
			[166, 175, 181],
			[177, 184, 189],
		]

		# Corresponding color labels
		self.color_labels = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
					   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

		# Train the classifier
		self.classifier = KNeighborsClassifier(n_neighbors=1)
		self.classifier.fit(self.rgb_values, self.color_labels)

		self.model_anom, self.model_anom_seg = load_models()

		# self.no_image = 3888

		# na false
		self.detectMonaLisas = False
		self.checkForAnomalys = False
		self.mona_lisas = []

		## TO DVOJE DEJ NA FALSE - ZA POTREBE ROBBOT KOMANDERJA KASNEJE
		self.readyToDetect = True
		self.center = True
		
		self.angle_tolerance = 5
		self.min_pixels_in_image = 25000
		self.min_x = 102
		self.min_y = 80

		self.get_logger().info(f"Node has been initialized! Will publish face markers to {marker_topic}.")

	def when_to_center_callback(self, data):
		self.center = True
		self.readyToDetect = True
		return
	
	def when_to_anomalys_callback(self, data):
		self.checkForAnomalys = True
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
		#if not self.readyToDetect:
		#	return

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

			# Process detected faces
			for bbox in detected_faces:
				cx = int((bbox[0] + bbox[2]) / 2)
				cy = int((bbox[1] + bbox[3]) / 2)

				x_min, y_min, x_max, y_max = bbox
				bottom_border_y = int(y_max.item())
				bottom_border_x_min = int(x_min.item())
				bottom_border_x_max = int(x_max.item())
				
				is_painting = False

				cv_image_cutout = cv_image[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
				pixels_in_image = cv_image_cutout.size
				x_shape , y_shape, _= cv_image_cutout.shape
				img_mean = cv_image_cutout.mean(axis=(0,1))
				#self.get_logger().info(f"{img_mean}\n")
				pred = self.classifier.predict([img_mean])
				if pred == 1:
					is_painting = True
					
				# če so vrednosti tega pod 90 pol je to verjetno slika
				if bottom_border_y >= cv_image.shape[0] or bottom_border_x_min >= cv_image.shape[1] or bottom_border_x_max >= cv_image.shape[1]:
					bottom_border_y = cv_image.shape[0] - 1
					bottom_border_x_min = cv_image.shape[1] - 1
					bottom_border_x_max = cv_image.shape[1] - 1
					
				if (all(not_drawn_image[bottom_border_y, bottom_border_x_min] < 90) and all(not_drawn_image[bottom_border_y, bottom_border_x_max] < 90)):
					is_painting = True

				# if self.center:
				# 	self.move_robot_to_center_the_image(offset_x, offset_y)

				if not is_painting:
					# Draw rectangle and center of bounding box
					cv_image = cv2.rectangle(cv_image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), self.detection_color, 3)
					cv_image = cv2.circle(cv_image, (cx, cy), 5, self.detection_color, -1)
					self.faces.append((cx, cy))

				if self.detectMonaLisas and is_painting:
					cv_image = cv2.rectangle(cv_image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 3)
					cv_image = cv2.circle(cv_image, (cx, cy), 5, (0, 255, 0), -1)
					self.mona_lisas.append((cx, cy))
				
					if self.center:
						self.move_robot_to_center_the_image(pixels_in_image, x_shape, y_shape)


				if self.checkForAnomalys and is_painting and pixels_in_image > 20000:
					un_im = not_drawn_image[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
					prediction, _, image_score = inference(un_im, self.model_anom, self.model_anom_seg)
					self.get_logger().info(f"Prediction: {prediction}, Image score: {image_score}")
					self.readyToDetect = False
					self.center = False
					self.checkForAnomalys = False
					msgAn = Bool()
					if prediction == 0:
						msgAn.data = True
					else:
						msgAn.data = False
					self.anomaly_result_pub.publish(msgAn)
					

			cv2.imshow("image", cv_image)
			key = cv2.waitKey(1)
			if key == 27:
				print("exiting")
				exit()
				

		except CvBridgeError as e:
			print(e)


	def move_robot_to_center_the_image(self, pixels_in_image, x_shape, y_shape):
		if pixels_in_image < self.min_pixels_in_image and x_shape < self.min_x and y_shape < self.min_y:
			twist = Twist()
			twist.linear.x = 0.2
			twist.angular.z = 0.0
			self.cmd_vel_pub.publish(twist)
			self.do_face_again_pub.publish((Bool(data=True)))
		else:
			self.centered_lisa_pub.publish((Bool(data=True)))
			print(pixels_in_image, x_shape, y_shape)
			self.center = False

	def calculate_normal(self, point_x, point_y, point_z):
		lineAB = point_y - point_x
		lineAC = point_z - point_x

		normal = np.cross(lineAB, lineAC)
		normal = normal / np.linalg.norm(normal)

		return normal
		

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

				radius = 10
				if x-radius < 0 or x+radius >= 320 or y-radius < 0 or y+radius >= 240:
					continue

				# # point 1 (left bottom corner)
				point1 = a[y+radius,x-radius,:]
				# # point 2 (right bottom corner)
				point2 = a[y+radius,x+radius,:]
				# # point 3 (center top)
				point3 = a[y-radius,x,:]

				if np.isinf(point1).any() or np.isinf(point2).any() or np.isinf(point3).any():
					continue

				# create marker

				marker_array = MarkerArray()
				marker = Marker()
				marker_normal = Marker()

				marker.header.frame_id = "/base_link"
				marker.header.stamp = data.header.stamp

				marker_normal.header.frame_id = "/base_link"
				marker_normal.header.stamp = data.header.stamp

				marker.type = 2
				marker.id = 0

				marker_normal.type = 0
				marker_normal.id = 1

				# Set the scale of the marker
				scale = 0.25
				marker.scale.x = scale
				marker.scale.y = scale
				marker.scale.z = scale

				# Set the scale of the marker
				marker_normal.scale.x = scale
				marker_normal.scale.y = scale
				marker_normal.scale.z = scale

				# Set the color
				marker.color.r = 1.0
				marker.color.g = 1.0
				marker.color.b = 1.0
				marker.color.a = 1.0

				# Set the color
				marker_normal.color.r = 1.0
				marker_normal.color.g = 0.0
				marker_normal.color.b = 1.0
				marker_normal.color.a = 1.0

				# use those three points to calculate the normal
				# if points are nan, skip
				if np.isnan(point1).any() or np.isnan(point2).any() or np.isnan(point3).any():
					continue

				normal = self.calculate_normal(point1, point2, point3)

				# points on the normal
				point_normal = d + 0.5 * normal
				
				# Set the pose of the marker
				marker.pose.position.x = float(d[0])
				marker.pose.position.y = float(d[1])
				marker.pose.position.z = float(d[2])

				# Set the pose of the marker
				marker_normal.pose.position.x = float(point_normal[0])
				marker_normal.pose.position.y = float(point_normal[1])
				marker_normal.pose.position.z = float(point_normal[2])

				marker_array.markers.append(marker)
				marker_array.markers.append(marker_normal)

				self.marker_pub.publish(marker_array)
		else:
			for x,y in self.mona_lisas:
				# get 3-channel representation of the poitn cloud in numpy format
				a = pc2.read_points_numpy(data, field_names= ("x", "y", "z"))
				a = a.reshape((height,width,3))

				# read center coordinates
				d = a[y,x,:]

				radius = 10
				if x-radius < 0 or x+radius >= 320 or y-radius < 0 or y+radius >= 240:
					continue

				# # point 1 (left bottom corner)
				point1 = a[y+radius,x-radius,:]
				# # point 2 (right bottom corner)
				point2 = a[y+radius,x+radius,:]
				# # point 3 (center top)
				point3 = a[y-radius,x,:]

				if np.isinf(point1).any() or np.isinf(point2).any() or np.isinf(point3).any():
					continue


				# create marker
				marker = Marker()
				marker_normal = Marker()
				
				marker.header.frame_id = "/base_link"
				marker.header.stamp = data.header.stamp

				marker_normal.header.frame_id = "/base_link"
				marker_normal.header.stamp = data.header.stamp

				marker.type = 2
				marker.id = 1

				marker_normal.type = 0
				marker_normal.id = 2

				# Set the scale of the marker
				scale = 0.25
				marker.scale.x = scale
				marker.scale.y = scale
				marker.scale.z = scale

				# Set the scale of the marker
				marker_normal.scale.x = scale
				marker_normal.scale.y = scale
				marker_normal.scale.z = scale

				# Set the color
				marker.color.r = 1.0
				marker.color.g = 0.0
				marker.color.b = 0.0
				marker.color.a = 1.0

				# Set the color
				marker_normal.color.r = 1.0
				marker_normal.color.g = 0.0
				marker_normal.color.b = 1.0
				marker_normal.color.a = 1.0

				# Set the pose of the marker
				marker.pose.position.x = float(d[0])
				marker.pose.position.y = float(d[1])
				marker.pose.position.z = float(d[2])

				# use those three points to calculate the normal
				# if points are nan, skip
				if np.isnan(point1).any() or np.isnan(point2).any() or np.isnan(point3).any():
					continue

				normal = self.calculate_normal(point1, point2, point3)

				# points on the normal
				point_normal = d + 0.6 * normal

				# Set the pose of the marker
				marker_normal.pose.position.x = float(point_normal[0])
				marker_normal.pose.position.y = float(point_normal[1])
				marker_normal.pose.position.z = float(point_normal[2])

				marker_array = MarkerArray()
				marker_array.markers.append(marker)
				marker_array.markers.append(marker_normal)

				self.mona_lisa_pub.publish(marker_array)

def main():
	print('Face detection node starting.')

	rclpy.init(args=None)
	node = detect_faces()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()