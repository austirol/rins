#!/usr/bin/python3

from matplotlib import pyplot as plt
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import tf2_ros
import time
from math import sqrt, cos, sin, pi
from PIL import Image as ImagePil, ImageDraw

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PointStamped, Vector3, Pose
from sensor_msgs_py import point_cloud2 as pc2
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Bool, String, Float32MultiArray
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, qos_profile_sensor_data, QoSReliabilityPolicy, QoSProfile

qos_profile = QoSProfile(
          durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
          reliability=QoSReliabilityPolicy.RELIABLE,
          history=QoSHistoryPolicy.KEEP_LAST,
          depth=1)

class ParkingDetector(Node):
    def __init__(self):
        super().__init__('transform_point')

        # Basic ROS stuff
        timer_frequency = 2
        timer_period = 1/timer_frequency

        # An object we use for converting images between ROS format and OpenCV format
        self.bridge = CvBridge()

        # Marker array object used for visualizations
        self.marker_array = MarkerArray()
        self.marker_num = 1

        self.parking_pos = []
        self.stopHough = False

        # subscribe to image
        self.image_sub = self.create_subscription(Image, "/top_camera/rgb/preview/image_raw", self.circle_hough_transform, 1)
        
        self.offset_pub = self.create_publisher(Float32MultiArray, 'circle_offset', 10)

        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        self.when_to_park_sub = self.create_subscription(Bool, "/when_to_park", self.when_to_park_callback, 1)
        self.do_another_hough_sub = self.create_subscription(Bool, "/do_another_hough", self.do_another_hough_callback, 1)
        self.do_another_hough = self.create_publisher(Bool, "/do_another_hough", 1)
        self.start_parking = False

        cv2.namedWindow("Circle Hough Transform", cv2.WINDOW_NORMAL)

        self.target_tolerance = 15
        self.move_tolerance = 100
        self.angular_speed = 0.2 
        self.first_circle = False
        self.linear_speed = 0.1
        self.rotated_towwards_center = False
        self.moved = False
        self.change_direction = False
        self.previous_offset = 0
        
    def do_another_hough_callback(self, data):
        self.stopHough = False


    def when_to_park_callback(self, data):
        self.start_parking = True

    def circle_hough_transform(self, data):

        if self.start_parking == False or self.stopHough == True:
            return

        try:
            img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        pil_img = ImagePil.fromarray(img)
        # use the hough_for_circles function
        img, x,y = hough_for_circles(pil_img, img.shape[1], img.shape[0], 105, 110, 100, 15, pil_img)
        print("DID A HOUGH TRANSFORM")

        if (x == 0 or y == 0):
            if not self.first_circle:
                twist = Twist()
                twist.angular.z = 0.5
                self.cmd_vel_pub.publish(twist)
                time.sleep(0.1)
                self.do_another_hough.publish(Bool(data=True))
                return
                
            if (len(self.parking_pos) > 0):
                offset_x = self.parking_pos[-1][0]
                offset_y = self.parking_pos[-1][1]
                print(f"Offset x: {offset_x}, Offset y: {offset_y}", self.previous_offset)
                if (offset_y <= self.previous_offset) and self.moved:
                    print("Changing direction")
                    self.change_direction = True

                self.move_robot(offset_y)
                self.stopHough = True
                time.sleep(0.1)
            
            return
        
        cv2.imshow("Circle Hough Transform", img)
        cv2.waitKey(1)

        height, width, _ = img.shape
        img_center_x = width // 2
        img_center_y = height // 2

        offset_x = x - img_center_x
        offset_y = y - img_center_y

        self.parking_pos.append([offset_x, offset_y])
        self.first_circle = True

        if abs(offset_x) > self.target_tolerance:
            self.rotated_towwards_center = False
            self.rotate_robot(offset_x)
            time.sleep(0.1)
        else:
            self.get_logger().info("Circle is centered.")
            self.rotated_towwards_center = True

        if self.rotated_towwards_center:
            if abs(offset_y) < self.move_tolerance:
                
                self.move_robot(offset_y)
                time.sleep(0.1)
            else:
                self.get_logger().info("PArked!")
                
        self.stopHough = True

    def move_robot(self, offset_y):
        twist = Twist()
        self.get_logger().info(f"I am not in the center. Moving...")
        # to zato k včasih je zmeden
        self.previous_offset = offset_y
        if not self.change_direction:
            if offset_y < 0:   
                twist.linear.x = -self.linear_speed
            else:
                twist.linear.x = self.linear_speed
        else:
            if offset_y < 0:   
                twist.linear.x = self.linear_speed
            else:
                twist.linear.x = -self.linear_speed

        self.moved = True
        self.cmd_vel_pub.publish(twist)
        self.do_another_hough.publish(Bool(data=True))

    def rotate_robot(self, offset_x):
        twist = Twist()
        self.get_logger().info(f"Circle is not centered. Rotating...")
        if offset_x > 0:   
            twist.angular.z = -self.angular_speed
        else:
            twist.angular.z = self.angular_speed

        self.cmd_vel_pub.publish(twist)
        self.do_another_hough.publish(Bool(data=True))

def edge_alprox(image, width, height):
    output_image = ImagePil.new("RGB", (width, height))
    draw = ImageDraw.Draw(output_image)
    input_pixels = image.load()
    intensity = np.zeros((width, height))
    for x in range(width):
        for y in range(height):
            intensity[x, y] = sum(input_pixels[x, y]) / 3

    for x in range(1, image.width - 1):
        for y in range(1, image.height - 1):
            # aproksimacija magnitude 
            magx = intensity[x + 1, y] - intensity[x - 1, y]
            magy = intensity[x, y + 1] - intensity[x, y - 1]

            # nariše rob
            color = int(sqrt(magx**2 + magy**2))
            draw.point((x, y), (color, color, color))
        
    return output_image

def hough_for_circles(image, width, height, rmin, rmax, steps, threshold, input_image):
    edges = edge_alprox(image, width, height)
    points = []
    for r in range(rmin, rmax + 1):
        for t in range(steps):
            points.append((r, int(r * cos(2 * pi * t / steps)), int(r * sin(2 * pi * t / steps))))

    acc = np.zeros((width, height, rmax + 1))
    for x in range(width):
        for y in range(height):
            if edges.getpixel((x, y)) == (255, 255, 255): 
                for r, dx, dy in points:
                    a = x - dx
                    b = y - dy
                    if a >= 0 and a < width and b >= 0 and b < height:
                        acc[a, b, r] += 1
    
    draw = ImageDraw.Draw(input_image)

    biggest_x = 0
    biggest_y = 0
    biggest_r = 0

    for x in range(width):
        for y in range(height):
            for r in range(rmin, rmax + 1):
                if acc[x, y, r] > threshold:
                    if acc[x, y, r] > acc[biggest_x, biggest_y, biggest_r]:
                        biggest_x = x
                        biggest_y = y
                        biggest_r = r

    print(f"Biggest circle at: {biggest_x}, {biggest_y}, {biggest_r}")

    draw.ellipse((biggest_x - biggest_r, biggest_y - biggest_r, biggest_x + biggest_r, biggest_y + biggest_r), outline=(255, 0, 0))
    draw.point((biggest_x, biggest_y), fill=(255, 0, 0))

    input_image = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)

    return input_image, biggest_x, biggest_y


def main():

    rclpy.init(args=None)
    rd_node = ParkingDetector()

    rclpy.spin(rd_node)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()