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

class QrReader(Node):
    def __init__(self):
        super().__init__('transform_point')

        # Basic ROS stuff
        timer_frequency = 2
        timer_period = 1/timer_frequency

        # An object we use for converting images between ROS format and OpenCV format
        self.bridge = CvBridge()

        self.when_to_detect = self.create_subscription(Bool, "/when_to_detect_qr", self.when_to_detect_callback, 1)
        # subscribe to image
        self.image_sub = self.create_subscription(Image, "/top_camera/rgb/preview/image_raw", self.qr_code_handler, 1)

        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        self.ended_pub = self.create_publisher(Bool, '/qr_detection_ended', 10)

        self.image_url = ""
        self.detectQR = False
        self.QR_detector = cv2.QRCodeDetector()
        cv2.namedWindow("QR CODE", cv2.WINDOW_NORMAL)

    def when_to_detect_callback(self, msg):
        self.detectQR = True
        return

    def qr_code_handler(self, msg):
        if not self.detectQR:
            return
        
        print("ON IT")
        try :
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            print(e)


        data, bbox, _ = self.QR_detector.detectAndDecode(cv_image)
        if bbox is None or data == "":
            print("QR Code not detected")
            twist = Twist()
            twist.angular.z = 0.5
            self.cmd_vel_pub.publish(twist)
            self.detectQR = True
            time.sleep(0.1)
            return 
        else: 
            print("QR Code detected")
            cv2.imshow("QR CODE", cv_image)
            cv2.waitKey(1)

            if data.startswith("https"):
                self.image_url = data
                print("URL: ", self.image_url)

        
        msg = Bool()
        msg.data = True
        self.ended_pub.publish(msg)
        self.detectQR = False
        
        return

def main():

    rclpy.init(args=None)
    rd_node = QrReader()

    rclpy.spin(rd_node)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()