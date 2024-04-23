#!/usr/bin/python3

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
import cv2
import numpy as np
import tf2_ros
import tf2_geometry_msgs as tfg
from tf2_ros import TransformException

from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, Vector3, Pose
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

qos_profile = QoSProfile(
          durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
          reliability=QoSReliabilityPolicy.RELIABLE,
          history=QoSHistoryPolicy.KEEP_LAST,
          depth=1)

class RingDetector(Node):
    def __init__(self):
        super().__init__('transform_point')

        self.depth_image = None
        self.marker_array = MarkerArray()
        self.marker_id = 0
        # Basic ROS stuff
        timer_frequency = 2
        timer_period = 1/timer_frequency

        # An object we use for converting images between ROS format and OpenCV format
        self.bridge = CvBridge()

        # Marker array object used for visualizations
        self.marker_array = MarkerArray()
        self.marker_num = 1

        # Subscribe to the image and/or depth topic
        self.image_sub = self.create_subscription(Image, "/oakd/rgb/preview/image_raw", self.image_callback, 1)
        self.depth_sub = self.create_subscription(Image, "/oakd/rgb/preview/depth", self.depth_callback, 1)

        # Publiser for the visualization markers
        self.marker_pub = self.create_publisher(Marker, "/ring", QoSReliabilityPolicy.BEST_EFFORT)

        # Object we use for transforming between coordinate frames
        self.tf_buf = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buf, self)

        cv2.namedWindow("Binary Image", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Detected contours", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Detected rings", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)        

    def image_callback(self, data):
        self.get_logger().info(f"I got a new image! Will try to find rings...")

        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        if self.depth_image is not None:
            depth_image = self.depth_image

        blue = cv_image[:,:,0]
        green = cv_image[:,:,1]
        red = cv_image[:,:,2]

        # Tranform image to grayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        # gray = red

        # Apply Gaussian Blur
        #gray = cv2.GaussianBlur(gray,(3,3),0)

        # Do histogram equalization
        #gray = cv2.equalizeHist(gray)

        # Binarize the image, there are different ways to do it
        #ret, thresh = cv2.threshold(img, 50, 255, 0)
        #ret, thresh = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 30)
        cv2.imshow("Binary Image", thresh)
        cv2.waitKey(1)

        # Extract contours
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Example of how to draw the contours, only for visualization purposes
        cv2.drawContours(gray, contours, -1, (255, 0, 0), 3)
        cv2.imshow("Detected contours", gray)
        cv2.waitKey(1)

        # Fit elipses to all extracted contours
        elps = []
        for cnt in contours:
            #     print cnt
            #     print cnt.shape
            if cnt.shape[0] >= 20:
                ellipse = cv2.fitEllipse(cnt)
                elps.append(ellipse)


        # Find two elipses with same centers
        candidates_3D = []
        candidates_2D = []
        candidates = []
        for n in range(len(elps)):
            for m in range(n + 1, len(elps)):
                # e[0] is the center of the ellipse (x,y), e[1] are the lengths of major and minor axis (major, minor), e[2] is the rotation in degrees
                
                e1 = elps[n]
                e2 = elps[m]
                dist = np.sqrt(((e1[0][0] - e2[0][0]) ** 2 + (e1[0][1] - e2[0][1]) ** 2))
                angle_diff = np.abs(e1[2] - e2[2])

                # The centers of the two elipses should be within 5 pixels of each other (is there a better treshold?)
                if dist >= 5:
                    continue

                # The rotation of the elipses should be whitin 4 degrees of eachother
                if angle_diff>4:
                    continue

                e1_minor_axis = e1[1][0]
                e1_major_axis = e1[1][1]

                e2_minor_axis = e2[1][0]
                e2_major_axis = e2[1][1]

                if e1_major_axis>=e2_major_axis and e1_minor_axis>=e2_minor_axis: # the larger ellipse should have both axis larger
                    le = e1 # e1 is larger ellipse
                    se = e2 # e2 is smaller ellipse
                elif e2_major_axis>=e1_major_axis and e2_minor_axis>=e1_minor_axis:
                    le = e2 # e2 is larger ellipse
                    se = e1 # e1 is smaller ellipse
                else:
                    continue # if one ellipse does not contain the other, it is not a ring
                
                # # The widths of the ring along the major and minor axis should be roughly the same
                #border_major = (le[1][1]-se[1][1])/2
                #border_minor = (le[1][0]-se[1][0])/2
                #border_diff = np.abs(border_major - border_minor)

                #if border_diff>4:
                #    continue

                # Get the depth of the center of the ellipses
                depth_center = depth_image[int(se[0][1]), int(se[0][0])]
                if depth_center == 0:
                    candidates_3D.append((e1,e2))
                    candidates.append((e1,e2))
                #self.get_logger().info(f"{depth_center}")

                if int(se[0][1]) > 160:
                    candidates_2D.append((e1,e2))
                    candidates.append((e1,e2))

                #candidates.append((e1,e2))

        print("Processing is done! found", len(candidates), "candidates for rings")

        # Plot the rings on the image
        for c in candidates:

            # the centers of the ellipses
            e1 = c[0]
            e2 = c[1]

            if c in candidates_3D:
                # Get a few points along the perimeter of the smaller ellipse
                e2_points = cv2.ellipse2Poly((int(e2[0][0]), int(e2[0][1])), (int(e2[1][0] / 2), int(e2[1][1] / 2)),
                                            int(e2[2]), 0, 360, 10)
                sampled_points = e2_points[np.random.choice(e2_points.shape[0], min(10, e2_points.shape[0]), replace=False)]

                # Extract color information at sampled points
                for point in sampled_points:
                    x, y = point
                    b = blue[y, x]
                    g = green[y, x]
                    r = red[y, x]
                    color = self.get_color(r, g, b)
                    if color is not None:
                        #self.get_logger().info(f"color at {point}: {color}")
                        self.publish_ring_marker(e1, color)

            # drawing the ellipses on the image
            cv2.ellipse(cv_image, e1, (0, 255, 0), 2)
            cv2.ellipse(cv_image, e2, (0, 255, 0), 2)

            # Get a bounding box, around the first ellipse ('average' of both elipsis)
            size = (e1[1][0]+e1[1][1])/2
            center = (e1[0][1], e1[0][0])

            x1 = int(center[0] - size / 2)
            x2 = int(center[0] + size / 2)
            x_min = x1 if x1>0 else 0
            x_max = x2 if x2<cv_image.shape[0] else cv_image.shape[0]

            y1 = int(center[1] - size / 2)
            y2 = int(center[1] + size / 2)
            y_min = y1 if y1 > 0 else 0
            y_max = y2 if y2 < cv_image.shape[1] else cv_image.shape[1]

            
            if len(candidates) > 0:
                cv2.imshow("Detected rings", cv_image)
                cv2.waitKey(1)

    def get_color(self, r, g, b):
        if abs(r-g) < 10 and abs(r-b) < 10 and abs(g-b) < 10:
            return None
        if abs(r-204) <= 10 and abs(g-71) <= 10 and abs(b-65) <= 10:
            return [1.0, 0.0, 0.0, 1.0]
        elif r < 60 and g > 120 and b < 60:
            return [0.0, 1.0, 0.0, 1.0]
        elif abs(r-48) <= 10 and abs(g-76) <= 10 and abs(b-100) <= 10:
            return [0.0, 0.0, 1.0, 1.0]
        elif r > 100 and b < 20:
            return [1.0, 1.0, 0.0, 1.0]
        else:
            return None
        
    def publish_ring_marker(self, e1, color):
        point_in_robot_frame = PointStamped()
        point_in_robot_frame.header.frame_id = "/base_link"
        point_in_robot_frame.header.stamp = self.get_clock().now().to_msg()

        point_in_robot_frame.point.x = e1[0][0]
        point_in_robot_frame.point.y = e1[0][1]
        point_in_robot_frame.point.z = 1.0

        # Now we look up the transform between the base_link and the map frames
        # and then we apply it to our PointStamped
        time_now = rclpy.time.Time()
        timeout = Duration(seconds=0.1)
        try:
            # An example of how you can get a transform from /base_link frame to the /map frame
            # as it is at time_now, wait for timeout for it to become available
            trans = self.tf_buf.lookup_transform("map", "base_link", time_now, timeout)
            self.get_logger().info(f"Looks like the transform is available.")

            # Now we apply the transform to transform the point_in_robot_frame to the map frame
            # The header in the result will be copied from the Header of the transform
            point_in_map_frame = tfg.do_transform_point(point_in_robot_frame, trans)
            self.get_logger().info(f"We transformed a PointStamped! JUHEEEJ: {point_in_map_frame}")

            # # If the transformation exists, create a marker from the point, in order to visualize it in Rviz
            marker_in_map_frame = self.create_marker(point_in_map_frame, self.marker_id, color)

			
            # # publishamo samo v primeru ko je marker nov torej ni iste barve
            new_marker = True
            for marker in self.marker_array.markers:
                if (
                    marker.color.r == color[0] and
                    marker.color.g == color[1] and
                    marker.color.b == color[2] and
                    marker.color.a == color[3]
                ):
                    new_marker = False
                    self.get_logger().info("ISTI")
                    break

            if new_marker:
                self.marker_pub.publish(marker_in_map_frame)
                self.marker_array.markers.append(marker_in_map_frame)
                # log
                self.get_logger().info(f"Ring detected at: {point_in_map_frame.point}")
                self.marker_id += 1

        except TransformException as te:
            self.get_logger().info(f"Cound not get the transform: {te}")

    def create_marker(self, point_stamped, marker_id, color):
        """You can the description of the Marker message here: https://docs.ros2.org/galactic/api/visualization_msgs/msg/Marker.html"""
        marker = Marker()

        marker.header = point_stamped.header

        marker.type = marker.CUBE
        marker.action = marker.ADD
        marker.id = marker_id

        # Set the scale of the marker
        scale = 0.15
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale

        # Set the color
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        marker.color.a = color[3]

        # Set the pose of the marker
        marker.pose.position.x = point_stamped.point.x
        marker.pose.position.y = point_stamped.point.y
        marker.pose.position.z = point_stamped.point.z

        return marker

    def depth_callback(self, data):

        try:
            depth_image = self.bridge.imgmsg_to_cv2(data, "32FC1")
        except CvBridgeError as e:
            print(e)

        depth_image[depth_image==np.inf] = 0
        
        # store the image
        self.depth_image = depth_image        
        
        # Do the necessairy conversion so we can visuzalize it in OpenCV
        image_1 = depth_image / 65536.0 * 255
        image_1 = image_1/np.max(image_1)*255

        image_viz = np.array(image_1, dtype= np.uint8)

        cv2.imshow("Depth window", image_viz)
        cv2.waitKey(1)
        

def main():

    rclpy.init(args=None)
    rd_node = RingDetector()

    rclpy.spin(rd_node)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
