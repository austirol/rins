#!/usr/bin/python3

from matplotlib import pyplot as plt
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import tf2_ros
from math import sqrt, cos, sin, pi
from PIL import Image as ImagePil, ImageDraw

from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PointStamped, Vector3, Pose
from sensor_msgs_py import point_cloud2 as pc2
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA, Bool, String
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

        # Subscribe to the image and/or depth topic
        #self.image_sub = self.create_subscription(Image, "/oakd/rgb/preview/image_raw", self.image_callback, 1)
        #self.depth_sub = self.create_subscription(Image, "/oakd/rgb/preview/depth", self.depth_callback, 1)

        #self.image_sub = self.create_subscription(Image, "/top_camera/rgb/preview/image_raw", self.image_callback, 1)
        #self.depth_sub = self.create_subscription(Image, "/top_camera/rgb/preview/depth", self.depth_callback, 1)

        self.image_sub = self.create_subscription(Image, "/top_camera/rgb/preview/image_raw", self.circle_hough_transform, 1)
        self.pointcloud_sub = self.create_subscription(PointCloud2, "/top_camera/rgb/preview/depth/points", self.pointcloud_callback, qos_profile_sensor_data)

        self.marker_pub = self.create_publisher(Marker, "/parking_marker", QoSReliabilityPolicy.BEST_EFFORT)

        self.when_to_park_sub = self.create_subscription(Bool, "/when_to_park", self.when_to_park_callback, 1)
        self.start_parking = False

        # Publiser for the visualization markers
        # self.marker_pub = self.create_publisher(Marker, "/ring", QoSReliabilityPolicy.BEST_EFFORT)

        # Object we use for transforming between coordinate frames
        # self.tf_buf = tf2_ros.Buffer()
        # self.tf_listener = tf2_ros.TransformListener(self.tf_buf)

        #cv2.namedWindow("Binary Image", cv2.WINDOW_NORMAL)
        #cv2.namedWindow("Detected contours", cv2.WINDOW_NORMAL)
        #cv2.namedWindow("Detected rings", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Circle Hough Transform", cv2.WINDOW_NORMAL)

    def when_to_park_callback(self, data):
        self.start_parking = True

    def circle_hough_transform(self, data):

        if self.start_parking == False:
            return
        # Apply Hough Circle Transform
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
            return
        
        cv2.imshow("Circle Hough Transform", img)
        cv2.waitKey(1)

        # apend the parking position to the list
        self.parking_pos.append((x,y))

    def pointcloud_callback(self, data):
        # get point cloud attributes
        height = data.height
        width = data.width
        point_step = data.point_step
        row_step = data.row_step

        # iterate over parking
        for x,y in self.parking_pos:

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
            scale = 0.1
            marker.scale.x = scale
            marker.scale.y = scale
            marker.scale.z = scale

            # Set the color
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 1.0
            marker.color.a = 1.0

            # Set the pose of the marker
            marker.pose.position.x = float(d[0]) - 0.37
            marker.pose.position.y = float(d[1]) - 0.25
            marker.pose.position.z = float(d[2])

            self.marker_pub.publish(marker)


    # def image_callback(self, data):
    #     self.get_logger().info(f"I got a new image! Will try to find rings...")

    #     try:
    #         cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    #     except CvBridgeError as e:
    #         print(e)

    #     blue = cv_image[:,:,0]
    #     green = cv_image[:,:,1]
    #     red = cv_image[:,:,2]

    #     # Tranform image to gayscale
    #     gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
    #     # gray = red

    #     # Apply Gaussian Blur
    #     # gray = cv2.GaussianBlur(gray,(3,3),0)

    #     # Do histogram equlization
    #     # gray = cv2.equalizeHist(gray)

    #     # Binarize the image, there are different ways to do it
    #     #ret, thresh = cv2.threshold(img, 50, 255, 0)
    #     #ret, thresh = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)
    #     thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 0, 30)
    #     cv2.imshow("Binary Image", thresh)
    #     cv2.waitKey(1)

    #     # Extract contours
    #     contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    #     # Example how to draw the contours, only for visualization purposes
    #     cv2.drawContours(gray, contours, -1, (255, 0, 0), 3)
    #     cv2.imshow("Detected contours", gray)
    #     cv2.waitKey(1)

    #     # Fit elipses to all extracted contours
    #     elps = []
    #     for cnt in contours:
    #         #     print cnt
    #         #     print cnt.shape
    #         if cnt.shape[0] >= 20:
    #             ellipse = cv2.fitEllipse(cnt)
    #             elps.append(ellipse)


    #     # Find two elipses with same centers
    #     candidates = []
    #     for n in range(len(elps)):
    #         for m in range(n + 1, len(elps)):
    #             # e[0] is the center of the ellipse (x,y), e[1] are the lengths of major and minor axis (major, minor), e[2] is the rotation in degrees
                
    #             e1 = elps[n]
    #             e2 = elps[m]
    #             dist = np.sqrt(((e1[0][0] - e2[0][0]) ** 2 + (e1[0][1] - e2[0][1]) ** 2))
    #             angle_diff = np.abs(e1[2] - e2[2])

    #             # The centers of the two elipses should be within 5 pixels of each other (is there a better treshold?)
    #             if dist >= 5:
    #                 continue

    #             # The rotation of the elipses should be whitin 4 degrees of eachother
    #             if angle_diff>4:
    #                 continue

    #             e1_minor_axis = e1[1][0]
    #             e1_major_axis = e1[1][1]

    #             e2_minor_axis = e2[1][0]
    #             e2_major_axis = e2[1][1]

    #             if e1_major_axis>=e2_major_axis and e1_minor_axis>=e2_minor_axis: # the larger ellipse should have both axis larger
    #                 le = e1 # e1 is larger ellipse
    #                 se = e2 # e2 is smaller ellipse
    #             elif e2_major_axis>=e1_major_axis and e2_minor_axis>=e1_minor_axis:
    #                 le = e2 # e2 is larger ellipse
    #                 se = e1 # e1 is smaller ellipse
    #             else:
    #                 continue # if one ellipse does not contain the other, it is not a ring
                
    #             # # The widths of the ring along the major and minor axis should be roughly the same
    #             # border_major = (le[1][1]-se[1][1])/2
    #             # border_minor = (le[1][0]-se[1][0])/2
    #             # border_diff = np.abs(border_major - border_minor)

    #             # if border_diff>4:
    #             #     continue
                    
    #             candidates.append((e1,e2))

    #     print("Processing is done! found", len(candidates), "candidates for rings")

    #     # Plot the rings on the image
    #     for c in candidates:

    #         # the centers of the ellipses
    #         e1 = c[0]
    #         e2 = c[1]

    #         # drawing the ellipses on the image
    #         cv2.ellipse(cv_image, e1, (0, 255, 0), 2)
    #         cv2.ellipse(cv_image, e2, (0, 255, 0), 2)

    #         # Get a bounding box, around the first ellipse ('average' of both elipsis)
    #         size = (e1[1][0]+e1[1][1])/2
    #         center = (e1[0][1], e1[0][0])

    #         x1 = int(center[0] - size / 2)
    #         x2 = int(center[0] + size / 2)
    #         x_min = x1 if x1>0 else 0
    #         x_max = x2 if x2<cv_image.shape[0] else cv_image.shape[0]

    #         y1 = int(center[1] - size / 2)
    #         y2 = int(center[1] + size / 2)
    #         y_min = y1 if y1 > 0 else 0
    #         y_max = y2 if y2 < cv_image.shape[1] else cv_image.shape[1]

    #     if len(candidates)>0:
    #             cv2.imshow("Detected rings",cv_image)
    #             cv2.waitKey(1)

    # def depth_callback(self,data):

    #     try:
    #         depth_image = self.bridge.imgmsg_to_cv2(data, "32FC1")
    #     except CvBridgeError as e:
    #         print(e)

    #     depth_image[depth_image==np.inf] = 0
        
    #     # Do the necessairy conversion so we can visuzalize it in OpenCV
    #     image_1 = depth_image / 65536.0 * 255
    #     image_1 =image_1/np.max(image_1)*255

    #     image_viz = np.array(image_1, dtype= np.uint8)

    #     cv2.imshow("Depth window", image_viz)
    #     cv2.waitKey(1)

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


    # for x in range(width):
    #     for y in range(height):
    #         for r in range(rmin, rmax + 1):
    #             if acc[x, y, r] > threshold:
    #                 draw.ellipse((x - r, y - r, x + r, y + r), outline=(255, 0, 0))

    return input_image, biggest_x, biggest_y


def main():

    rclpy.init(args=None)
    rd_node = ParkingDetector()

    rclpy.spin(rd_node)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()