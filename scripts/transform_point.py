#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration

from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker

import tf2_geometry_msgs as tfg
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from rclpy.qos import qos_profile_sensor_data, QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

qos_profile = QoSProfile(
          durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
          reliability=QoSReliabilityPolicy.RELIABLE,
          history=QoSHistoryPolicy.KEEP_LAST,
          depth=1)

class TranformPoints(Node):
    """Demonstrating some convertions and loading the map as an image"""
    def __init__(self):
        super().__init__('map_goals')

        # Basic ROS stuff
        timer_frequency = 1
        timer_period = 1/timer_frequency

        # Functionality variables
        self.marker_id = 0

        # For listening and loading the 
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # For publishing the markers
        self.marker_pub = self.create_publisher(Marker, "/marker_pos", QoSReliabilityPolicy.BEST_EFFORT)
        self.marker_pub2 = self.create_publisher(Marker, "/marker_pos_rings", QoSReliabilityPolicy.BEST_EFFORT)
        self.marker_pub3 = self.create_publisher(Marker, "/marker_pos_parking", QoSReliabilityPolicy.BEST_EFFORT)

        # for green ring to park under
        self.green_ring_pub = self.create_publisher(Marker, "/green_ring", QoSReliabilityPolicy.BEST_EFFORT)
        # For subscribing to the markers
        self.marker_sub = self.create_subscription(Marker, "/people_marker", self.timer_callback, 1)
        self.marker_sub_parking = self.create_subscription(Marker, "/parking_marker", self.timer_callback2, 1)
        self.marker_sub_ring = self.create_subscription(Marker, "/ring_marker", self.publish_ring_marker, 1)

        # Create a timer, to do the main work.
        # self.timer = self.create_timer(timer_period, self.timer_callback)
        self.face_pos = []
        self.ring_pos = []
        self.parking_pos = []

    def timer_callback(self, msg):
        # Create a PointStamped in the /base_link frame of the robot
        # The point is located 0.5m in from of the robot
        # "Stamped" means that the message type contains a Header
        point_in_robot_frame = PointStamped()
        point_in_robot_frame.header.frame_id = "/base_link"
        point_in_robot_frame.header.stamp = self.get_clock().now().to_msg()

        point_in_robot_frame.point.x = msg.pose.position.x
        point_in_robot_frame.point.y = msg.pose.position.y
        point_in_robot_frame.point.z = msg.pose.position.z 

        # Now we look up the transform between the base_link and the map frames
        # and then we apply it to our PointStamped
        time_now = rclpy.time.Time()
        timeout = Duration(seconds=0.1)
        try:
            # An example of how you can get a transform from /base_link frame to the /map frame
            # as it is at time_now, wait for timeout for it to become available
            trans = self.tf_buffer.lookup_transform("map", "base_link", time_now, timeout)
            self.get_logger().info(f"Looks like the transform is available.")

            # Now we apply the transform to transform the point_in_robot_frame to the map frame
            # The header in the result will be copied from the Header of the transform
            point_in_map_frame = tfg.do_transform_point(point_in_robot_frame, trans)
            self.get_logger().info(f"We transformed a PointStamped! JUHEEEJ: {point_in_map_frame}")

            # # If the transformation exists, create a marker from the point, in order to visualize it in Rviz
            marker_in_map_frame = self.create_marker(point_in_map_frame, self.marker_id)

			
            # # publishamo samo v primeru ko je marker nov torej ni v arrayu self.face_pos ali bližini 0.5
            if len(self.face_pos) == 0:
                self.marker_pub.publish(marker_in_map_frame)
                self.face_pos.append({"x":point_in_map_frame.point.x, "y":point_in_map_frame.point.y, "z":point_in_map_frame.point.z})
                # log
                self.get_logger().info(f"Face detected at: {point_in_map_frame.point}")
                self.marker_id += 1

            else:
                for i in self.face_pos:
                    if abs(i["x"]-point_in_map_frame.point.x) < 0.8 and abs(i["y"]-point_in_map_frame.point.y) < 0.8 and abs(i["z"]-point_in_map_frame.point.z) < 0.8:
                        # log
                        self.get_logger().info(f"ISTI")
                        break
                else:
                    self.marker_pub.publish(marker_in_map_frame)
                    self.face_pos.append({"x":point_in_map_frame.point.x, "y":point_in_map_frame.point.y, "z":point_in_map_frame.point.z})
                    # log
                    self.get_logger().info(f"Face detected at: {point_in_map_frame.point}")
                    self.marker_id += 1

            # # Publish the marker if marker_id is set
            #self.marker_pub.publish(marker_in_map_frame)
            #self.get_logger().info(f"The marker has been published to /marker_pos. You are able to visualize it in Rviz")

            # # Increase the marker_id, so we dont overwrite the same marker.
            

        except TransformException as te:
            self.get_logger().info(f"Cound not get the transform: {te}")

    def timer_callback2(self, msg):
        # Create a PointStamped in the /base_link frame of the robot
        # The point is located 0.5m in from of the robot
        # "Stamped" means that the message type contains a Header
        point_in_robot_frame = PointStamped()
        point_in_robot_frame.header.frame_id = "/base_link"
        point_in_robot_frame.header.stamp = self.get_clock().now().to_msg()

        point_in_robot_frame.point.x = msg.pose.position.x + 0.1
        point_in_robot_frame.point.y = msg.pose.position.y
        point_in_robot_frame.point.z = msg.pose.position.z 

        # Now we look up the transform between the base_link and the map frames
        # and then we apply it to our PointStamped
        time_now = rclpy.time.Time()
        timeout = Duration(seconds=0.1)
        try:
            # An example of how you can get a transform from /base_link frame to the /map frame
            # as it is at time_now, wait for timeout for it to become available
            trans = self.tf_buffer.lookup_transform("map", "base_link", time_now, timeout)

            # Now we apply the transform to transform the point_in_robot_frame to the map frame
            # The header in the result will be copied from the Header of the transform
            point_in_map_frame = tfg.do_transform_point(point_in_robot_frame, trans)
            
            # # If the transformation exists, create a marker from the point, in order to visualize it in Rviz
            marker_in_map_frame = self.create_marker_parking(point_in_map_frame, self.marker_id)

            if (len(self.parking_pos) == 0):
                self.marker_pub3.publish(marker_in_map_frame)
                self.parking_pos.append({"x":point_in_map_frame.point.x, "y":point_in_map_frame.point.y, "z":point_in_map_frame.point.z})
                # log
                self.get_logger().info(f"Parking spot: {point_in_map_frame.point}")
                self.marker_id += 1
            else:
                for i in self.parking_pos:
                    if abs(i["x"]-point_in_map_frame.point.x) < 0.8 and abs(i["y"]-point_in_map_frame.point.y) < 0.8 and abs(i["z"]-point_in_map_frame.point.z) < 0.8:
                        # log
                        self.get_logger().info(f"ISTI")
                        break
                else:
                    self.marker_pub3.publish(marker_in_map_frame)
                    self.parking_pos.append({"x":point_in_map_frame.point.x, "y":point_in_map_frame.point.y, "z":point_in_map_frame.point.z})
                    # log
                    self.get_logger().info(f"Parking spot: {point_in_map_frame.point}")
                    self.marker_id += 1

            # # Publish the marker if marker_id is set
            #self.marker_pub.publish(marker_in_map_frame)
            #self.get_logger().info(f"The marker has been published to /marker_pos. You are able to visualize it in Rviz")

            # # Increase the marker_id, so we dont overwrite the same marker.
            
        except TransformException as te:
            self.get_logger().info(f"Cound not get the transform: {te}")


    def publish_ring_marker(self, msg):
        point_in_robot_frame = PointStamped()
        point_in_robot_frame.header.frame_id = "/base_link"
        point_in_robot_frame.header.stamp = self.get_clock().now().to_msg()

        point_in_robot_frame.point.x = msg.pose.position.x
        point_in_robot_frame.point.y = msg.pose.position.y
        point_in_robot_frame.point.z = msg.pose.position.z

        # Now we look up the transform between the base_link and the map frames
        # and then we apply it to our PointStamped
        time_now = rclpy.time.Time()
        timeout = Duration(seconds=0.1)
        try:
            # An example of how you can get a transform from /base_link frame to the /map frame
            # as it is at time_now, wait for timeout for it to become available
            trans = self.tf_buffer.lookup_transform("map", "base_link", time_now, timeout)
            self.get_logger().info(f"Looks like the transform is available.")

            # Now we apply the transform to transform the point_in_robot_frame to the map frame
            # The header in the result will be copied from the Header of the transform
            point_in_map_frame = tfg.do_transform_point(point_in_robot_frame, trans)
            self.get_logger().info(f"We transformed a PointStamped! JUHEEEJ: {point_in_map_frame}")

            # # If the transformation exists, create a marker from the point, in order to visualize it in Rviz
            marker_in_map_frame = self.create_marker2(point_in_map_frame, self.marker_id, msg.color)

            if len(self.ring_pos) == 0:
                self.marker_pub2.publish(marker_in_map_frame)

                # if it's green ring publish it to the green_ring topic
                if msg.color.r == 0.0 and msg.color.g == 1.0 and msg.color.b == 0.0:
                    self.green_ring_pub.publish(marker_in_map_frame)
                
                self.ring_pos.append({"x":point_in_map_frame.point.x, "y":point_in_map_frame.point.y, "z":point_in_map_frame.point.z})
                self.get_logger().info(f"1{self.ring_pos}")
                self.get_logger().info(f"RING DETECTED AT: {point_in_map_frame.point}")
                self.marker_id += 1
            else:
                for i in self.ring_pos:
                    if abs(i["x"]-point_in_map_frame.point.x) < 0.75 and abs(i["y"]-point_in_map_frame.point.y) < 0.75 and abs(i["z"]-point_in_map_frame.point.z) < 0.75:
                        # log
                        self.get_logger().info(f"ISTI")
                        break
                self.marker_pub2.publish(marker_in_map_frame)

                # if it's green ring publish it to the green_ring topic
                if msg.color.r == 0.0 and msg.color.g == 1.0 and msg.color.b == 0.0:
                    self.green_ring_pub.publish(marker_in_map_frame)
                
                self.ring_pos.append({"x":point_in_map_frame.point.x, "y":point_in_map_frame.point.y, "z":point_in_map_frame.point.z})
                #self.get_logger().info(f"Ring detected at: {point_in_map_frame.point}")
                self.marker_id += 1

        except TransformException as te:
            self.get_logger().info(f"Cound not get the transform: {te}")

    def create_marker2(self, point_stamped, marker_id, color):
        """You can the description of the Marker message here: https://docs.ros2.org/galactic/api/visualization_msgs/msg/Marker.html"""
        marker = Marker()

        marker.header = point_stamped.header

        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.id = marker_id

        # Set the scale of the marker
        scale = 0.15
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale

        # Set the color
        marker.color.r = color.r
        marker.color.g = color.g
        marker.color.b = color.b
        marker.color.a = color.a

        # Set the pose of the marker
        marker.pose.position.x = point_stamped.point.x
        marker.pose.position.y = point_stamped.point.y
        marker.pose.position.z = point_stamped.point.z

        return marker

    def create_marker(self, point_stamped, marker_id):
        """You can the description of the Marker message here: https://docs.ros2.org/galactic/api/visualization_msgs/msg/Marker.html"""
        marker = Marker()

        marker.header = point_stamped.header

        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.id = marker_id

        # Set the scale of the marker
        scale = 0.15
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale

        # Set the color
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        # Set the pose of the marker
        marker.pose.position.x = point_stamped.point.x
        marker.pose.position.y = point_stamped.point.y 
        marker.pose.position.z = point_stamped.point.z

        return marker
    
    def create_marker_parking(self, point_stamped, marker_id):
        """You can the description of the Marker message here: https://docs.ros2.org/galactic/api/visualization_msgs/msg/Marker.html"""
        marker = Marker()

        marker.header = point_stamped.header

        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.id = marker_id

        # Set the scale of the marker
        scale = 0.15
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale

        # Set the color
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 1.0

        # Set the pose of the marker
        marker.pose.position.x = point_stamped.point.x
        marker.pose.position.y = point_stamped.point.y
        marker.pose.position.z = point_stamped.point.z

        return marker

def main():

    rclpy.init(args=None)
    node = TranformPoints()
    
    rclpy.spin(node)
    
    rclpy.shutdown()

if __name__ == '__main__':
    main()