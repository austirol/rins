#! /usr/bin/env python3
# Mofidied from Samsung Research America
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import asyncio
from visualization_msgs.msg import Marker, MarkerArray
from enum import Enum
import time
import pyttsx3
import math
import numpy as np
import cv2
import speech_recognition as sr

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import Quaternion, PoseStamped, PoseWithCovarianceStamped, PointStamped, Twist
from nav2_msgs.action import Spin, NavigateToPose
from std_msgs.msg import String, Bool, Float32MultiArray
from sensor_msgs.msg import Image
from builtin_interfaces.msg import Duration
from lifecycle_msgs.srv import GetState

from turtle_tf2_py.turtle_tf2_broadcaster import quaternion_from_euler

from irobot_create_msgs.action import Dock, Undock
from irobot_create_msgs.msg import DockStatus

from cv_bridge import CvBridge, CvBridgeError

import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration as rclpyDuration
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data

class TaskResult(Enum):
    UNKNOWN = 0
    SUCCEEDED = 1
    CANCELED = 2
    FAILED = 3

amcl_pose_qos = QoSProfile(
          durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
          reliability=QoSReliabilityPolicy.RELIABLE,
          history=QoSHistoryPolicy.KEEP_LAST,
          depth=1)

class RobotCommander(Node):

    def __init__(self, node_name='robot_commander', namespace=''):
        super().__init__(node_name=node_name, namespace=namespace)
        
        self.pose_frame_id = 'map'
        self.engine = pyttsx3.init()
        
        # Flags and helper variables
        self.goal_handle = None
        self.result_future = None
        self.feedback = None
        self.status = None
        self.initial_pose_received = False
        self.is_docked = None
        self.initial_pose = None

        self.bridge = CvBridge()

        # ROS2 subscribers
        self.create_subscription(DockStatus,
                                 'dock_status',
                                 self._dockCallback,
                                 qos_profile_sensor_data)
        
        self.localization_pose_sub = self.create_subscription(PoseWithCovarianceStamped,
                                                              'amcl_pose',
                                                              self._amclPoseCallback,
                                                              amcl_pose_qos)
        
        # ROS2 publishers
        self.initial_pose_pub = self.create_publisher(PoseWithCovarianceStamped,
                                                      'initialpose',
                                                      10)
        
        # arm publisher to change top camera
        self.top_camera_pub = self.create_publisher(String, '/arm_command', 1)
        # publisher za ringe kdaj začne detectad
        self.when_to_detect_rings = self.create_publisher(Bool, '/when_to_detect_rings', 1)
        self.when_to_park_pub = self.create_publisher(Bool, '/when_to_park', 1)
        self.when_to_detect_cylinders = self.create_publisher(Bool, '/when_to_detected_cylinder', 1)
        self.when_to_detect_qr = self.create_publisher(Bool, '/when_to_detect_qr', 1)
        self.when_to_detect_faces = self.create_publisher(Bool, '/when_to_detect_faces', 1)
        self.center_mona_lisa_pub = self.create_publisher(Bool, '/center_mona_lisa', 1)
        self.when_to_detect_lisas_pub = self.create_publisher(Bool, '/detect_mona_lisas', 1)
        self.when_to_check_for_anomaly = self.create_publisher(Bool, '/check_for_anomalys', 1)
        
        # marker position listener
        self.marker_pos_sub = self.create_subscription(MarkerArray, "/marker_pos", self.face_handler, 1)
        self.ring_marker_sub = self.create_subscription(Marker, "/marker_pos_rings", self.ring_handler, 1)
        self.cylinder_sub = self.create_subscription(Marker, "/detected_cylinder", self.cylinder_handler, 1)
        self.mona_lisa_sub = self.create_subscription(MarkerArray, "/marker_lisa", self.mona_handler, 1)

        self.ended_qr_sub = self.create_subscription(Bool, '/qr_detection_ended', self.qr_code_hanlder, 1)
        self.done_parking_sub = self.create_subscription(Bool, '/done_parking', self.done_parking_callback, 1)
        self.centered_lisa_sub = self.create_subscription(Bool, '/centered_lisa', self.centered_lisa_callback, 1)

        self.anomaly_result_sub = self.create_subscription(Bool, '/anomaly_result', self.anomaly_hanlder, 1)


        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', 10)
        
        # ROS2 Action clients
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.spin_client = ActionClient(self, Spin, 'spin')
        self.undock_action_client = ActionClient(self, Undock, 'undock')
        self.dock_action_client = ActionClient(self, Dock, 'dock')

        # pictures
        self.face_pos = []
        self.face_flag = []
        self.face_normals = []
        self.detect_faces = True

        # rings
        self.rings_and_positons = {}
        self.the_right_color = ""
        self.done = False

        # cylinder
        self.cylinder_pos = []
        self.cylinder_flag = []
        self.continue_walking = False

        # parking
        self.done_parking = False

        # speech recognition
        self.recognizer = sr.Recognizer()
        self.detected_colors = []

        self.mona_lisas = []
        self.mona_normals = []
        self.mona_flag = []

        self.detectLisa = False
        self.centeredMonaLisa = True
        self.is_not_anomaly = True
        self.got_anomaly_result = False
        self.the_right_lisa = False

        self.get_logger().info(f"Robot commander has been initialized!")

    def qr_code_hanlder(self, msg):
        self.continue_walking = True
        return
    
    def done_parking_callback(self, msg):
        self.done_parking = True
        return
    
    def anomaly_hanlder(self, msg):
        self.is_not_anomaly = msg.data
        self.got_anomaly_result = True
        print("Anomaly detected: ", self.is_not_anomaly)
        return

    def face_handler(self, msg):
        msg_face = msg.markers[0]
        msg_normal = msg.markers[1]
        x = msg_face.pose.position.x
        y = msg_face.pose.position.y
        z = msg_face.pose.position.z
        # index = len(self.face_pos)

        poiint_normal = {"x":msg_normal.pose.position.x, "y":msg_normal.pose.position.y, "z":msg_normal.pose.position.z}

        point = {"x":x, "y":y, "z":z}

        self.face_pos.append(point)
        self.face_flag.append(False)
        self.face_normals.append(poiint_normal)

        return
    
    def mona_handler(self, msg):
        msg_mona = msg.markers[0]
        msg_normal = msg.markers[1]

        x = msg_mona.pose.position.x
        y = msg_mona.pose.position.y
        z = msg_mona.pose.position.z
        # index = len(self.face_pos)

        poiint_normal = {"x":msg_normal.pose.position.x, "y":msg_normal.pose.position.y, "z":msg_normal.pose.position.z}

        point = {"x":x, "y":y, "z":z}

        self.mona_lisas.append(point)
        self.mona_flag.append(False)
        self.mona_normals.append(poiint_normal)

        return
    
    def centered_lisa_callback(self, msg):
        self.centeredMonaLisa = msg.data
        print("Centered Mona Lisa: ", self.centeredMonaLisa)
        return
    
    def ring_handler(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z

        point = {"x":x, "y":y, "z":z}
        
        color_red = msg.color.r
        color_green = msg.color.g
        color_blue = msg.color.b
        color_name = ""

        if color_red == 1.0 and color_green == 0.0 and color_blue == 0.0:
            color_name = "red"
        elif color_red == 0.0 and color_green == 1.0 and color_blue == 0.0:
            color_name = "green"
        elif color_red == 0.0 and color_green == 0.0 and color_blue == 1.0:
            color_name = "blue"
        elif color_red == 0.0 and color_green == 0.0 and color_blue == 0.0:
            color_name = "black"

        self.rings_and_positons[color_name] = point
    
        return
    
    def cylinder_handler(self, msg):
        print("Cylinder detected!")
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z

        point = {"x":x, "y":y, "z":z}

        self.cylinder_pos.append(point)
        self.cylinder_flag.append(False)
        return

    def destroyNode(self):
        self.nav_to_pose_client.destroy()
        super().destroy_node()     

    def goToPose(self, pose, behavior_tree=''):
        """Send a `NavToPose` action request."""
        self.debug("Waiting for 'NavigateToPose' action server")
        while not self.nav_to_pose_client.wait_for_server(timeout_sec=1.0):
            self.info("'NavigateToPose' action server not available, waiting...")

        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose
        self.goal_now = pose
        goal_msg.behavior_tree = behavior_tree

        self.info('Navigating to goal: ' + str(pose.pose.position.x) + ' ' +
                  str(pose.pose.position.y) + '...')
        send_goal_future = self.nav_to_pose_client.send_goal_async(goal_msg,
                                                                   self._feedbackCallback)
        rclpy.spin_until_future_complete(self, send_goal_future)
        self.goal_handle = send_goal_future.result()

        if not self.goal_handle.accepted:
            self.error('Goal to ' + str(pose.pose.position.x) + ' ' +
                       str(pose.pose.position.y) + ' was rejected!')
            return False

        self.result_future = self.goal_handle.get_result_async()
        return True

    def spin(self, spin_dist=1.57, time_allowance=10):
        self.debug("Waiting for 'Spin' action server")
        while not self.spin_client.wait_for_server(timeout_sec=1.0):
            self.info("'Spin' action server not available, waiting...")
        goal_msg = Spin.Goal()
        goal_msg.target_yaw = spin_dist
        goal_msg.time_allowance = Duration(sec=time_allowance)

        self.info(f'Spinning to angle {goal_msg.target_yaw}....')
        send_goal_future = self.spin_client.send_goal_async(goal_msg, self._feedbackCallback)
        rclpy.spin_until_future_complete(self, send_goal_future)
        self.goal_handle = send_goal_future.result()

        if not self.goal_handle.accepted:
            self.error('Spin request was rejected!')
            return False

        self.result_future = self.goal_handle.get_result_async()
        return True
    
    def undock(self):
        """Perform Undock action."""
        self.info('Undocking...')
        self.undock_send_goal()

        while not self.isUndockComplete():
            time.sleep(0.1)

    def undock_send_goal(self):
        goal_msg = Undock.Goal()
        self.undock_action_client.wait_for_server()
        goal_future = self.undock_action_client.send_goal_async(goal_msg)

        rclpy.spin_until_future_complete(self, goal_future)

        self.undock_goal_handle = goal_future.result()

        if not self.undock_goal_handle.accepted:
            self.error('Undock goal rejected')
            return

        self.undock_result_future = self.undock_goal_handle.get_result_async()

    def isUndockComplete(self):
        """
        Get status of Undock action.

        :return: ``True`` if undocked, ``False`` otherwise.
        """
        if self.undock_result_future is None or not self.undock_result_future:
            return True

        rclpy.spin_until_future_complete(self, self.undock_result_future, timeout_sec=0.1)

        if self.undock_result_future.result():
            self.undock_status = self.undock_result_future.result().status
            if self.undock_status != GoalStatus.STATUS_SUCCEEDED:
                self.info(f'Goal with failed with status code: {self.status}')
                return True
        else:
            return False

        self.info('Undock succeeded')
        return True
    
    def dock(self):
        """Perform Dock action."""
        self.info("Going to initial pose... Namreč zaznal sem tri obraze")
        goal_initial_pose = PoseStamped()
        goal_initial_pose.header.frame_id = 'map'
        goal_initial_pose.header.stamp = self.get_clock().now().to_msg()
        goal_initial_pose.pose = self.initial_pose.pose

        self.cancelTask()
        self.goToPose(goal_initial_pose)
        while not self.isTaskComplete():
            time.sleep(1)
            self.get_logger().info("Waiting for the task to complete... LOL4")
        

        self.info('Docking...')
        self.dock_send_goal()
            
        while not self.isDockComplete():
            time.sleep(1)

    def dock_send_goal(self):
        goal_msg = Dock.Goal()
        self.dock_action_client.wait_for_server()
        goal_future = self.dock_action_client.send_goal_async(goal_msg)

        rclpy.spin_until_future_complete(self, goal_future)

        self.dock_goal_handle = goal_future.result()

        if not self.dock_goal_handle.accepted:
            self.error('Dock goal rejected')
            return

        self.dock_result_future = self.dock_goal_handle.get_result_async()

    def isDockComplete(self):
        """
        Get status of Dock action.

        :return: ``True`` if docked, ``False`` otherwise.
        """
        if self.dock_result_future is None or not self.dock_result_future:
            return True

        rclpy.spin_until_future_complete(self, self.dock_result_future, timeout_sec=0.1)

        if self.dock_result_future.result():
            self.dock_status = self.dock_result_future.result().status
            if self.dock_status != GoalStatus.STATUS_SUCCEEDED:
                self.info(f'Goal with failed with status code: {self.status}')
                return True
            
        else:   
            return False
        

        self.info('Dock succeeded')
        self.is_docked = True
        return True

    def cancelTask(self):
        """Cancel pending task request of any type."""
        self.info('Canceling current task.')
        if self.result_future:
            future = self.goal_handle.cancel_goal_async()
            rclpy.spin_until_future_complete(self, future)
        return

    def isTaskComplete(self):
        """Check if the task request of any type is complete yet."""
        if not self.result_future:
            # task was cancelled or completed
            return True
        rclpy.spin_until_future_complete(self, self.result_future, timeout_sec=0.10)
        if self.result_future.result():
            self.status = self.result_future.result().status
            if self.status != GoalStatus.STATUS_SUCCEEDED:
                self.debug(f'Task with failed with status code: {self.status}')
                return True
        else:
            # Timed out, still processing, not complete yet
            return False

        self.debug('Task succeeded!')
        return True

    def getFeedback(self):
        """Get the pending action feedback message."""
        return self.feedback

    def getResult(self):
        """Get the pending action result message."""
        if self.status == GoalStatus.STATUS_SUCCEEDED:
            return TaskResult.SUCCEEDED
        elif self.status == GoalStatus.STATUS_ABORTED:
            return TaskResult.FAILED
        elif self.status == GoalStatus.STATUS_CANCELED:
            return TaskResult.CANCELED
        else:
            return TaskResult.UNKNOWN

    def waitUntilNav2Active(self, navigator='bt_navigator', localizer='amcl'):
        """Block until the full navigation system is up and running."""
        self._waitForNodeToActivate(localizer)
        if not self.initial_pose_received:
            time.sleep(1)
        self._waitForNodeToActivate(navigator)
        self.info('Nav2 is ready for use!')
        return

    def _waitForNodeToActivate(self, node_name):
        # Waits for the node within the tester namespace to become active
        self.debug(f'Waiting for {node_name} to become active..')
        node_service = f'{node_name}/get_state'
        state_client = self.create_client(GetState, node_service)
        while not state_client.wait_for_service(timeout_sec=1.0):
            self.info(f'{node_service} service not available, waiting...')

        req = GetState.Request()
        state = 'unknown'
        while state != 'active':
            self.debug(f'Getting {node_name} state...')
            future = state_client.call_async(req)
            rclpy.spin_until_future_complete(self, future)
            if future.result() is not None:
                state = future.result().current_state.label
                self.debug(f'Result of get_state: {state}')
            time.sleep(2)
        return
    
    def YawToQuaternion(self, angle_z = 0.):
        quat_tf = quaternion_from_euler(0, 0, angle_z)

        # Convert a list to geometry_msgs.msg.Quaternion
        quat_msg = Quaternion(x=quat_tf[0], y=quat_tf[1], z=quat_tf[2], w=quat_tf[3])
        return quat_msg

    def _amclPoseCallback(self, msg):
        self.debug('Received amcl pose')
        self.initial_pose_received = True
        self.current_pose = msg.pose
        return

    def _feedbackCallback(self, msg):
        self.debug('Received action feedback message')
        self.feedback = msg.feedback
        return
    
    def _dockCallback(self, msg: DockStatus):
        self.is_docked = msg.is_docked

    def setInitialPose(self, pose):
        msg = PoseWithCovarianceStamped()
        msg.pose.pose = pose
        msg.header.frame_id = self.pose_frame_id
        msg.header.stamp = 0
        self.info('Publishing Initial Pose')
        self.initial_pose_pub.publish(msg)
        return

    def info(self, msg):
        self.get_logger().info(msg)
        return

    def warn(self, msg):
        self.get_logger().warn(msg)
        return

    def error(self, msg):
        self.get_logger().error(msg)
        return

    def debug(self, msg):
        self.get_logger().debug(msg)
        return
    

def check_if_three_faces(rc):
    # check if three flags are true
    stevec = 0
    for i in range(len(rc.face_flag)):
        if rc.face_flag[i]:
            stevec += 1

        if stevec == 3:
            rc.get_logger().info("Tri obrazi so bili najdeni!")
            rc.dock()
            while rc.is_docked is None:
                rclpy.spin_once(rc, timeout_sec=0.5)
            if rc.is_docked:
                return True
    
def angle(vector1, vector2):
    dot_product = sum(a*b for a, b in zip(vector1, vector2))
    magnitude1 = math.sqrt(sum(a**2 for a in vector1))
    magnitude2 = math.sqrt(sum(b**2 for b in vector2))
    cos_theta = dot_product/(magnitude1*magnitude2)
    theta = math.acos(cos_theta)
    return theta

def quaternion_to_yaw(quaternion):
    x, y, z, w = quaternion
    sin_yaw = 2.0 * (w * z + x * y)
    cos_yaw = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(sin_yaw, cos_yaw)
    return yaw

def generate_goal_message(self, x, y, theta=0.2):
        goal_pose = PoseStamped()
        goal_pose.header.frame_id = 'map'
        goal_pose.header.stamp = self.get_clock().now().to_msg()

        goal_pose.pose.position.x = x
        goal_pose.pose.position.y = y
        goal_pose.pose.orientation = self.YawToQuaternion(theta)

        return goal_pose

def recognize_colors_from_speech(rc):
    # Initialize recognizer
    colors = ["red", "blue", "green", "yellow", "black"]

    while True:
        with sr.Microphone() as source:
            # Adjust for ambient noise
            rc.recognizer.adjust_for_ambient_noise(source)
            rc.get_logger().info("Say two colors")
            
            audio = rc.recognizer.listen(source)
        
            try:
                # Recognize speech using Google Web Speech API
                speech_text = rc.recognizer.recognize_google(audio).lower()
                rc.get_logger().info(f"You said: {speech_text}")
                
                # Check for the recognized colors in the speech text
                recognized_colors = [color for color in colors if color in speech_text]
                
                if len(recognized_colors) >= 2:
                    rc.get_logger().info(f"Recognized colors: {recognized_colors[:2]}")
                    for color in recognized_colors[:2]:
                        rc.detected_colors.append(color)
                    return recognized_colors[:2]
                elif len(recognized_colors) == 1:
                    rc.get_logger().info(f"Only one color recognized: {recognized_colors[0]}")
                    rc.get_logger().info("Please say another color.")
                else:
                    rc.get_logger().info("Okay so dont tell me.")
                    return None
        
            except sr.UnknownValueError:
                rc.get_logger().info("Google Speech Recognition could not understand audio. Please try again.")
            except sr.RequestError as e:
                rc.get_logger().info(f"Could not request results from Google Speech Recognition service; {e}")
                return None
            

def approach_face(rc):
    # da bo samo v tisti funckiji z liso approachal
    if rc.detectLisa:
        return
    for j, flag in enumerate(rc.face_flag):
        if not flag and not rc.is_docked and rc.detect_faces:
            #shrani trenutni goal
            goal_save = rc.goal_now
            
            #skenslaj goal
            rc.cancelTask()
            time.sleep(1)
            
            # pojdi do normale
            goal_x = float(rc.face_normals[j]["x"])
            goal_y = float(rc.face_normals[j]["y"])

            face_pos_x = float(rc.face_pos[j]["x"])
            face_pos_y = float(rc.face_pos[j]["y"])

            # turn to face
            razlika_y = face_pos_y - goal_y
            razlika_x = face_pos_x - goal_x
            kot = math.atan2(razlika_y, razlika_x)

            goal_pose = generate_goal_message(rc, goal_x, goal_y, kot)

            rc.goToPose(goal_pose)
            while not rc.isTaskComplete():
                rc.get_logger().info("Grem do obraza LOL2")
                time.sleep(1)


            #text to speach
            rc.get_logger().info("Text to speech!")
            rc.engine.say("Do you know where I should look for the Mona Lisa photo")
            rc.engine.runAndWait()
            # speech recignition
            two_colors = recognize_colors_from_speech(rc)
            rc.face_flag[j] = True

            print("Detected colors: ", rc.detected_colors)
            if len(rc.detected_colors) == 4:
                rc.get_logger().info("Found four colors")
                for color in rc.detected_colors:
                    if rc.detected_colors.count(color) == 2:
                        rc.the_right_color = color
                        rc.detect_faces = False
                        break

            # tuki se to odkomentira če hočmo da gre k zazna tri obraze nazaj v dock
            #if check_if_three_faces(rc):
            #    return  # return if three faces are detected

            if not rc.is_docked and rc.detect_faces:
                #restoraj goal
                rc.goToPose(goal_save)
                while not rc.isTaskComplete():
                    rc.get_logger().info("Waiting for the task to complete... LOL3")
                    # tuki approach face odkomentiraš da bi ti takoj šel do naslednjega obraza
                    approach_face(rc)
                   
                    # rc.cleaner()
                    time.sleep(1)
            else:
                rc.cancelTask()
                break


def approach_mona_lisa(rc):
    if not rc.detectLisa and rc.the_right_lisa:
        return
    
    for j, flag in enumerate(rc.mona_flag):
        if not flag and not rc.is_docked and rc.detectLisa and not rc.the_right_lisa:
            #shrani trenutni goal
            goal_save = rc.goal_now
            #shrani trenutno pozicijo
            pos_save = rc.current_pose
            
            #skenslaj goal
            rc.cancelTask()
            time.sleep(1)
            
            # pojdi do face
            goal_x = float(rc.mona_normals[j]["x"])
            goal_y = float(rc.mona_normals[j]["y"])

            face_pos_x = float(rc.mona_lisas[j]["x"])
            face_pos_y = float(rc.mona_lisas[j]["y"])

            # turn to face
            razlika_y = face_pos_y - goal_y
            razlika_x = face_pos_x - goal_x
            kot = math.atan2(razlika_y, razlika_x)            

            goal_pose = generate_goal_message(rc, goal_x, goal_y, kot)

            rc.goToPose(goal_pose)
            while not rc.isTaskComplete():
                rc.get_logger().info("Grem do MONA LISE LOL2")
                time.sleep(1)

           
            # center mona lisa
            rc.get_logger().info("Centering Mona Lisa")
            msg = Bool()
            msg.data = True
            rc.center_mona_lisa_pub.publish(msg)
            rc.centeredMonaLisa = False

            while not rc.centeredMonaLisa:
                rclpy.spin_once(rc, timeout_sec=0.5)

            rc.mona_flag[j] = True

            ### model od antona oz anomaly detection še pride
            rc.get_logger().info("Publishing to /check_for_anomaly")
            msg = Bool()
            msg.data = True
            rc.when_to_check_for_anomaly.publish(msg)

            while not rc.got_anomaly_result:
                rclpy.spin_once(rc, timeout_sec=0.5)

            if not rc.is_not_anomaly:
                rc.get_logger().info("Anomaly detected")
            else:
                rc.engine.say("Mona lisa found")
                rc.engine.runAndWait()
                msg = String()
                msg.data = "point"
                rc.top_camera_pub.publish(msg)

                rc.the_right_lisa = True

            rc.got_anomaly_result = False


            if not rc.the_right_lisa:
                #restoraj goal
                rc.goToPose(goal_save)
                while not rc.isTaskComplete():
                    rc.get_logger().info("Waiting for the task to complete... LOL3")
                    # tuki approach face odkomentiraš da bi ti takoj šel do naslednjega obraza
                    approach_mona_lisa(rc)
                   
                    # rc.cleaner()
                    time.sleep(1)


def approach_ring(rc, color):
    ### če ne vem barve še ne grem alpa če še ne vem kje je ring
    if (color == "" or rc.done or color not in rc.rings_and_positons):
        return
    
    goal_save = rc.goal_now

    rc.cancelTask()
    time.sleep(1)
    rc.get_logger().info(f'Approaching ring of color {color}')
        
    current_pos = rc.current_pose
    point = rc.rings_and_positons[color]
    goal_x = float(point["x"]) 
    goal_y = float(point["y"]) 

    current_pose_vector = np.array([current_pos.pose.position.x, current_pos.pose.position.y])
    goal_pose_vector = np.array([goal_x, goal_y])
    direction_vector = goal_pose_vector - current_pose_vector
    normalized_direction = direction_vector / np.linalg.norm(direction_vector)
    new_goal_pose = goal_pose_vector - 0.05 * normalized_direction
    goal_x_new = float(new_goal_pose[0])
    goal_y_new = float(new_goal_pose[1])

    goal_pose = generate_goal_message(rc, goal_x_new, goal_y_new)

    rc.goToPose(goal_pose)
    while not rc.isTaskComplete():
        rc.get_logger().info(f'GOING TOOOOOO RING {color}')
        current_pos = rc.current_pose
        time.sleep(3)

    # spremeni kamero
    rc.get_logger().info("Spreminjam kamero v parking mode")
    msg = String()
    msg.data = "look_for_parking"
    rc.top_camera_pub.publish(msg)

    time.sleep(9)

    # publish to /when_to_park
    msg = Bool()
    msg.data = True
    rc.when_to_park_pub.publish(msg)

    rc.done = True

    ## počakamo
    while not rc.done_parking:
        rclpy.spin_once(rc, timeout_sec=0.5)

    print("Publishing to /when_to_detected_cylinder")
    msg = Bool()
    msg.data = True
    rc.when_to_detect_cylinders.publish(msg)
    
    ## vrtimo dokler ne najdemo cilindra
    while len(rc.cylinder_flag) == 0:
        twis = Twist()
        twis.angular.z = 0.5
        rc.cmd_vel_pub.publish(twis)
        time.sleep(0.1)
        rclpy.spin_once(rc, timeout_sec=0.5)

    approach_cylindr(rc, goal_save)

def approach_cylindr(rc, goal_save):
    for j, flag in enumerate(rc.cylinder_flag):
        if not flag and not rc.is_docked:
            #shrani trenutno pozicijo
            pos_save = rc.current_pose
            
            #skenslaj goal
            rc.cancelTask()
            time.sleep(1)
            
            # gremo do cilindra
            current_x = float(pos_save.pose.position.x)
            current_y = float(pos_save.pose.position.y)
            goal_x = float(rc.cylinder_pos[j]["x"])
            goal_y = float(rc.cylinder_pos[j]["y"])

            current_pose_vector = np.array([current_x, current_y])
            goal_pose_vector = np.array([goal_x, goal_y])
            direction_vector = goal_pose_vector - current_pose_vector
            normalized_direction = direction_vector / np.linalg.norm(direction_vector)
            new_goal_pose = goal_pose_vector - 0.2 * normalized_direction
            goal_x_new = float(new_goal_pose[0])
            goal_y_new = float(new_goal_pose[1])

            razlika_y = goal_y - goal_y_new
            razlika_x = goal_x - goal_x_new
            kot = math.atan2(razlika_y, razlika_x)
            goal_pose = generate_goal_message(rc, goal_x_new, goal_y_new, kot)

            rc.goToPose(goal_pose)
            while not rc.isTaskComplete():
                rc.get_logger().info("Grem do cilindra")
                time.sleep(1)

            
            # sprememba kamere
            msg = String()
            msg.data = "look_for_qr"
            rc.top_camera_pub.publish(msg)

            # publish to /when_to_detect_qr
            msg = Bool()
            msg.data = True
            rc.when_to_detect_qr.publish(msg)
            
            while not rc.continue_walking:
                # TALE UKAZ TI NE USTAV CELOTNEGA PROGRAMA
                rclpy.spin_once(rc, timeout_sec=0.5)


            rc.cylinder_flag[j] = True
            ### od tuki naprej zaznava mona lise
            rc.detectLisa = True
            msg = Bool()
            msg.data = True
            rc.when_to_detect_lisas_pub.publish(msg)

            rc.goToPose(goal_save)
            while not rc.isTaskComplete():
                rc.get_logger().info("Waiting for the task to complete... LOL3")
                time.sleep(1)


def main(args=None):
    
    rclpy.init(args=args)
    rc = RobotCommander()

    # Wait until Nav2 and Localizer are available
    rc.waitUntilNav2Active()
    rc.initial_pose = rc.current_pose

    # kamera change
    rc.get_logger().info("Publishing to /arm_command")
    msg = String()
    msg.data = "up"
    rc.top_camera_pub.publish(msg)

    # za obraze
    msg = Bool()
    msg.data = True
    rc.when_to_detect_faces.publish(msg)

    # Check if the robot is docked, only continue when a message is recieved
    while rc.is_docked is None:
        rclpy.spin_once(rc, timeout_sec=0.5)

    # If it is docked, undock it first
    if rc.is_docked:
        rc.undock()
    
    # Finally send it a goal to reach
    goal_pose = PoseStamped()
    goal_pose.header.frame_id = 'map'
    goal_pose.header.stamp = rc.get_clock().now().to_msg()

    list_of_points = [[1.0, -2.0, 1.57],[2.5, -1.25, -1.8],[2.17, 0.45, -0.00],[1.0, 0.0, -1.57],[0.35, 3.25, -1.57],[-1.5, 4.5, 0.0],[-1.0, 1.2, 0.0],[1.1, 1.69, -1.57],[-1.55, -0.65, -1.57],[-0.27, -0.27, 0.0]]
    # publish to /when_to_detect_rings
    msg = Bool()
    msg.data = True
    rc.when_to_detect_rings.publish(msg)

    for i in range(len(list_of_points)):
        if not rc.is_docked and not rc.the_right_lisa:
            goal_pose.pose.position.x = list_of_points[i][0]
            goal_pose.pose.position.y = list_of_points[i][1]
            goal_pose.pose.orientation = rc.YawToQuaternion(list_of_points[i][2])
            rc.goToPose(goal_pose)
            while not rc.isTaskComplete():
                if not rc.is_docked:
                    # rc.get_logger().info("Waiting for the task to complete... LOL")
                    approach_face(rc)
                    approach_ring(rc, rc.the_right_color)
                    approach_mona_lisa(rc)
                    time.sleep(0.1)
                else:
                    rc.cancelTask()
                    break
        else:
            rc.cancelTask()
            break

    
    #rc.destroyNode()
    

    # And a simple example
if __name__=="__main__":
    main()