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


from visualization_msgs.msg import Marker
from enum import Enum
import time

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Quaternion, PoseStamped, PoseWithCovarianceStamped, PointStamped
from lifecycle_msgs.srv import GetState
from nav2_msgs.action import Spin, NavigateToPose
from turtle_tf2_py.turtle_tf2_broadcaster import quaternion_from_euler

from irobot_create_msgs.action import Dock, Undock
from irobot_create_msgs.msg import DockStatus

import rclpy
from rclpy.action import ActionClient
from rclpy.duration import Duration as rclpyDuration
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data

from playsound import playsound


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
        
        # Flags and helper variables
        self.goal_handle = None
        self.result_future = None
        self.feedback = None
        self.status = None
        self.initial_pose_received = False
        self.is_docked = None

        self.goal_now = None
        self.undocked = False

        self.goal_save = None
        self.pos_save = None

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
        
        # marker position listenr
        self.marker_pos_sub = self.create_subscription(Marker, "/marker_pos", self.face_handler, 1)
        
        # ROS2 Action clients
        self.nav_to_pose_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.spin_client = ActionClient(self, Spin, 'spin')
        self.undock_action_client = ActionClient(self, Undock, 'undock')
        self.dock_action_client = ActionClient(self, Dock, 'dock')

        # pictures
        self.face_pos = []
        self.face_flag = []

        self.get_logger().info(f"Robot commander has been initialized!")

    # def cleaner(self):
    #     self.get_logger().info("CLEANING")
    #     epsilon = 0.50
    #     duplicates = []
    #     for i, pos1 in enumerate(self.face_pos):
    #         for j, pos2 in enumerate(self.face_pos[i+1:]):
    #             same = 0
    #             for coor in pos1.keys():
    #                 if abs(pos1[coor]-pos2[coor]) < epsilon:
    #                     same += 1
    #             if same == 3:
    #                 duplicates.append(j+i+1)

    #     for i, duplicate in enumerate(duplicates):
    #         self.face_pos.pop(duplicate-i)
        
    def _go_to_face(self):
        self.info("tukaj")
        if self.undocked:
            for i, flag in enumerate(self.face_flag):
                if not flag:
                    #shrani trenutni goal
                    self.goal_save = self.goal_now
                    #shrani trenutno pozicijo
                    self.pos_save = self.current_pose
                    # self.get_logger().info(self.pos_save)
                    # geometry_msgs.msg.PoseWithCovariance(pose=geometry_msgs.msg.Pose(position=geometry_msgs.msg.Point(x=-0.48143899956437886, y=-0.37357906631237975, z=0.0), orientation=geometry_msgs.msg.Quaternion(x=0.0, y=0.0, z=-0.644360343322105, w=0.764722006976273)), covariance=array([0.00867094, 0.00057082, 0.        , 0.        , 0.        ,
                    #    0.        , 0.00057082, 0.02656898, 0.        , 0.        ,
                    #    0.        , 0.        , 0.        , 0.        , 0.        ,
                    #    0.        , 0.        , 0.        , 0.        , 0.        ,
                    #    0.        , 0.        , 0.        , 0.        , 0.        ,
                    #    0.        , 0.        , 0.        , 0.        , 0.        ,
                    #    0.        , 0.        , 0.        , 0.        , 0.        ,
                    #    0.02857909]))
                    #skenslaj goal
                    self.cancelTask()
                    time.sleep(1)
                    # self.info("1")
                    #pojdi do face
                    goal_pose = PoseStamped()
                    goal_pose.header.frame_id = 'map'
                    goal_pose.header.stamp = self.get_clock().now().to_msg()
                    # self.info("1")
                    goal_pose.pose.position.x = self.face_pos[i]["x"]
                    goal_pose.pose.position.y = self.face_pos[i]["y"]
                    goal_pose.pose.orientation = self.pos_save.pose.orientation
                    # self.info("1")
                    self.goToPose(goal_pose)
                    self.info("1")
                    while not self.isTaskComplete():
                        self.info("Waiting for the task to complete... LOL")
                        # rc.cleaner()
                        time.sleep(1)
                    #text to speach
                    playsound('mojca.m4a')
                    self.info('playing sound using  playsound')
                    #pojdi nazaj
                    goal_pose.pose.position.x = self.pos_save.pose.position.x
                    goal_pose.pose.position.y = self.pos_save.pose.position.y
                    goal_pose.pose.orientation = self.pos_save.pose.orientation
                    self.goToPose(goal_pose)
                    while not self.isTaskComplete():
                        self.info("Waiting for the task to complete... LOL")
                        # rc.cleaner()
                        time.sleep(1)
                    #restoraj goal
                    self.goToPose(self.goal_save)
                    while not self.isTaskComplete():
                        self.info("Waiting for the task to complete... LOL")
                        # rc.cleaner()
                        time.sleep(1)
        return

        

    def face_handler(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        z = msg.pose.position.z
        # index = len(self.face_pos)

        point = {"x":x, "y":y, "z":z}

        self.face_pos.append(point)
        self.face_flag.append(False)
        self._go_to_face()


    #     epsilon = 0.50

    #     if len(self.face_pos) == 0:
    #         if len(self.face_pos) == index:
    #             self.face_pos.append({})
    #         self.face_pos[index] = {"x":x, "y":y, "z":z}
    #     else:
    #         for i in self.face_pos:
    #             self.get_logger().info(f"pridi {x, y, z} {self.face_pos}")
    #             same = 0
    #             for j in i.keys():
    #                 if abs(i[j]-point[j]) < epsilon:
    #                     self.get_logger().info(str(abs(i[j]-point[j])))
    #                     same += 1
    #             if same != 3:
    #                 if len(self.face_pos) == index:
    #                     self.face_pos.append({})
    #                 self.face_pos[index] = {"x":x, "y":y, "z":z}
    #                 time.sleep(0.5)
                    
    #             else:
    #                 self.get_logger().info(f"JUHEEEEEJ3: isti je")

    #     self.get_logger().info(f"JUHEEEJ2: {len(self.face_pos)}")
        

    #     self.cleaner()

        # time.sleep(0.1)
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
        self.undocked = True
        

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

    def cancelTask(self):
        """Cancel pending task request of any type."""
        self.info('Canceling current task.')
        if self.result_future:
            # self.info('Canceled current task.')
            future = self.goal_handle.cancel_goal_async()
            # self.info('Canceled current task.')
            time.sleep(1)
            #če je odkomentirano ne dela
            # rclpy.spin_until_future_complete(self, future)
        self.info('Canceled current task.')
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
        self.undocked = not self.is_docked

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
    
def main(args=None):
    
    rclpy.init(args=args)
    rc = RobotCommander()

    # Wait until Nav2 and Localizer are available
    rc.waitUntilNav2Active()

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
    
    list_of_points = [[1.0, -2.0, 1.7],[2.23, -1.6, -1.7],[1.0, 0.0, -1.5],[0.3, 3.25, -1.7],[-1.5, 4.5, 0.0],[-1.0, 1.2, -2.75],[1.0, 1.65, -1.7],[-1.55, -0.65, -1.7],[-0.27, -0.27, 0.0]]

    for i in range(len(list_of_points)):
        goal_pose.pose.position.x = list_of_points[i][0]
        goal_pose.pose.position.y = list_of_points[i][1]
        goal_pose.pose.orientation = rc.YawToQuaternion(list_of_points[i][2])
        rc.goToPose(goal_pose)
        while not rc.isTaskComplete():
            rc.info("Waiting for the task to complete... LOL")
            # rc.cleaner()
            time.sleep(1)
    
    rc.destroyNode()

    # And a simple example
if __name__=="__main__":
    main()