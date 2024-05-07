from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='dis_tutorial3',
            executable='transform_point.py',
            output='screen',
        ),
        Node(
            package='dis_tutorial3',
            executable='arm_mover_actions.py',
            output='screen',
        ),
        Node(
            package='dis_tutorial3',
            executable='cylinder_segmentation',
            output='screen',
        ),
        Node(
            package='dis_tutorial3',
            executable='parking_detect.py',
            output='screen',
        ),
        Node(
            package='dis_tutorial3',
            executable='robot_commander.py',
            output='screen',
        )
        ,
        Node(
            package='dis_tutorial3',
            executable='detect_rings.py',
            output='screen',
        )
    ])