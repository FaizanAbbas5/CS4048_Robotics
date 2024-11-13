#!/usr/bin/env python3

"""
Example showing how to build a world for fossil exploration.
"""
import os
import rclpy
import threading
import numpy as np
import time
from rclpy.node import Node
from geometry_msgs.msg import Point, Pose2D
from std_msgs.msg import String

from pyrobosim.core import Robot, World
from pyrobosim.gui import start_gui
from pyrobosim.navigation import ConstantVelocityExecutor, RRTPlanner
from pyrobosim.utils.pose import Pose
from pyrobosim.utils.general import get_data_folder
from pyrobosim_ros.ros_interface import WorldROSWrapper

class FossilExplorationNode(WorldROSWrapper):
    def __init__(self):
        super().__init__(state_pub_rate=0.1, dynamics_rate=0.01)
        
        # Create timers for robot behaviors
        self.explorer_timer = self.create_timer(2.0, self.explorer_behavior)
        self.explorer_exploring = False
        
        # Create publishers for robot status
        self.explorer_status_pub = self.create_publisher(
            String, 
            'explorer_status', 
            10
        )
        
        self.get_logger().info('FossilExplorationNode initialized')

    def get_robot_by_name(self, name):
        """Helper function to get robot by name"""
        for robot in self.world.robots:
            if robot.name == name:
                return robot
        return None

    def explorer_behavior(self):
        """Basic explorer robot behavior"""
        if not hasattr(self, 'world'):
            self.get_logger().warn('World not yet initialized')
            return
            
        explorer = self.get_robot_by_name('explorer')
        if explorer is None:
            self.get_logger().warn('Explorer robot not found')
            return
            
        if not self.explorer_exploring:
            # Generate a random point in the exploration zone
            x = np.random.uniform(-4, 4)
            y = np.random.uniform(-4, 4)
            
            self.get_logger().info(f'Planning move to ({x:.2f}, {y:.2f})')
            
            # Try to move to that point
            try:
                path = explorer.path_planner.plan(explorer.get_pose(), Pose(x=x, y=y))
                if path is not None:
                    explorer.follow_path(path)  # Changed from execute_path to follow_path
                    status_msg = String()
                    status_msg.data = f'Moving to ({x:.2f}, {y:.2f})'
                    self.explorer_status_pub.publish(status_msg)
                    self.explorer_exploring = True
                    self.get_logger().info('Started moving to target')
                else:
                    self.get_logger().warn('Failed to plan path to target position')
            except Exception as e:
                self.get_logger().error(f'Error in explorer movement: {str(e)}')
        else:
            # Check if robot has reached its destination
            if not explorer.is_moving():
                self.explorer_exploring = False
                status_msg = String()
                status_msg.data = 'Reached destination, planning next move'
                self.explorer_status_pub.publish(status_msg)
                self.get_logger().info('Reached destination')


def create_fossil_world():
    """Create our fossil exploration world"""
    world = World()

    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set the metadata from YAML files
    world.set_metadata(
        locations=os.path.join(current_dir, "fossil_location_data.yaml"),
        objects=os.path.join(current_dir, "fossil_object_data.yaml"),
    )

    # Add a main exploration room
    exploration_coords = [(-5, -5), (5, -5), (5, 5), (-5, 5)]
    world.add_room(name="exploration_zone", footprint=exploration_coords, color=[0.8, 0.8, 0.8])

    # Add base station
    base = world.add_location(
        category="base_station",
        parent="exploration_zone",
        pose=Pose(x=0.0, y=0.0, yaw=0.0)
    )

    # Add some initial fossil sites
    world.add_location(
        category="fossil_site",
        parent="exploration_zone",
        pose=Pose(x=2.0, y=2.0, yaw=0.0)
    )

    world.add_location(
        category="fossil_site",
        parent="exploration_zone",
        pose=Pose(x=-2.0, y=-2.0, yaw=0.0)
    )

    # Add robots with path planning capabilities
    planner_config = {
        "world": world,
        "bidirectional": True,
        "rrt_connect": False,
        "rrt_star": True,
        "collision_check_step_dist": 0.025,
        "max_connection_dist": 0.5,
        "rewire_radius": 1.5,
        "compress_path": False,
    }
    
    # Explorer robot
    explorer_planner = RRTPlanner(**planner_config)
    explorer = Robot(
        name="explorer",
        radius=0.2,
        path_executor=ConstantVelocityExecutor(linear_velocity=0.5),  # Added velocity
        path_planner=explorer_planner,
    )
    world.add_robot(explorer, loc="exploration_zone")
    
    # Collector robot
    collector_planner = RRTPlanner(**planner_config)
    collector = Robot(
        name="collector",
        radius=0.2,
        path_executor=ConstantVelocityExecutor(linear_velocity=0.5),  # Added velocity
        path_planner=collector_planner,
    )
    world.add_robot(collector, loc="exploration_zone")

    return world


if __name__ == "__main__":
    # Initialize ROS
    rclpy.init()
    
    # Create and initialize node
    node = FossilExplorationNode()
    
    # Create and set the world
    world = create_fossil_world()
    node.set_world(world)
    
    # Log that we're starting
    node.get_logger().info('Starting FossilExplorationNode')

    # Start ROS node in separate thread
    ros_thread = threading.Thread(target=lambda: node.start(wait_for_gui=True))
    ros_thread.start()

    # Start GUI in main thread
    start_gui(node.world)