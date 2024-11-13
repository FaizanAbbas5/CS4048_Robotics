#!/usr/bin/env python3

"""
Example showing how to build a world and use it with pyrobosim,
additionally starting up a ROS interface.
"""
import os
import rclpy
import threading
import numpy as np

from pyrobosim.core import Robot, World, WorldYamlLoader
from pyrobosim.gui import start_gui
from pyrobosim.navigation import ConstantVelocityExecutor, RRTPlanner
from pyrobosim.utils.general import get_data_folder
from pyrobosim.utils.pose import Pose
from pyrobosim_ros.ros_interface import WorldROSWrapper


data_folder = get_data_folder()


def check_for_fossils(self, robot_pose, detection_radius=0.5):
        """Check for fossils near the robot"""
        if not hasattr(self, 'world'):
            return None
            
        fossil_objects = [obj for obj in self.world.objects if obj.category == 'fossil']
        
        # Debug print all fossil objects
        self.get_logger().debug(f"All fossils: {[(obj.category, obj.pose.x, obj.pose.y) for obj in fossil_objects]}")
        
        for fossil in fossil_objects:
            dx = fossil.pose.x - robot_pose.x
            dy = fossil.pose.y - robot_pose.y
            distance = np.sqrt(dx*dx + dy*dy)
            
            if distance < detection_radius:
                return fossil
        return None

    def remove_fossil_from_world(self, x, y, radius=0.3):
        """Remove a fossil object from the world at the given coordinates"""
        if not hasattr(self, 'world'):
            return False
            
        fossil_objects = [obj for obj in self.world.objects if obj.category == 'fossil']
        
        for fossil in fossil_objects:
            dx = fossil.pose.x - x
            dy = fossil.pose.y - y
            distance = np.sqrt(dx*dx + dy*dy)
            
            if distance < radius:
                try:
                    # Get the object ID
                    object_id = fossil.name if hasattr(fossil, 'name') else None
                    self.get_logger().info(f"Attempting to remove fossil object with ID: {object_id}")
                    
                    # Remove the object
                    if hasattr(self.world, 'remove_object'):
                        self.world.remove_object(object_id)
                        self.get_logger().info(f"Removed fossil {object_id} at ({x:.2f}, {y:.2f})")
                    else:
                        self.world.objects.remove(fossil)
                        self.get_logger().info(f"Removed fossil at ({x:.2f}, {y:.2f}) from objects list")
                    
                    # Update internal tracking
                    if fossil in self.discovered_fossils:
                        self.discovered_fossils.remove(fossil)
                    
                    # Force a world state update if possible
                    if hasattr(self, 'publish_state'):
                        self.publish_state()
                        
                    return True
                except Exception as e:
                    self.get_logger().error(f"Error removing fossil: {str(e)}")
                    return False
                
        return False

def create_fossil_world():
    world = World()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    world.set_metadata(
        locations=os.path.join(current_dir, "fossil_location_data.yaml"),
        objects=os.path.join(current_dir, "fossil_object_data.yaml"),
    )

    exploration_coords = [(-5, -5), (5, -5), (5, 5), (-5, 5)]
    world.add_room(name="exploration_zone", footprint=exploration_coords, color=[0.8, 0.8, 0.8])

    # Add base station as location
    base = world.add_location(
        name="base_station0",
        category="base_station",
        parent="exploration_zone",
        pose=Pose(x=0.0, y=0.0, yaw=0.0)
    )

    # Add fossils as objects
    world.add_object(
        name="fossil0",
        category="fossil",
        parent="exploration_zone",
        pose=Pose(x=3.0, y=3.0, yaw=0.0)
    )

    world.add_object(
        name="fossil1",
        category="fossil",
        parent="exploration_zone",
        pose=Pose(x=-2.0, y=-2.0, yaw=0.0)
    )

    planner_config = {
        "world": world,
        "bidirectional": True,
        "rrt_connect": True,
        "rrt_star": True,
        "collision_check_step_dist": 0.05,
        "max_connection_dist": 1.0,
        "rewire_radius": 2.0,
        "compress_path": True
    }
    
    explorer_planner = RRTPlanner(**planner_config)
    explorer = Robot(
        name="explorer",
        radius=0.2,
        path_executor=ConstantVelocityExecutor(linear_velocity=0.5),
        path_planner=explorer_planner,
    )
    world.add_robot(explorer, loc="exploration_zone")
    
    collector_planner = RRTPlanner(**planner_config)
    collector = Robot(
        name="collector",
        radius=0.2,
        path_executor=ConstantVelocityExecutor(linear_velocity=0.5),
        path_planner=collector_planner,
    )
    world.add_robot(collector, loc="exploration_zone")

    return world


def create_world_from_yaml(world_file):
    return WorldYamlLoader().from_yaml(os.path.join(data_folder, world_file))


def create_ros_node():
    """Initializes ROS node"""
    rclpy.init()
    node = WorldROSWrapper(state_pub_rate=0.1, dynamics_rate=0.01)
    node.declare_parameter("world_file", value="")

    # Set the world
    world_file = node.get_parameter("world_file").get_parameter_value().string_value
    if world_file == "":
        node.get_logger().info("Creating demo world programmatically.")
        world = create_world()
    else:
        node.get_logger().info(f"Using world file {world_file}.")
        world = create_world_from_yaml(world_file)

    node.set_world(world)

    return node


if __name__ == "__main__":
    node = create_ros_node()

    # Start ROS node in separate thread
    ros_thread = threading.Thread(target=lambda: node.start(wait_for_gui=True))
    ros_thread.start()

    # Start GUI in main thread
    start_gui(node.world)
