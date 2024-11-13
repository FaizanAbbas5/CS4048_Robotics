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
        
        self.explorer_timer = self.create_timer(2.0, self.explorer_behavior)
        self.explorer_exploring = False
        self.explorer_status_pub = self.create_publisher(String, 'explorer_status', 10)
        
        self.fossil_discovery_pub = self.create_publisher(String, 'fossil_discoveries', 10)
        self.discovered_fossils = set()

        self.test_done = False  # Track if we've done the test
        self.returning_to_base = False  # Track if collector is returning to base
        self.test_timer = self.create_timer(10.0, self.test_fossil_discovery)  # Will trigger test
        
        self.collector_timer = self.create_timer(2.0, self.collector_behavior)
        self.fossil_discovery_sub = self.create_subscription(
            String,
            'fossil_discoveries',
            self.fossil_discovery_callback,
            10
        )
        self.collection_queue = []
        self.collector_busy = False
        
        self.get_logger().info('FossilExplorationNode initialized')

    def get_robot_by_name(self, name):
        for robot in self.world.robots:
            if robot.name == name:
                return robot
        return None

    def explorer_behavior(self):
        if not hasattr(self, 'world'):
            self.get_logger().warn('World not yet initialized')
            return
                
        explorer = self.get_robot_by_name('explorer')
        if explorer is None:
            self.get_logger().warn('Explorer robot not found')
            return
        
        current_pose = explorer.get_pose()
        fossil_site = self.check_for_fossils(current_pose)
        
        if fossil_site is not None and fossil_site not in self.discovered_fossils:

            self.discovered_fossils.add(fossil_site)
            
            msg = String()
            msg.data = f"FOSSIL_FOUND:{fossil_site.pose.x},{fossil_site.pose.y}"
            self.fossil_discovery_pub.publish(msg)
            
            self.get_logger().info(f'Found fossil at ({fossil_site.pose.x:.2f}, {fossil_site.pose.y:.2f})')
                
        if not self.explorer_exploring:

            x = np.random.uniform(-4, 4)
            y = np.random.uniform(-4, 4)
            
            self.get_logger().info(f'Planning move to ({x:.2f}, {y:.2f})')

            try:
                path = explorer.path_planner.plan(explorer.get_pose(), Pose(x=x, y=y))
                if path is not None:
                    explorer.follow_path(path)
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
            if not explorer.is_moving():
                self.explorer_exploring = False
                status_msg = String()
                status_msg.data = 'Reached destination, planning next move'
                self.explorer_status_pub.publish(status_msg)
                self.get_logger().info('Reached destination')
            

    def check_for_fossils(self, robot_pose, detection_radius=0.5):
        if not hasattr(self, 'world'):
            return None
            
        fossil_sites = [loc for loc in self.world.locations if loc.category == 'fossil_site']
        
        for site in fossil_sites:
            dx = site.pose.x - robot_pose.x
            dy = site.pose.y - robot_pose.y
            distance = np.sqrt(dx*dx + dy*dy)
            
            if distance < detection_radius:
                return site
        return None

    def test_fossil_discovery(self):
        if not self.test_done:  # Only trigger once
            msg = String()
            msg.data = "FOSSIL_FOUND:3.0,3.0"  # Test coordinates
            self.fossil_discovery_pub.publish(msg)
            self.get_logger().info("Manually triggered fossil discovery at (3.0, 3.0)")
            self.test_done = True  # Prevent further triggers

    def fossil_discovery_callback(self, msg):
        if msg.data.startswith('FOSSIL_FOUND:'):
            x, y = map(float, msg.data.split(':')[1].split(','))
            if (x, y) not in [(loc[0], loc[1]) for loc in self.collection_queue]:  # Prevent duplicates
                self.collection_queue.append((x, y))
                self.get_logger().info(f'Added fossil at ({x}, {y}) to collection queue')

    def collector_behavior(self):
        if not hasattr(self, 'world'):
            return
                
        collector = self.get_robot_by_name('collector')
        if collector is None:
            return
                
        if not self.collector_busy and not self.returning_to_base and self.collection_queue:
            x, y = self.collection_queue[0]
            
            try:
                path = collector.path_planner.plan(collector.get_pose(), Pose(x=x, y=y))
                if path is not None:
                    collector.follow_path(path)
                    self.collector_busy = True
                    self.get_logger().info(f'Collector moving to fossil at ({x}, {y})')
            except Exception as e:
                self.get_logger().error(f'Error in collector movement: {str(e)}')
        
        elif self.collector_busy and not collector.is_moving():
            if self.collection_queue:
                self.collection_queue.pop(0)
            self.collector_busy = False
            self.returning_to_base = True
            self.get_logger().info('Fossil collected, returning to base')
            
            try:
                path = collector.path_planner.plan(collector.get_pose(), Pose(x=0.0, y=0.0))
                if path is not None:
                    collector.follow_path(path)
            except Exception as e:
                self.get_logger().error(f'Error returning to base: {str(e)}')
        
        elif self.returning_to_base and not collector.is_moving():
            self.returning_to_base = False
            self.get_logger().info('Reached base station')


def create_fossil_world():
    world = World()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    world.set_metadata(
        locations=os.path.join(current_dir, "fossil_location_data.yaml"),
        objects=os.path.join(current_dir, "fossil_object_data.yaml"),
    )

    exploration_coords = [(-5, -5), (5, -5), (5, 5), (-5, 5)]
    world.add_room(name="exploration_zone", footprint=exploration_coords, color=[0.8, 0.8, 0.8])

    base = world.add_location(
        category="base_station",
        parent="exploration_zone",
        pose=Pose(x=0.0, y=0.0, yaw=0.0)
    )

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
    
    explorer_planner = RRTPlanner(**planner_config)
    explorer = Robot(
        name="explorer",
        radius=0.2,
        path_executor=ConstantVelocityExecutor(linear_velocity=0.5),  # Added velocity
        path_planner=explorer_planner,
    )
    world.add_robot(explorer, loc="exploration_zone")
    
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
    rclpy.init()
    
    node = FossilExplorationNode()
    
    world = create_fossil_world()
    node.set_world(world)
    
    node.get_logger().info('Starting FossilExplorationNode')

    ros_thread = threading.Thread(target=lambda: node.start(wait_for_gui=True))
    ros_thread.start()

    start_gui(node.world)