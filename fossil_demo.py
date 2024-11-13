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

        self.test_done = False
        self.returning_to_base = False
        self.test_timer = self.create_timer(5.0, self.test_fossil_discovery)
        
        self.collector_timer = self.create_timer(2.0, self.collector_behavior)
        self.fossil_discovery_sub = self.create_subscription(
            String,
            'fossil_discoveries',
            self.fossil_discovery_callback,
            10
        )
        self.collection_queue = []
        self.collector_busy = False

        # Add tracking for collected fossils
        self.collected_fossils = []
        
        self.get_logger().info('FossilExplorationNode initialized')

    def safe_plan_path(self, robot, start_pose, goal_pose, retry_count=3):
        for attempt in range(retry_count):
            try:
                path = robot.path_planner.plan(start_pose, goal_pose)
                if path is not None:
                    return path
                # If path is None, try with slightly modified goal
                offset = 0.2 * (attempt + 1)
                modified_goal = Pose(
                    x=goal_pose.x + np.random.uniform(-offset, offset),
                    y=goal_pose.y + np.random.uniform(-offset, offset),
                    yaw=goal_pose.yaw
                )
                path = robot.path_planner.plan(start_pose, modified_goal)
                if path is not None:
                    return path
            except Exception as e:
                self.get_logger().warn(f"Planning attempt {attempt + 1} failed: {str(e)}")
                continue
        return None

    def collect_fossil(self, fossil, collector):
        if not hasattr(fossil, 'pose'):
            return False
            
        try:
            # Store the fossil's original pose
            fossil.original_pose = Pose(
                x=fossil.pose.x,
                y=fossil.pose.y,
                yaw=fossil.pose.yaw
            )
            
            # Update the fossil's pose to match the collector's position
            # Offset slightly to make it visible
            fossil.pose.x = collector.get_pose().x
            fossil.pose.y = collector.get_pose().y
            
            # Add to collected fossils list
            self.collected_fossils.append(fossil)
            
            self.get_logger().info(f"Collected fossil at ({fossil.original_pose.x:.2f}, {fossil.original_pose.y:.2f})")
            return True
            
        except Exception as e:
            self.get_logger().error(f"Error collecting fossil: {str(e)}")
            return False

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
        
        # Print current position periodically
        self.get_logger().debug(f"Explorer at ({current_pose.x:.2f}, {current_pose.y:.2f})")
        
        fossil_site = self.check_for_fossils(current_pose)
        
        if fossil_site is not None and fossil_site not in self.discovered_fossils:
            self.discovered_fossils.add(fossil_site)
            
            msg = String()
            msg.data = f"FOSSIL_FOUND:{fossil_site.pose.x},{fossil_site.pose.y}"
            self.fossil_discovery_pub.publish(msg)
            
            self.get_logger().info(f'Found fossil at ({fossil_site.pose.x:.2f}, {fossil_site.pose.y:.2f})')
                
        if not self.explorer_exploring:
            # Keep explorer away from the edges
            margin = 0.5  # Stay 0.5 units away from edges
            x = np.random.uniform(-4 + margin, 4 - margin)
            y = np.random.uniform(-4 + margin, 4 - margin)
            
            self.get_logger().info(f'Planning move to ({x:.2f}, {y:.2f})')
            
            goal_pose = Pose(x=x, y=y)
            path = self.safe_plan_path(explorer, explorer.get_pose(), goal_pose)
            
            if path is not None:
                explorer.follow_path(path)
                status_msg = String()
                status_msg.data = f'Moving to ({x:.2f}, {y:.2f})'
                self.explorer_status_pub.publish(status_msg)
                self.explorer_exploring = True
                self.get_logger().info('Started moving to target')
            else:
                self.get_logger().warn('Failed to plan path to target position after multiple attempts')
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
        
        # Debug print all fossil sites
        self.get_logger().debug(f"All fossil sites: {[(site.category, site.pose.x, site.pose.y) for site in fossil_sites]}")
        
        for site in fossil_sites:
            dx = site.pose.x - robot_pose.x
            dy = site.pose.y - robot_pose.y
            distance = np.sqrt(dx*dx + dy*dy)
            
            if distance < detection_radius:
                return site
        return None

    def get_valid_collection_pose(self, fossil_x, fossil_y, robot, approach_radius=0.8):
        current_pose = robot.get_pose()
        self.get_logger().info(f"Robot at ({current_pose.x:.2f}, {current_pose.y:.2f}), trying to reach fossil at ({fossil_x:.2f}, {fossil_y:.2f})")
        
        # Try different radii if needed
        for radius in [0.8, 1.0, 1.2]:
            angles = np.linspace(0, 2*np.pi, 16)  # Try 16 positions around the fossil
            
            for i, angle in enumerate(angles):
                # Calculate position at radius from fossil
                x = fossil_x + radius * np.cos(angle)
                y = fossil_y + radius * np.sin(angle)
                
                self.get_logger().debug(f"Trying approach position {i+1}/16 at ({x:.2f}, {y:.2f}) with radius {radius}")
                
                # Try to plan a path to this position
                goal_pose = Pose(x=x, y=y)
                path = self.safe_plan_path(robot, robot.get_pose(), goal_pose)
                
                if path is not None:
                    self.get_logger().info(f"Found valid collection pose at ({x:.2f}, {y:.2f}) with radius {radius}")
                    return goal_pose
                
        self.get_logger().warn(f"Could not find any valid approach positions for fossil at ({fossil_x}, {fossil_y})")
        return None

    def get_valid_base_pose(self, robot, base_radius=0.8):
        current_pose = robot.get_pose()
        self.get_logger().info(f"Robot at ({current_pose.x:.2f}, {current_pose.y:.2f}), trying to find path to base")
        
        angles = np.linspace(0, 2*np.pi, 16)  # Try 16 positions around the base
        
        for radius in [0.8, 1.0, 1.2]:  # Start with minimum safe distance
            for i, angle in enumerate(angles):
                # Calculate position at radius from base
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)
                
                self.get_logger().debug(f"Trying base approach position at ({x:.2f}, {y:.2f})")
                
                goal_pose = Pose(x=x, y=y)
                path = self.safe_plan_path(robot, robot.get_pose(), goal_pose)
                
                if path is not None:
                    self.get_logger().info(f"Found valid base return pose at ({x:.2f}, {y:.2f})")
                    return goal_pose
        
        self.get_logger().warn("Could not find any valid approach positions for base return")
        return None

    def remove_fossil_from_world(self, x, y, radius=0.3):
        if not hasattr(self, 'world'):
            return False
            
        fossil_sites = [loc for loc in self.world.locations if loc.category == 'fossil_site']
        
        for site in fossil_sites:
            dx = site.pose.x - x
            dy = site.pose.y - y
            distance = np.sqrt(dx*dx + dy*dy)
            
            if distance < radius:
                try:
                    # Get the location ID
                    location_id = site.name if hasattr(site, 'name') else None
                    self.get_logger().info(f"Attempting to remove fossil with ID: {location_id}")
                    
                    # Try to remove using the world's remove_location method
                    if hasattr(self.world, 'remove_location'):
                        self.world.remove_location(location_id)
                        self.get_logger().info(f"Removed fossil {location_id} at ({x:.2f}, {y:.2f}) using remove_location")
                    else:
                        # Fallback: try to remove from locations list
                        self.world.locations.remove(site)
                        self.get_logger().info(f"Removed fossil at ({x:.2f}, {y:.2f}) from locations list")
                    
                    # Update any internal tracking
                    if site in self.discovered_fossils:
                        self.discovered_fossils.remove(site)
                    
                    # Force a world state update if possible
                    if hasattr(self, 'publish_state'):
                        self.publish_state()
                        
                    return True
                except Exception as e:
                    self.get_logger().error(f"Error removing fossil: {str(e)}")
                    self.get_logger().error(f"Location properties: {dir(site)}")  # Debug info
                    return False
                
        return False

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

    def deposit_fossils_at_base(self, collector):
        for fossil in self.collected_fossils:
            try:
                # Place fossils in a circle around the base
                angle = len(self.collected_fossils) * (2 * np.pi / max(len(self.collected_fossils), 1))
                deposit_radius = 0.3  # Distance from base center
                
                fossil.pose.x = deposit_radius * np.cos(angle)
                fossil.pose.y = deposit_radius * np.sin(angle)
                
                self.get_logger().info(f"Deposited fossil at base position ({fossil.pose.x:.2f}, {fossil.pose.y:.2f})")
                
            except Exception as e:
                self.get_logger().error(f"Error depositing fossil: {str(e)}")
        
        # Clear the collected fossils list
        self.collected_fossils = []

    def collector_behavior(self):
        if not hasattr(self, 'world'):
            return
                
        collector = self.get_robot_by_name('collector')
        if collector is None:
            return
                
        if not self.collector_busy and not self.returning_to_base and self.collection_queue:
            # Moving to fossil
            x, y = self.collection_queue[0]
            
            # Get a valid pose near the fossil
            goal_pose = self.get_valid_collection_pose(x, y, collector)
            
            if goal_pose is not None:
                path = collector.path_planner.plan(collector.get_pose(), goal_pose)
                if path is not None:
                    collector.follow_path(path)
                    self.collector_busy = True
                    self.get_logger().info(f'Collector moving near fossil at ({x}, {y})')
                else:
                    self.get_logger().warn(f'Failed to plan path to collection pose')
            else:
                self.get_logger().warn(f'Failed to find valid collection pose near ({x}, {y})')
                self.collection_queue.append(self.collection_queue.pop(0))
        
        elif self.collector_busy and not collector.is_moving():
            # At fossil location, collect fossil and plan return to base
            if self.collection_queue:
                x, y = self.collection_queue.pop(0)
                
                # Find and collect the fossil
                fossil_objects = [obj for obj in self.world.objects if obj.category == 'fossil']
                for fossil in fossil_objects:
                    dx = fossil.pose.x - x
                    dy = fossil.pose.y - y
                    distance = np.sqrt(dx*dx + dy*dy)
                    if distance < 0.3:  # Collection radius
                        self.collect_fossil(fossil, collector)
                        break
                
            self.collector_busy = False
            self.returning_to_base = True
            self.get_logger().info('Fossil collected, planning return to base')
            
            # Try to find a valid return pose near base
            goal_pose = self.get_valid_base_pose(collector)
            
            if goal_pose is not None:
                path = collector.path_planner.plan(collector.get_pose(), goal_pose)
                if path is not None:
                    collector.follow_path(path)
                    self.get_logger().info(f'Returning to base position at ({goal_pose.x:.2f}, {goal_pose.y:.2f})')
                else:
                    self.get_logger().error('Failed to plan path to valid base position')
                    self.returning_to_base = False
            else:
                self.get_logger().error('Could not find any valid base return position')
                self.returning_to_base = False
        
        elif self.returning_to_base and not collector.is_moving():
            time.sleep(0.5)  # Small delay to ensure we've reached position
            
            # Deposit fossils at base
            self.deposit_fossils_at_base(collector)
            
            self.returning_to_base = False
            self.get_logger().info('Successfully reached base station')
            self.get_logger().info(f'Remaining fossils in queue: {len(self.collection_queue)}')

def create_fossil_world():
    world = World()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    world.set_metadata(
        locations=os.path.join(current_dir, "fossil_location_data.yaml"),
        objects=os.path.join(current_dir, "fossil_object_data.yaml"),
    )

    exploration_coords = [(-5, -5), (5, -5), (5, 5), (-5, 5)]
    world.add_room(name="exploration_zone", footprint=exploration_coords, color=[0.8, 0.8, 0.8])

    # Add base stationf
    base = world.add_location(
        name="base_station0",  # Explicit name
        category="base_station",
        parent="exploration_zone",
        pose=Pose(x=0.0, y=0.0, yaw=0.0)
    )

    # Add fossils with explicit names
    world.add_location(
        name="fossil_site0",  # Explicit name
        category="fossil_site",
        parent="exploration_zone",
        pose=Pose(x=3.0, y=3.0, yaw=0.0)
    )

    world.add_location(
        name="fossil_site1",  # Explicit name
        category="fossil_site",
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


if __name__ == "__main__":
    rclpy.init()
    
    node = FossilExplorationNode()
    
    world = create_fossil_world()
    node.set_world(world)
    
    node.get_logger().info('Starting FossilExplorationNode')

    ros_thread = threading.Thread(target=lambda: node.start(wait_for_gui=True))
    ros_thread.start()

    start_gui(node.world)