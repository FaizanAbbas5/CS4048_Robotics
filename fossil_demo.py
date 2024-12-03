import os
from fossil_world_generation import create_fossil_world
import rclpy
import threading
import numpy as np
import time
import random
import json
from rclpy.node import Node
from geometry_msgs.msg import Point, Pose2D
from std_msgs.msg import String
from rclpy.executors import MultiThreadedExecutor

from typing import Optional, List, Tuple
import math
from shapely.geometry import Point, LineString

from pyrobosim.manipulation import GraspGenerator, ParallelGraspProperties

from pyrobosim.core import Robot, World, ObjectSpawn, Location
from pyrobosim.core.objects import Object

from pyrobosim.gui import start_gui
from pyrobosim.navigation import ConstantVelocityExecutor, RRTPlanner
from pyrobosim.utils.pose import Pose
from pyrobosim.utils.general import get_data_folder
from pyrobosim_ros.ros_interface import WorldROSWrapper
from fossil_collector_robot import CollectorRobot, FOSSIL_DISCOVERIES_TOPIC
from charging_coordinator import ChargingCoordinator
from explorer_battery_manager import ExplorerBatteryManager


class ExplorationGrid:
    # Model the world to allow the explorer to explore it
    def __init__(self, width, height, world=None, resolution=0.5):
        self.world = world
        self.resolution = resolution
        self.width = int(width / resolution)
        self.height = int(height / resolution)

        # -1 = unknown, 0 = open space, 1 = obstacle
        self.grid = np.full((self.width, self.height), -1)

        # Store object locations and types
        self.object_locations = {}  # (x,y) -> {'type': type, 'certainty': value}

        # Track coverage and exploration
        self.coverage_count = np.zeros((self.width, self.height))
        self.last_visit_time = np.zeros((self.width, self.height))
        self.current_time = 0

        # Track boundaries for valid exploration
        self.world_bounds = {
            "min_x": -width / 2 + 1.0,  # Add margin from edge
            "max_x": width / 2 - 1.0,
            "min_y": -height / 2 + 1.0,
            "max_y": height / 2 - 1.0,
        }

        # The explorer traverses the map with a spiral movement
        self.exploration_mode = "spiral"  # ['spiral', 'wall_follow', 'parallel']
        self.spiral_params = {
            "angle": 0.0,
            "radius": 1.0,
            "step": 0.3,
            "max_radius": min(width, height) / 3,
        }
        self.parallel_params = {
            "current_line": 0,
            "line_spacing": 2.0,
            "direction": 1,  # 1 or -1
        }
        self.wall_follow_params = {
            "wall_distance": 1.0,
            "current_wall": None,  # Track which wall we're following
            "follow_direction": 1,  # 1 for right wall, -1 for left wall
        }

        self.stuck_count = 0
        self.MAX_STUCK_COUNT = 5
        self.failed_targets = set()  # Keep track of unreachable targets

    def set_world(self, world):
        self.world = world

    def update_cell(self, x, y, cell_type, certainty=1.0):
        grid_x, grid_y = self.world_to_grid(x, y)
        if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
            if cell_type == "obstacle":
                self.grid[grid_x, grid_y] = 1
            elif cell_type == "open":
                self.grid[grid_x, grid_y] = 0

            self.current_time += 1
            self.coverage_count[grid_x, grid_y] += 1
            self.last_visit_time[grid_x, grid_y] = self.current_time

    def add_object(self, x, y, obj_type, certainty=1.0):
        # adds objects to the world, takes x and y coordinates and the object type as input
        grid_x, grid_y = self.world_to_grid(x, y)
        key = (grid_x, grid_y)
        self.object_locations[key] = {
            "type": obj_type,
            "certainty": certainty,
            "world_x": x,
            "world_y": y,
        }
        self.update_cell(x, y, "obstacle", certainty)

    def is_valid_point(self, x, y):
        return (
            self.world_bounds["min_x"] <= x <= self.world_bounds["max_x"]
            and self.world_bounds["min_y"] <= y <= self.world_bounds["max_y"]
        )

    def get_exploration_direction(self, current_x, current_y):
        grid_x, grid_y = self.world_to_grid(current_x, current_y)

        local_coverage_threshold = 5
        radius = 2
        local_area_visits = 0

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                x, y = grid_x + dx, grid_y + dy
                if 0 <= x < self.width and 0 <= y < self.height:
                    local_area_visits += self.coverage_count[x, y]

        if local_area_visits > local_coverage_threshold:
            # Look for less-visited areas
            best_score = float("inf")
            best_dir = None
            search_radius = 5

            for dx in range(-search_radius, search_radius + 1):
                for dy in range(-search_radius, search_radius + 1):
                    x, y = grid_x + dx, grid_y + dy
                    if 0 <= x < self.width and 0 <= y < self.height:
                        coverage_score = self.coverage_count[x, y]
                        time_factor = (
                            self.current_time - self.last_visit_time[x, y]
                        ) / 100.0
                        combined_score = coverage_score - time_factor

                        if combined_score < best_score:
                            world_x, world_y = self.grid_to_world(x, y)
                            if self.is_valid_point(world_x, world_y):
                                best_score = combined_score
                                best_dir = (dx, dy)

            return best_dir

        return super().get_exploration_direction(current_x, current_y)

    def mark_area_explored(self, center_x, center_y, radius=2.0):
        grid_x, grid_y = self.world_to_grid(center_x, center_y)
        radius_cells = int(radius / self.resolution)

        coverage_increment = 0.5

        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                if dx * dx + dy * dy <= radius_cells * radius_cells:
                    x, y = grid_x + dx, grid_y + dy
                    if 0 <= x < self.width and 0 <= y < self.height:
                        if self.grid[x, y] == -1:
                            self.grid[x, y] = 0
                        self.coverage_count[x, y] += coverage_increment
                        self.last_visit_time[x, y] = self.current_time

    def world_to_grid(self, x, y):
        grid_x = int((x + self.width * self.resolution / 2) / self.resolution)
        grid_y = int((y + self.height * self.resolution / 2) / self.resolution)
        return min(max(grid_x, 0), self.width - 1), min(max(grid_y, 0), self.height - 1)

    def grid_to_world(self, grid_x, grid_y):
        world_x = grid_x * self.resolution - self.width * self.resolution / 2
        world_y = grid_y * self.resolution - self.height * self.resolution / 2
        return world_x, world_y

    def get_spiral_target(self, current_x, current_y):
        self.spiral_params["radius"] += self.spiral_params["step"]
        self.spiral_params["angle"] += math.pi / 8

        next_x = current_x + self.spiral_params["radius"] * math.cos(
            self.spiral_params["angle"]
        )
        next_y = current_y + self.spiral_params["radius"] * math.sin(
            self.spiral_params["angle"]
        )

        if self.spiral_params["radius"] >= self.spiral_params["max_radius"]:
            self.exploration_mode = "parallel"
            self.spiral_params["radius"] = 1.0
            self.spiral_params["angle"] = 0.0

        return next_x, next_y

    def get_parallel_target(self, current_x, current_y):
        line_y = (
            -self.height / 2
            + self.parallel_params["current_line"]
            * self.parallel_params["line_spacing"]
        )

        if self.parallel_params["direction"] == 1:
            target_x = self.world_bounds["max_x"] - 1.0
        else:
            target_x = self.world_bounds["min_x"] + 1.0

        if abs(current_x - target_x) < 1.0:
            self.parallel_params["direction"] *= -1
            self.parallel_params["current_line"] += 1

            if (
                self.parallel_params["current_line"]
                * self.parallel_params["line_spacing"]
                > self.height
            ):
                self.exploration_mode = "wall_follow"

        return target_x, line_y

    def get_wall_follow_target(self, current_x, current_y, obstacles):
        if not obstacles:
            return None, None

        nearest_obstacle = min(
            obstacles,
            key=lambda o: math.sqrt(
                (o.pose.x - current_x) ** 2 + (o.pose.y - current_y) ** 2
            ),
        )

        angle = math.atan2(
            nearest_obstacle.pose.y - current_y, nearest_obstacle.pose.x - current_x
        )

        target_x = nearest_obstacle.pose.x + (
            self.wall_follow_params["wall_distance"]
            * math.cos(
                angle + self.wall_follow_params["follow_direction"] * math.pi / 2
            )
        )
        target_y = nearest_obstacle.pose.y + (
            self.wall_follow_params["wall_distance"]
            * math.sin(
                angle + self.wall_follow_params["follow_direction"] * math.pi / 2
            )
        )

        return target_x, target_y

    def is_target_valid(self, x, y):
        if not self.is_valid_point(x, y):
            return False

        grid_x, grid_y = self.world_to_grid(x, y)
        if (grid_x, grid_y) in self.failed_targets:
            return False

        if self.grid[grid_x][grid_y] == 1:
            return False

        return True

    def get_next_exploration_target(self, current_x, current_y):
        if not self.is_valid_point(current_x, current_y):
            return 0, 0

        unexplored_count = 0
        total_cells = 0
        for x in range(self.width):
            for y in range(self.height):
                if self.is_valid_point(*self.grid_to_world(x, y)):
                    total_cells += 1
                    if self.grid[x][y] == -1:
                        unexplored_count += 1

        exploration_ratio = (1 - (unexplored_count / total_cells)) * 1.92

        if exploration_ratio >= 0.95:
            if not (abs(current_x) < 0.5 and abs(current_y) < 0.5):
                return 0, 0
            else:
                return current_x, current_y

        next_x, next_y = None, None

        if self.stuck_count >= self.MAX_STUCK_COUNT:
            self.stuck_count = 0
            self.exploration_mode = "spiral"
            self.spiral_params["radius"] = 1.0
            self.spiral_params["angle"] = 0.0
            return 0, 0

        for _ in range(5):
            if self.exploration_mode == "spiral":
                next_x, next_y = self.get_spiral_target(current_x, current_y)
            elif self.exploration_mode == "parallel":
                next_x, next_y = self.get_parallel_target(current_x, current_y)
            elif self.exploration_mode == "wall_follow":
                obstacles = [
                    loc
                    for loc in self.world.locations
                    if loc.category in ["rock", "bush"]
                ]
                next_x, next_y = self.get_wall_follow_target(
                    current_x, current_y, obstacles
                )
                if next_x is None:
                    self.exploration_mode = "spiral"
                    continue

            if next_x is not None and next_y is not None:
                if self.is_target_valid(next_x, next_y):
                    break
                else:
                    grid_x, grid_y = self.world_to_grid(next_x, next_y)
                    self.failed_targets.add((grid_x, grid_y))

            if self.exploration_mode == "spiral":
                self.spiral_params["radius"] += self.spiral_params["step"]
            elif self.exploration_mode == "parallel":
                self.parallel_params["current_line"] += 1

        if next_x is None or next_y is None:
            return 0, 0

        next_x = max(
            min(next_x, self.world_bounds["max_x"]), self.world_bounds["min_x"]
        )
        next_y = max(
            min(next_y, self.world_bounds["max_y"]), self.world_bounds["min_y"]
        )

        return next_x, next_y

    def map_obstacle_boundary(self, obstacle_pose, obstacle_type):
        if obstacle_type == "rock":
            radius = 1.25 / 2
        elif obstacle_type == "bush":
            radius = 0.6
        else:
            radius = 0.3

        grid_x, grid_y = self.world_to_grid(obstacle_pose.x, obstacle_pose.y)
        radius_cells = int(radius / self.resolution)

        safety_margin = int(0.5 / self.resolution)
        total_radius = radius_cells + safety_margin

        for dx in range(-total_radius, total_radius + 1):
            for dy in range(-total_radius, total_radius + 1):
                dist = math.sqrt(dx * dx + dy * dy)
                x, y = grid_x + dx, grid_y + dy
                if 0 <= x < self.width and 0 <= y < self.height:
                    if dist <= radius_cells:
                        self.grid[x, y] = 1
                    elif dist <= total_radius:
                        if self.grid[x, y] != 1:
                            self.grid[x, y] = 0.5

    def is_path_clear(self, start_x, start_y, end_x, end_y):
        start_grid_x, start_grid_y = self.world_to_grid(start_x, start_y)
        end_grid_x, end_grid_y = self.world_to_grid(end_x, end_y)

        line_points = self.get_line_points(
            start_grid_x, start_grid_y, end_grid_x, end_grid_y
        )

        for x, y in line_points:
            if 0 <= x < self.width and 0 <= y < self.height:
                if self.grid[x, y] == 1:
                    return False
        return True

    def get_line_points(self, x0, y0, x1, y1):
        # Get all grid points along a line using Bresenham's algorithm
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1

        if dx > dy:
            err = dx / 2.0
            while x != x1:
                points.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                points.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy

        points.append((x, y))
        return points

    def find_optimal_path(self, start_x, start_y, end_x, end_y, max_attempts=5):
        if self.is_path_clear(start_x, start_y, end_x, end_y):
            return [(end_x, end_y)]

        waypoints = []
        current_x, current_y = start_x, start_y

        for _ in range(max_attempts):
            blocked_point = self.find_blocking_point(current_x, current_y, end_x, end_y)
            if blocked_point is None:
                waypoints.append((end_x, end_y))
                break

            bypass_point = self.find_bypass_point(
                current_x, current_y, blocked_point, end_x, end_y
            )
            if bypass_point is None:
                break

            waypoints.append(bypass_point)
            current_x, current_y = bypass_point

            if self.is_path_clear(current_x, current_y, end_x, end_y):
                waypoints.append((end_x, end_y))
                break

        return waypoints

    def find_blocking_point(self, start_x, start_y, end_x, end_y):
        points = self.get_line_points(
            *self.world_to_grid(start_x, start_y), *self.world_to_grid(end_x, end_y)
        )

        for x, y in points:
            if 0 <= x < self.width and 0 <= y < self.height:
                if self.grid[x, y] == 1:
                    return self.grid_to_world(x, y)
        return None

    def find_bypass_point(
        self, start_x, start_y, blocked_point, end_x, end_y, radius=1.5
    ):
        for angle in np.linspace(0, 2 * np.pi, 16):
            test_x = blocked_point[0] + radius * math.cos(angle)
            test_y = blocked_point[1] + radius * math.sin(angle)

            if self.is_valid_point(test_x, test_y) and self.is_path_clear(
                start_x, start_y, test_x, test_y
            ):
                return (test_x, test_y)

        return None


class FossilExplorationNode(WorldROSWrapper):
    def __init__(self):
        super().__init__(state_pub_rate=0.1, dynamics_rate=0.01)
        # The grid is set to 20 x 20
        self.exploration_grid = ExplorationGrid(width=20.0, height=20.0)

        self.charging_coordinator = ChargingCoordinator()
        self.explorer_battery = None

        self.explorer_timer = self.create_timer(2.0, self.explorer_behavior)
        # This is a callback for managing the explorer's battery
        self.battery_timer = self.create_timer(1.0, self.battery_behavior)
        self.explorer_exploring = False
        self.explorer_status_pub = self.create_publisher(String, "explorer_status", 10)

        self.fossil_discovery_pub = self.create_publisher(
            String, FOSSIL_DISCOVERIES_TOPIC, 10
        )
        self.discovered_fossils = set()

        self.test_done = True
        self.returning_to_base = False
        self.test_timer = self.create_timer(5.0, self.test_fossil_discovery)

        self.fossil_discovery_sub = self.create_subscription(
            String, FOSSIL_DISCOVERIES_TOPIC, self.fossil_discovery_callback, 10
        )
        # collection queue is set to empty to begin with
        self.collection_queue = []
        self.collector_busy = False

        # Initially, there are no fossils collected
        self.collected_fossils = []

        self.get_logger().info(
            r"""
 __          __  _                            _          _   _            ______            _ _   ______            _                 _   _             
 \ \        / / | |                          | |        | | | |          |  ____|          (_) | |  ____|          | |               | | (_)            
  \ \  /\  / /__| | ___ ___  _ __ ___   ___  | |_ ___   | |_| |__   ___  | |__ ___  ___ ___ _| | | |__  __  ___ __ | | ___  _ __ __ _| |_ _  ___  _ __  
   \ \/  \/ / _ \ |/ __/ _ \| '_ ` _ \ / _ \ | __/ _ \  | __| '_ \ / _ \ |  __/ _ \/ __/ __| | | |  __| \ \/ / '_ \| |/ _ \| '__/ _` | __| |/ _ \| '_ \ 
    \  /\  /  __/ | (_| (_) | | | | | |  __/ | || (_) | | |_| | | |  __/ | | | (_) \__ \__ \ | | | |____ >  <| |_) | | (_) | | | (_| | |_| | (_) | | | |
     \/  \/ \___|_|\___\___/|_| |_| |_|\___|  \__\___/   \__|_| |_|\___| |_|  \___/|___/___/_|_| |______/_/\_\ .__/|_|\___/|_|  \__,_|\__|_|\___/|_| |_|
                                                                                                             | |                                        
                                                                                                             |_|                                        
    """
        )

        self.get_logger().info("FossilExplorationNode initialized")

    def safe_plan_path(self, robot, start_pose, goal_pose, retry_count=3):
        for attempt in range(retry_count):
            try:
                path = robot.path_planner.plan(start_pose, goal_pose)
                if path is not None:
                    return path
                offset = 0.2 * (attempt + 1)
                modified_goal = Pose(
                    x=goal_pose.x + np.random.uniform(-offset, offset),
                    y=goal_pose.y + np.random.uniform(-offset, offset),
                    yaw=goal_pose.yaw,
                )
                path = robot.path_planner.plan(start_pose, modified_goal)
                if path is not None:
                    return path
            except Exception as e:
                self.get_logger().warn(
                    f"Planning attempt {attempt + 1} failed: {str(e)}"
                )
                continue
        return None

    def set_world(self, world):
        super().set_world(world)
        self.exploration_grid.set_world(world)

        # Initialise explorer battery manager
        explorer = self.get_robot_by_name("explorer")
        if explorer:
            self.explorer_battery = ExplorerBatteryManager(
                explorer, world, self.charging_coordinator
            )

    def print_exploration_progress(self):
        # This method tracks the percentage of the map explored by the explorer
        unexplored = 0
        total = 0
        for x in range(self.exploration_grid.width):
            for y in range(self.exploration_grid.height):
                if self.exploration_grid.is_valid_point(
                    *self.exploration_grid.grid_to_world(x, y)
                ):
                    total += 1
                    if self.exploration_grid.grid[x][y] == -1:
                        unexplored += 1

        exploration_ratio = (1 - (unexplored / total)) * 1.92
        # The exploration progress can be seen through the terminal
        self.get_logger().info(f"Exploration progress: {exploration_ratio:.2%}")

    def collect_fossil(self, fossil, collector):
        # This method allows the 'collector' robot to collect fossils
        if not hasattr(fossil, "pose"):
            return False

        try:
            # here we get the collector robot for fossil collection
            robot = self.get_robot_by_name("collector")
            success = robot.pick_object("fossil1")
            self.get_logger().info(f"Collected fossil at ({success})")
            return success

            fossil.original_pose = Pose(
                x=fossil.pose.x, y=fossil.pose.y, yaw=fossil.pose.yaw
            )

            fossil.pose.x = collector.get_pose().x
            fossil.pose.y = collector.get_pose().y

            self.collected_fossils.append(fossil)

            # The coordinates of the collection site can be seen from the terminal
            self.get_logger().info(
                f"Collected fossil at ({fossil.original_pose.x:.2f}, {fossil.original_pose.y:.2f})"
            )
            return True

        except Exception as e:
            self.get_logger().error(f"Error collecting fossil: {str(e)}")
            return False

    def check_along_path(self, explorer):
        current_pose = explorer.get_pose()

        detected_objects = self.check_for_objects(current_pose)

        if detected_objects:
            for obj, category in detected_objects:
                self.get_logger().info(
                    f"Found {category} while moving at ({obj.pose.x:.2f}, {obj.pose.y:.2f})"
                )

                # This allows the explorer to publish information about an explored fossil
                if category == "fossil_site_box":
                    if obj not in self.discovered_fossils:
                        self.discovered_fossils.add(obj)
                        self.publish_fossil_object(obj)

                self.exploration_grid.add_object(obj.pose.x, obj.pose.y, category)

                if category in ["rock", "bush"]:
                    self.get_logger().info(
                        f"Obstacle dimensions: category={category}, position=({obj.pose.x:.2f}, {obj.pose.y:.2f})"
                    )
                    if hasattr(obj, "footprint"):
                        self.get_logger().info(f"Footprint info: {obj.footprint}")

    def publish_fossil_object(self, loc: Location):
        msg = String()
        data = {
            "name": "fossil" + loc.name[-1:],
        }
        # msg.data = f"FOSSIL_FOUND:{obj.pose.x},{obj.pose.y}"
        msg.data = json.dumps(data)
        self.fossil_discovery_pub.publish(msg)

    def get_robot_by_name(self, name):
        for robot in self.world.robots:
            if robot.name == name:
                return robot
        return None

    def get_unstuck_position(self, current_pose, search_radius=2.0, increments=8):
        for radius in np.arange(0.5, search_radius, 0.5):
            for angle in np.linspace(0, 2 * np.pi, increments):
                test_x = current_pose.x + radius * math.cos(angle)
                test_y = current_pose.y + radius * math.sin(angle)

                if self.exploration_grid.is_valid_point(
                    test_x, test_y
                ) and self.exploration_grid.is_path_clear(
                    current_pose.x, current_pose.y, test_x, test_y
                ):
                    return test_x, test_y

        return None, None

    def battery_behavior(self):
        # a method to update the explorer's battery
        if self.explorer_battery:
            self.explorer_battery.update_battery()

    def get_nearest_base_pose(self):
        base = self.world.get_location_by_name("base_station0")
        if not base:
            return None

        nearest_pose = None
        min_dist = float("inf")

        explorer = self.get_robot_by_name("explorer")
        if not explorer:
            return None

        for pose in base.nav_poses:
            dist = explorer.get_pose().get_linear_distance(pose)
            if dist < min_dist:
                min_dist = dist
                nearest_pose = pose

        return nearest_pose

    def explorer_behavior(self):
        if not hasattr(self, "world") or not self.explorer_battery:
            return

        explorer = self.get_robot_by_name("explorer")
        if explorer is None:
            return

        current_pose = explorer.get_pose()

        # Handle charging states
        if self.explorer_battery.needs_charging():
            if self.explorer_battery.is_at_charger():
                if self.explorer_battery.battery.charge < 95.0:
                    self.get_logger().info(
                        f"Currently charging... ({self.explorer_battery.battery.charge}%)"
                    )
                    return
                else:
                    self.get_logger().info("Fully charged, resuming exploration")
            else:
                self.get_logger().info("Battery level low, seeking charging station...")
                if explorer.is_moving():
                    if self.explorer_battery.is_scanning_enabled():
                        self.check_along_path(explorer)
                return

        # Only perform scanning if enabled in battery manager
        if self.explorer_battery.is_scanning_enabled():
            detected_objects = self.check_for_objects(current_pose)
            if detected_objects:
                for obj, category in detected_objects:
                    self.get_logger().info(
                        f"Detected {category} at ({obj.pose.x:.2f}, {obj.pose.y:.2f})"
                    )

                    if category in ["rock", "bush"]:
                        self.exploration_grid.map_obstacle_boundary(obj.pose, category)
                        self.get_logger().info(
                            f"Mapped {category} boundary at ({obj.pose.x:.2f}, {obj.pose.y:.2f})"
                        )

                    elif (
                        category == "fossil_site_box"
                        and obj not in self.discovered_fossils
                    ):
                        self.discovered_fossils.add(obj)
                        self.publish_fossil_object(obj)
                        self.exploration_grid.add_object(
                            obj.pose.x, obj.pose.y, category
                        )

                    self.exploration_grid.add_object(obj.pose.x, obj.pose.y, category)

            # Only mark area as explored if scanning is enabled
            self.exploration_grid.mark_area_explored(current_pose.x, current_pose.y)

        # Once the explorer has explored the required percentage of the map, it returns to the base
        # This is done to mimic a real-life situation, preserving the robot's battery.
        if not explorer.is_moving():
            next_x, next_y = self.exploration_grid.get_next_exploration_target(
                current_pose.x, current_pose.y
            )

            if (
                abs(current_pose.x) < 0.5
                and abs(current_pose.y) < 0.5
                and abs(next_x) < 0.5
                and abs(next_y) < 0.5
            ):
                self.get_logger().info("Exploration complete, staying at base")
                return

            if abs(next_x) < 0.1 and abs(next_y) < 0.1:
                self.get_logger().info("No more exploration targets, returning to base")
                base_pose = self.get_nearest_base_pose()
                if base_pose:
                    success = self.explorer_battery.execute_path_safely(base_pose)
                    if not success:
                        self.get_logger().warn(
                            "Failed to plan path to base, will try alternative approach"
                        )
                        unstuck_x, unstuck_y = self.get_unstuck_position(current_pose)
                        if unstuck_x is not None:
                            intermediate_pose = Pose(x=unstuck_x, y=unstuck_y)
                            success = self.explorer_battery.execute_path_safely(
                                intermediate_pose
                            )
                            if success:
                                self.get_logger().info("Found alternative path to base")
                return

            if next_x is not None and next_y is not None:
                waypoints = self.exploration_grid.find_optimal_path(
                    current_pose.x, current_pose.y, next_x, next_y
                )

                if waypoints:
                    attempted_waypoints = set()
                    success = False

                    for wx, wy in waypoints:
                        if (wx, wy) in attempted_waypoints:
                            continue

                        attempted_waypoints.add((wx, wy))
                        goal_pose = Pose(x=wx, y=wy)

                        try:
                            success = self.explorer_battery.execute_path_safely(
                                goal_pose
                            )

                            if success:
                                self.get_logger().info(
                                    f"Moving to waypoint ({wx:.2f}, {wy:.2f}) "
                                    f"in {self.exploration_grid.exploration_mode} mode"
                                )
                                self.exploration_grid.stuck_count = 0
                                self.print_exploration_progress()
                                break
                        except Exception as e:
                            self.get_logger().warn(f"Path execution failed: {str(e)}")
                            continue

                    if not success:
                        self.exploration_grid.stuck_count += 1
                        self.get_logger().warn(
                            f"Failed to reach any waypoints (stuck count: {self.exploration_grid.stuck_count})"
                        )

                        if (
                            self.exploration_grid.stuck_count
                            >= self.exploration_grid.MAX_STUCK_COUNT
                        ):
                            self.get_logger().info(
                                "Maximum stuck count reached, attempting to return to base"
                            )
                            base_pose = self.get_nearest_base_pose()
                            if base_pose:
                                try:
                                    success = self.explorer_battery.execute_path_safely(
                                        base_pose
                                    )
                                    if success:
                                        self.get_logger().info(
                                            "Successfully planning return to base"
                                        )
                                        self.exploration_grid.stuck_count = 0
                                        return
                                except Exception as e:
                                    self.get_logger().warn(
                                        f"Failed to plan return to base: {str(e)}"
                                    )
                else:
                    self.get_logger().info("Attempting to get unstuck...")
                    unstuck_x, unstuck_y = self.get_unstuck_position(current_pose)

                    if unstuck_x is not None:
                        goal_pose = Pose(x=unstuck_x, y=unstuck_y)
                        try:
                            success = self.explorer_battery.execute_path_safely(
                                goal_pose
                            )

                            if success:
                                self.get_logger().info(
                                    f"Moving to clear position at ({unstuck_x:.2f}, {unstuck_y:.2f})"
                                )
                                self.exploration_grid.stuck_count = 0
                                return
                        except Exception as e:
                            self.get_logger().warn(
                                f"Failed to move to unstuck position: {str(e)}"
                            )

                    self.exploration_grid.stuck_count += 1
                    if (
                        self.exploration_grid.stuck_count
                        >= self.exploration_grid.MAX_STUCK_COUNT
                    ):
                        self.get_logger().warn(
                            "Unable to find clear path, switching exploration mode"
                        )
                        if self.exploration_grid.exploration_mode == "spiral":
                            self.exploration_grid.exploration_mode = "parallel"
                        elif self.exploration_grid.exploration_mode == "parallel":
                            self.exploration_grid.exploration_mode = "wall_follow"
                        else:
                            self.exploration_grid.exploration_mode = "spiral"
                            self.spiral_params = {
                                "angle": 0.0,
                                "radius": 1.0,
                                "step": 0.3,
                            }
                        self.exploration_grid.stuck_count = 0
            else:
                base_pose = self.get_nearest_base_pose()
                if base_pose:
                    self.get_logger().info(
                        "No exploration targets found, returning to base"
                    )
                    try:
                        success = self.explorer_battery.execute_path_safely(base_pose)
                        if not success:
                            self.get_logger().warn(
                                "Failed to plan path to base, will try again"
                            )
                    except Exception as e:
                        self.get_logger().warn(f"Error returning to base: {str(e)}")

        if explorer.is_moving() and self.explorer_battery.is_scanning_enabled():
            self.check_along_path(explorer)
            self.exploration_grid.mark_area_explored(current_pose.x, current_pose.y)

    def get_first_object(self):
        fossil_objects: list[Object] = [
            obj for obj in self.world.objects if obj.category == "fossil"
        ]
        if len(fossil_objects) > 0:
            first = fossil_objects[0]
            return first

    def check_line_of_sight(self, start_x, start_y, end_x, end_y, obstacles):
        for obstacle in obstacles:
            if obstacle.category not in ["rock", "bush"]:
                continue

            line = LineString([(start_x, start_y), (end_x, end_y)])

            if hasattr(obstacle, "polygon") and obstacle.polygon.intersects(line):
                return False

            if hasattr(obstacle, "pose"):
                obstacle_point = Point(obstacle.pose.x, obstacle.pose.y)
                if line.distance(obstacle_point) < 0.6:
                    return False
        return True

    def check_for_objects(
        self,
        robot_pose,
        detection_params={
            "radius": 3.0,
            "min_probability": 0.3,  # Lowered threshold for better detection
            "n_rays": 64,
            "scan_interval": 0.2,
        },
    ):
        if not hasattr(self, "world"):
            return None

        fossil_sites = [
            loc for loc in self.world.locations if loc.category == "fossil_site_box"
        ]
        static_obstacles = [
            loc for loc in self.world.locations if loc.category in ["rock", "bush"]
        ]

        robot = self.get_robot_by_name("explorer")
        detected_objects = []

        scan_points = []
        for radius in np.arange(
            0, detection_params["radius"], detection_params["scan_interval"]
        ):
            angles = np.linspace(0, 2 * np.pi, detection_params["n_rays"])
            for angle in angles:
                x = robot_pose.x + radius * np.cos(angle)
                y = robot_pose.y + radius * np.sin(angle)
                scan_points.append((x, y))

        for obj in fossil_sites + static_obstacles:
            best_visibility = 0
            closest_distance = float("inf")

            obj_radius = 0.15 if obj.category == "fossil_site_box" else 0.6

            for scan_x, scan_y in scan_points:
                dx = obj.pose.x - scan_x
                dy = obj.pose.y - scan_y
                distance = math.sqrt(dx * dx + dy * dy)

                if distance <= detection_params["radius"]:
                    if self.check_line_of_sight(
                        scan_x, scan_y, obj.pose.x, obj.pose.y, static_obstacles
                    ):
                        visibility = 1.0 - (distance / detection_params["radius"])
                        best_visibility = max(best_visibility, visibility)
                        closest_distance = min(closest_distance, distance)

            if best_visibility >= detection_params["min_probability"]:
                self.get_logger().info(
                    f"Detected {obj.category} at ({obj.pose.x:.2f}, {obj.pose.y:.2f}) "
                    f"with visibility {best_visibility:.2f} at distance {closest_distance:.2f}"
                )
                detected_objects.append((obj, obj.category))

        return detected_objects

    def get_valid_collection_pose(self, fossil_x, fossil_y, robot, approach_radius=0.8):
        current_pose = robot.get_pose()
        self.get_logger().info(
            f"Robot at ({current_pose.x:.2f}, {current_pose.y:.2f}), trying to reach fossil at ({fossil_x:.2f}, {fossil_y:.2f})"
        )

        for radius in [0.8, 1.0, 1.2]:
            angles = np.linspace(0, 2 * np.pi, 16)

            for i, angle in enumerate(angles):
                x = fossil_x + radius * np.cos(angle)
                y = fossil_y + radius * np.sin(angle)

                self.get_logger().debug(
                    f"Trying approach position {i+1}/16 at ({x:.2f}, {y:.2f}) with radius {radius}"
                )

                goal_pose = Pose(x=x, y=y)
                path = self.safe_plan_path(robot, robot.get_pose(), goal_pose)

                if path is not None:
                    self.get_logger().info(
                        f"Found valid collection pose at ({x:.2f}, {y:.2f}) with radius {radius}"
                    )
                    return goal_pose

        self.get_logger().warn(
            f"Could not find any valid approach positions for fossil at ({fossil_x}, {fossil_y})"
        )
        return None

    def get_valid_base_pose(self, robot, base_radius=0.8):
        current_pose = robot.get_pose()
        self.get_logger().info(
            f"Robot at ({current_pose.x:.2f}, {current_pose.y:.2f}), trying to find path to base"
        )

        angles = np.linspace(0, 2 * np.pi, 16)

        for radius in [0.8, 1.0, 1.2]:
            for i, angle in enumerate(angles):
                x = radius * np.cos(angle)
                y = radius * np.sin(angle)

                self.get_logger().debug(
                    f"Trying base approach position at ({x:.2f}, {y:.2f})"
                )

                goal_pose = Pose(x=x, y=y)
                path = self.safe_plan_path(robot, robot.get_pose(), goal_pose)

                if path is not None:
                    self.get_logger().info(
                        f"Found valid base return pose at ({x:.2f}, {y:.2f})"
                    )
                    return goal_pose

        self.get_logger().warn(
            "Could not find any valid approach positions for base return"
        )
        return None

    def remove_fossil_from_world(self, x, y, radius=0.3):
        # Method to deal with a fossil once it has been collected
        if not hasattr(self, "world"):
            return False

        fossil_sites = [
            loc for loc in self.world.locations if loc.category == "fossil_site"
        ]

        for site in fossil_sites:
            dx = site.pose.x - x
            dy = site.pose.y - y
            distance = np.sqrt(dx * dx + dy * dy)

            if distance < radius:
                try:
                    location_id = site.name if hasattr(site, "name") else None
                    self.get_logger().info(
                        f"Attempting to remove fossil with ID: {location_id}"
                    )

                    if hasattr(self.world, "remove_location"):
                        self.world.remove_location(location_id)
                        self.get_logger().info(
                            f"Removed fossil {location_id} at ({x:.2f}, {y:.2f}) using remove_location"
                        )
                    else:
                        self.world.locations.remove(site)
                        self.get_logger().info(
                            f"Removed fossil at ({x:.2f}, {y:.2f}) from locations list"
                        )

                    if site in self.discovered_fossils:
                        self.discovered_fossils.remove(site)
                    if hasattr(self, "publish_state"):
                        self.publish_state()

                    return True
                except Exception as e:
                    self.get_logger().error(f"Error removing fossil: {str(e)}")
                    self.get_logger().error(
                        f"Location properties: {dir(site)}"
                    )  # Debug info
                    return False

        return False

    def test_fossil_discovery(self):
        if not self.test_done:
            msg = String()
            msg.data = "FOSSIL_FOUND:3.0,3.0"
            self.fossil_discovery_pub.publish(msg)
            self.get_logger().info("Manually triggered fossil discovery at (3.0, 3.0)")
            self.test_done = True

    def fossil_discovery_callback(self, msg):
        if msg.data.startswith("FOSSIL_FOUND:"):
            x, y = map(float, msg.data.split(":")[1].split(","))
            if (x, y) not in [(loc[0], loc[1]) for loc in self.collection_queue]:
                self.collection_queue.append((x, y))
                self.get_logger().info(
                    f"Added fossil at ({x}, {y}) to collection queue"
                )

    def deposit_fossils_at_base(self, collector):
        # Once a fossil has been collected from the fossil site, the collector returns it to the base
        for fossil in self.collected_fossils:
            try:
                angle = len(self.collected_fossils) * (
                    2 * np.pi / max(len(self.collected_fossils), 1)
                )
                deposit_radius = 0.3

                fossil.pose.x = deposit_radius * np.cos(angle)
                fossil.pose.y = deposit_radius * np.sin(angle)

                self.get_logger().info(
                    f"Deposited fossil at base position ({fossil.pose.x:.2f}, {fossil.pose.y:.2f})"
                )

            except Exception as e:
                self.get_logger().error(f"Error depositing fossil: {str(e)}")

        self.collected_fossils = []


def main():
    # Initialize ROS2 (python client library)
    rclpy.init()

    # Create the world with a fixed random seed
    # The values can be changed to test the program in different settings
    world = create_fossil_world(n_rocks=8, n_bushes=4, random_seed=237)
    # world = create_fossil_world(width=30, height=40, n_rocks=10, n_bushes=4, n_chargers=2, n_fossils=20, random_seed=345)

    # Create the nodes
    exploration_node = FossilExplorationNode()
    collector_node = CollectorRobot()

    # Set the shared world for both nodes
    exploration_node.set_world(world)
    collector_node.set_world(world)

    # Use a MultiThreadedExecutor for managing multiple nodes
    executor = MultiThreadedExecutor()
    executor.add_node(exploration_node)
    executor.add_node(collector_node)

    # Define a function to spin the ROS nodes
    def spin_ros():
        exploration_node.get_logger().info("Starting nodes...")
        try:
            executor.spin()
        except KeyboardInterrupt:
            exploration_node.get_logger().info("Shutting down nodes...")
        finally:
            executor.shutdown()
            exploration_node.destroy_node()
            collector_node.destroy_node()
            rclpy.shutdown()

    # Run the ROS nodes in a separate thread
    ros_thread = threading.Thread(target=spin_ros, daemon=True)
    ros_thread.start()

    # Start the GUI in the main thread
    start_gui(world)


if __name__ == "__main__":
    main()
