import rclpy
import threading
import numpy as np
from std_msgs.msg import String
from rclpy.action import ActionClient
import json
from json import JSONDecodeError
from pyrobosim.core import Robot, ObjectSpawn, Location
from pyrobosim.core.objects import Object
from pyrobosim.utils.knowledge import query_to_entity
from pyrobosim.utils.motion import Path

from pyrobosim.gui import start_gui
from pyrobosim.navigation import RRTPlanner
from pyrobosim.utils.pose import Pose
from pyrobosim.utils.general import get_data_folder
from pyrobosim_ros.ros_interface import WorldROSWrapper
from pyrobosim_msgs.msg import TaskAction, TaskPlan

from pyrobosim_msgs.action import ExecuteTaskPlan

from pyrobosim.planning.actions import TaskAction, TaskPlan

from fossil_world_generation import create_fossil_world
from battery import Battery


class FossilException(Exception):
    """smtg"""


class AlreadyHoldingFossil(FossilException):
    def __str__(self) -> str:
        return "Robot is already holding a fossil."


COLLECTOR = "collector"
FOSSIL_DISCOVERIES_TOPIC = "fossil_discoveries"


class CollectorRobot(WorldROSWrapper):

    def __init__(self):
        super().__init__(state_pub_rate=0.1, dynamics_rate=0.01)

        self.fossil_discovery_sub = self.create_subscription(
            String, FOSSIL_DISCOVERIES_TOPIC, self.fossil_discovery_callback, 10
        )
        self.collection_queue = []

        self.is_holding_fossil = False

        self.battery_timer = self.create_timer(1.0, self.battery_behavior)

        self.base_station_timer = self.create_timer(1.0, self.base_station_behaviour)

        # Action client for a task plan
        self.plan_client = ActionClient(self, ExecuteTaskPlan, "execute_task_plan")

        self.first = True

        # battery
        self.battery = Battery(100.0, [], 2.0, 1.0, {})

        self.on_charge_finished_path = None
        self.after_charge_action = None  # "pickup", "place"

    def get_robot(self) -> Robot:
        collector: Robot = self.world.get_robot_by_name(COLLECTOR)
        return collector

    def is_at_charger(self):
        locs: list[Location] = self.world.get_locations(["charger"])
        robot_pose = self.get_robot().get_pose()
        for loc in locs:
            for pose in loc.nav_poses:
                # print("checking", pose)
                if robot_pose.is_approx(pose):
                    return True
        return False

    def is_at_base(self):
        locs: list[Location] = self.world.get_locations(["base_station"])
        robot_pose = self.get_robot().get_pose()
        for loc in locs:
            for pose in loc.nav_poses:
                # print("checking", pose)
                if robot_pose.is_approx(pose):
                    return True
        return False

    def get_nearest_pose(self, poses):
        nearest_pose = None
        min_dist = None

        robot = self.get_robot()
        for p in poses:
            dist = robot.get_pose().get_linear_distance(p)

            if min_dist is None or dist < min_dist:
                min_dist = dist
                nearest_pose = p
        return nearest_pose

    def get_nearest_base_nav_pose(self):
        base: Location = self.world.get_location_by_name("base_station0")
        return self.get_nearest_pose(base.nav_poses)

    def get_pickup_pose_for_object(self, fossil_obj_name):
        fossil_obj: Object = self.world.get_object_by_name(fossil_obj_name)
        spawn: ObjectSpawn = fossil_obj.parent
        return self.get_nearest_pose(spawn.nav_poses)

    def base_station_behaviour(self):
        robot = self.get_robot()
        if self.is_at_base() and robot.manipulated_object is not None:
            robot.place_object()
            self.collection_queue.pop(0)

            try:
                # self.go_to_fossil_safely(data["name"])

                # g = Pose(x=-9.457152293593143, y=-6.258005503665991, z=0.00, q=[1.0, 0.0, 0.0, 0.0])
                g = self.get_pickup_pose_for_object(self.collection_queue[0])
                self.go_to_pose_through_charger(g)
            except Exception as error:
                self.get_logger().warn("Warning")

    def battery_behavior(self, charge_rate: float = 10.0, drain_rate: float = 10.0):
        robot = self.get_robot()

        if self.is_at_charger():
            print("charging...")
            new_charge = min(self.battery.charge + charge_rate, 100.0)
            self.change_battery_charge(new_charge)

            if (
                self.battery.charge == 100.0
                and self.on_charge_finished_path is not None
            ):
                result = self.follow_path_with_drain(self.on_charge_finished_path)
                self.on_charge_finished_path = None

                if self.after_charge_action == "pickup":
                    print("Trying to pick up fossil.")
                    robot.pick_object("fossil")
                    nearest_base_pose = self.get_nearest_base_nav_pose()
                    self.go_to_pose_through_charger(nearest_base_pose)

                # elif self.after_charge_action == "place":
                #     print("Trying to place fossil.")
                #     robot.place_object()
                self.after_charge_action = None

        # object = world.get_object_by_name("fossil1")
        # spawn: ObjectSpawn = object.parent
        # for pose in spawn.nav_poses:
        #     print("fossil1", pose.x, pose.y, pose.z, pose.q)

        # elif robot.is_moving():
        #     robot.battery_level = max(self.battery.charge - drain_rate, 0.0)
        #     self.battery.charge = min(self.battery.charge - drain_rate, 0.0)

        self.world.gui.on_robot_changed()

    def fossil_discovery_callback(self, msg):
        """Handle new messages on fossil discovery topic.

        accept messages of form {"name": "string"}
        """
        try:
            data = json.loads(msg.data)
            self.get_logger().info(f"Recieved fossil data:\n{data}")
        except JSONDecodeError:
            self.get_logger().error(
                f"Wrong message format recieved on topic {FOSSIL_DISCOVERIES_TOPIC}, cannot convert to json: {msg.data}"
            )
            return

        if data["name"]:
            fossil_obj_name = data["name"]

            if len(self.collection_queue) == 0:
                try:
                    # self.go_to_fossil_safely(data["name"])

                    # g = Pose(x=-9.457152293593143, y=-6.258005503665991, z=0.00, q=[1.0, 0.0, 0.0, 0.0])
                    g = self.get_pickup_pose_for_object(fossil_obj_name)
                    self.go_to_pose_through_charger(g)
                except Exception as error:
                    self.get_logger().warn("Warning")

            self.collection_queue.append(fossil_obj_name)
            # parent: Location = world.get_object_by_name(data["name"]).parent
            # g = parent.nav_poses[0]
            # self.go_to_pose(g)

            # self.execute_collect_fossil_plan(data["name"])

        if msg.data.startswith("FOSSIL_FOUND:"):
            x, y = map(float, msg.data.split(":")[1].split(","))
            if (x, y) not in [
                (loc[0], loc[1]) for loc in self.collection_queue
            ]:  # Prevent duplicates
                self.collection_queue.append((x, y))
                self.get_logger().info(
                    f"Added fossil at ({x}, {y}) to collection queue"
                )

                self.go_to(x, y)

    def go_to(self, x, y):

        # Get a valid pose near the fossil
        # goal_pose = self.get_valid_collection_pose(x, y, collector)
        goal_pose = Pose(x=x, y=y, yaw=np.pi)

        self.go_to_pose(goal_pose)

    def go_to_pose(self, pose):
        collector: Robot = self.world.get_robot_by_name("collector")
        goal_pose = pose

        if goal_pose is not None:
            path = collector.path_planner.plan(collector.get_pose(), goal_pose)
            if path is not None:
                collector.follow_path(path)
                self.collector_busy = True
                # self.get_logger().info(f'Collector moving near fossil at ({x}, {y})')
            else:
                self.get_logger().warn(f"Failed to plan path to collection pose")
        else:
            # self.get_logger().warn(f'Failed to find valid collection pose near ({x}, {y})')
            self.collection_queue.append(self.collection_queue.pop(0))

    def closest_location(self):
        robot: Robot = self.world.get_robot_by_name(COLLECTOR)
        locations: list[Location] = self.world.get_locations()

        print("collector pos", robot.get_pose())
        print("locations", len(locations), [l.pose for l in locations])
        dists = [robot.get_pose().get_linear_distance(l.pose) for l in locations]
        min_index = np.argmin(dists)

        self.get_logger().info(f"min: {min_index}")
        for l in locations:
            d = robot.get_pose().get_linear_distance(l.pose)
            self.get_logger().info(f"dist: {d}, {type(d)}")

        return locations[min_index]

    def get_plan_to_fossil(self, fossil_name: str) -> Path:
        """Return path planned by robot's RRTPlanner."""
        # get nearest navigable point to fossil
        query_list = [elem for elem in fossil_name.split(" ") if elem]
        goal = query_to_entity(
            self.world,
            query_list,
            mode="location",
            robot=self.get_robot(),
            resolution_strategy="nearest",
        )
        goal_node = self.world.graph_node_from_entity(goal, robot=self.get_robot())

        return self.get_plan_to_pose(goal_node.pose)

    def get_plan_to_pose(self, goal_pose: Pose) -> Path:
        """Return path planned by robot's RRTPlanner."""
        planner: RRTPlanner = self.get_robot().path_planner
        start = self.get_robot().get_pose()
        print("planning", start, goal_pose)
        plan = planner.plan(start, goal_pose)
        print("plan", plan)
        return plan

    def execute_collect_fossil_plan(self, fossil_obj_name: str):
        actions = [
            TaskAction(type="navigate", target_location=fossil_obj_name),
            TaskAction(type="detect", object=fossil_obj_name),
            TaskAction(type="pick", object=fossil_obj_name),
            TaskAction(type="navigate", target_location="base_station"),
            TaskAction(type="place"),
        ]
        plan = TaskPlan(actions=actions)
        robot = self.world.get_robot_by_name(COLLECTOR)
        result, num_completed = robot.execute_plan(plan)

        self.get_logger().info(f"result: {num_completed}, {result.message}")

        planner: RRTPlanner = robot.path_planner
        print("latest path poses:", planner.latest_path.poses)
        self.battery.completeDrainForPath(
            planner.latest_path.poses, planner.latest_path.poses[0]
        )
        print("drain result:", self.battery.charge)

    def go_to_fossil_safely(self, fossil_name):
        query_list = [elem for elem in fossil_name.split(" ") if elem]
        goal = query_to_entity(
            self.world,
            query_list,
            mode="location",
            robot=self.get_robot(),
            resolution_strategy="nearest",
        )
        goal_node = self.world.graph_node_from_entity(goal, robot=self.get_robot())

        # self.go_to_pose_safely(goal_node.pose)
        self.go_to_pose_through_charger(goal_node.pose)

    def change_battery_charge(self, new_charge):
        self.battery.charge = new_charge
        self.get_robot().battery_level = new_charge
        self.world.gui.on_robot_changed()

    def follow_path_with_drain(self, path: Path):
        robot = self.get_robot()
        result = robot.follow_path(path)
        # update battery charge
        drain = self.battery.get_drain_for_path(path)
        self.change_battery_charge(self.battery.charge - drain)

        return result

    def go_to_pose_through_charger(self, goal: Pose):
        path = self.get_plan_to_pose(goal)
        robot = self.world.get_robot_by_name(COLLECTOR)

        if self.is_safe_path(path):
            result = self.follow_path_with_drain(path)
            if robot.manipulated_object is None:
                robot.pick_object("fossil")
                # go home to base (placing will be done automatically)
                # g = Pose(x=0.0, y=1.5, yaw=-1.57)
                g = self.get_nearest_base_nav_pose()
                self.go_to_pose_through_charger(g)

        else:
            # go to charger on the way
            planner: RRTPlanner = robot.path_planner
            path_to_charger, charger_location = self.battery.get_optimal_charger(
                robot.get_pose(), self.world, planner, goal=goal
            )
            print("optimal charger:", charger_location)

            result = self.follow_path_with_drain(path_to_charger)

            path_from_charger_to_goal = planner.plan(
                path_to_charger.poses[path_to_charger.num_poses - 1], goal
            )
            self.on_charge_finished_path = path_from_charger_to_goal
            self.after_charge_action = (
                "place" if robot.manipulated_object is not None else "pickup"
            )

            # result = robot.follow_path(path_from_charger_to_goal)

    def is_safe_path(self, path: Path) -> bool:
        """Returns whther the robot can complete path and then go to a charger worst case scenario."""
        robot = self.world.get_robot_by_name(COLLECTOR)
        goal_pose = path.poses[path.num_poses - 1]
        path_to_charger, charger_location = self.battery.get_optimal_charger(
            goal_pose, self.world, robot.path_planner
        )

        drain_path = self.battery.get_drain_for_path(path)
        drain_charger_path = self.battery.get_drain_for_path(path_to_charger)

        return self.battery.charge >= drain_path + drain_charger_path


if __name__ == "__main__":
    rclpy.init()

    node = CollectorRobot()

    world = create_fossil_world(n_rocks=8, random_seed=237)
    node.set_world(world)

    node.get_logger().info("Starting FossilExplorationNode")

    ros_thread = threading.Thread(target=lambda: node.start(wait_for_gui=True))
    ros_thread.start()

    start_gui(node.world)
