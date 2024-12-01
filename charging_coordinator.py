from typing import Optional, Dict
from pyrobosim.core import Robot, World, Location
from pyrobosim.utils.pose import Pose
from pyrobosim.navigation import RRTPlanner
from pyrobosim.utils.motion import Path

class ChargingCoordinator:
    def __init__(self):
        self.charging_stations: Dict[str, str] = {}  # charger_name -> robot_name
        self.reservations: Dict[str, str] = {}  # charger_name -> robot_name
    
    def is_charger_available(self, charger_name: str) -> bool:
        return (charger_name not in self.charging_stations and 
                charger_name not in self.reservations)
    
    def get_robot_at_charger(self, charger_name: str) -> Optional[str]:
        return self.charging_stations.get(charger_name)
    
    def reserve_charger(self, charger_name: str, robot_name: str) -> bool:
        if self.is_charger_available(charger_name):
            self.reservations[charger_name] = robot_name
            return True
        return False
    
    def start_charging(self, charger_name: str, robot_name: str) -> bool:
        if (charger_name in self.reservations and 
            self.reservations[charger_name] == robot_name):
            del self.reservations[charger_name]
            self.charging_stations[charger_name] = robot_name
            return True
        return False
    
    def stop_charging(self, charger_name: str, robot_name: str) -> bool:
        if (charger_name in self.charging_stations and 
            self.charging_stations[charger_name] == robot_name):
            del self.charging_stations[charger_name]
            return True
        return False
    
    def get_best_available_charger(
        self,
        start: Pose, 
        world: World, 
        planner: RRTPlanner,
        goal: Pose = None
    ) -> tuple[Optional[Path], Optional[Location]]:
        chargers: list[Location] = world.get_locations(["charger"])
        path_length = None
        best_charger = None
        path_to_charger = None

        for charger in chargers:
            if not self.is_charger_available(charger.name):
                continue
                
            for docking_pose in charger.nav_poses:
                plan_to_charger = planner.plan(start, docking_pose)
                
                if plan_to_charger is None or plan_to_charger.num_poses == 0:
                    continue
                    
                if goal is not None:
                    plan_to_goal = planner.plan(docking_pose, goal)
                    if plan_to_goal is None:
                        continue
                    length = plan_to_charger.length + plan_to_goal.length
                else:
                    length = plan_to_charger.length
                    
                if path_length is None or length < path_length:
                    path_length = length
                    best_charger = charger
                    path_to_charger = plan_to_charger
                    
        return path_to_charger, best_charger