from battery import Battery
from pyrobosim.core import Robot, World
from pyrobosim.utils.pose import Pose
from pyrobosim.utils.motion import Path
import numpy as np
import logging

class ExplorerBatteryManager:
    def __init__(self, robot: Robot, world: World, charging_coordinator):
        self.robot = robot
        self.world = world
        self.charging_coordinator = charging_coordinator
        
        # Initialise battery with parameters suited for explorer
        self.battery = Battery(
            capacity=100.0,
            chargingStations=[],
            drainPerDistanceUnit=1.0,  # Explorer moves faster, so drains more
            drainPerRadianRotate=0.5,
            actionDrains={"scan": 0.5}
        )
        
        # State management
        self.current_charger = None
        self.reserved_charger = None
        self.on_charge_finished_path = None
        self.charging = False
        self.MIN_SAFE_BATTERY = 30.0
        self.is_seeking_charger = False
        self.last_charge_update = 0
        self.pre_charge_position = None
        self.scanning_enabled = True
        self.path_execution_retries = 0
        self.MAX_PATH_RETRIES = 3
        
        # Configure logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def disable_scanning(self):
        self.scanning_enabled = False
        self.logger.info("Scanning disabled")
        
    def enable_scanning(self):
        self.scanning_enabled = True
        self.logger.info("Scanning enabled")
        
    def is_scanning_enabled(self):
        return self.scanning_enabled
        
    def update_battery(self, charge_rate: float=10.0):
        if self.is_at_charger():
            if not self.charging:
                charger = self.get_current_charger()
                if charger and self.charging_coordinator.start_charging(charger.name, self.robot.name):
                    self.charging = True
                    self.is_seeking_charger = False
                    self.disable_scanning()
                    self.last_charge_update = self.battery.charge
                    self.logger.info(f"Started charging at {self.battery.charge}%")
                    
            if self.charging:
                new_charge = min(self.battery.charge + charge_rate, 100.0)
                self.change_battery_charge(new_charge)
                self.logger.info(f"Current charge: {self.battery.charge}%")
                
                if new_charge <= self.last_charge_update:
                    self.logger.warning('Battery not charging properly, resetting charge state')
                    self.charging = False
                    self.current_charger = None
                    self.enable_scanning()
                    return
                    
                self.last_charge_update = new_charge
                
                if self.battery.charge >= 95.0:
                    self.logger.info("Fully charged, preparing to resume exploration")
                    self.finish_charging()

    def finish_charging(self):
        charger = self.get_current_charger()
        if charger:
            self.charging_coordinator.stop_charging(charger.name, self.robot.name)
            self.logger.info(f"Released charger {charger.name}")
        
        self.charging = False
        self.current_charger = None
        self.is_seeking_charger = False
        self.enable_scanning()
        
        if self.on_charge_finished_path is not None:
            self.logger.info("Attempting to resume planned path")
            if self.validate_stored_path():
                result = self.resume_after_charging()
                if result:
                    self.logger.info("Successfully resumed path")
                else:
                    self.logger.info("Failed to resume path, attempting replan")
                    self.replan_path()
            else:
                self.logger.info("Stored path invalid, replanning")
                self.replan_path()
            self.on_charge_finished_path = None
    
    def validate_stored_path(self) -> bool:
        if self.on_charge_finished_path is None:
            return False
        
        if not self.on_charge_finished_path.poses:
            self.logger.warning("Stored path has no poses")
            return False
            
        current_pose = self.robot.get_pose()
        path_start = self.on_charge_finished_path.poses[0]
        
        if current_pose.get_linear_distance(path_start) > 0.1:
            self.logger.warning("Current position too far from path start")
            return False
            
        try:
            new_path = self.robot.path_planner.plan(current_pose, self.on_charge_finished_path.poses[-1])
            if new_path is None:
                self.logger.warning("Path is no longer valid in current environment")
                return False
        except Exception as e:
            self.logger.error(f"Path validation error: {str(e)}")
            return False
            
        return True

    def resume_after_charging(self) -> bool:
        try:
            if self.on_charge_finished_path is None:
                return False
                
            self.path_execution_retries = 0
            while self.path_execution_retries < self.MAX_PATH_RETRIES:
                result = self.follow_path_with_drain(self.on_charge_finished_path)
                if result:
                    return True
                    
                self.path_execution_retries += 1
                self.logger.warning(f"Path execution retry {self.path_execution_retries}")
                
            return False
        except Exception as e:
            self.logger.error(f"Error resuming path: {str(e)}")
            return False

    def replan_path(self) -> bool:
        if self.pre_charge_position:
            try:
                new_path = self.robot.path_planner.plan(
                    self.robot.get_pose(),
                    self.pre_charge_position
                )
                if new_path:
                    self.logger.info("Successfully replanned path")
                    return self.follow_path_with_drain(new_path)
                else:
                    self.logger.warning("Failed to replan path")
            except Exception as e:
                self.logger.error(f"Path replanning error: {str(e)}")
        return False

    def is_at_charger(self) -> bool:
        chargers = self.world.get_locations(["charger"])
        robot_pose = self.robot.get_pose()
        POSITION_TOLERANCE = 0.1
        
        for charger in chargers:
            for nav_pose in charger.nav_poses:
                distance = robot_pose.get_linear_distance(nav_pose)
                if distance <= POSITION_TOLERANCE:
                    self.current_charger = charger
                    return True
        return False

    def needs_charging(self) -> bool:
        if self.is_at_charger() and self.battery.charge >= 100.0:
            if self.charging:
                charger = self.get_current_charger()
                if charger:
                    self.charging_coordinator.stop_charging(charger.name, self.robot.name)
                self.charging = False
                self.current_charger = None
                return False
                
        if self.is_at_charger() or self.is_seeking_charger:
            return True
                
        if self.battery.charge <= self.MIN_SAFE_BATTERY:
            path_to_charger, charger = self.charging_coordinator.get_best_available_charger(
                self.robot.get_pose(), 
                self.world,
                self.robot.path_planner
            )
            
            if path_to_charger is not None:
                if self.charging_coordinator.reserve_charger(charger.name, self.robot.name):
                    self.is_seeking_charger = True
                    self.pre_charge_position = self.robot.get_pose()
                    self.follow_path_with_drain(path_to_charger)
                    return True
            elif self.battery.charge <= 15.0:
                return True
        return False
    
    def get_current_charger(self):
        if self.is_at_charger():
            return self.current_charger
        return None
    
    def change_battery_charge(self, new_charge: float):
        self.battery.charge = new_charge
        self.robot.battery_level = new_charge
        if hasattr(self.world, 'gui'):
            self.world.gui.on_robot_changed()
            
    def follow_path_with_drain(self, path: Path) -> bool:
        try:
            result = self.robot.follow_path(path)
            if result:
                path_drain = self.battery.get_drain_for_path(path)
                scan_drain = self.battery.actionDrains.get("scan", 0) * (path.num_poses / 2)
                total_drain = path_drain + scan_drain
                self.change_battery_charge(self.battery.charge - total_drain)
            return result
        except Exception as e:
            self.logger.error(f"Path following error: {str(e)}")
            return False
            
    def execute_path_safely(self, goal: Pose) -> bool:
        if goal is None:
            self.logger.warning("Invalid goal pose provided")
            return False

        if self.needs_charging():
            self.pre_charge_position = self.robot.get_pose()
            
            path_to_charger, charger = self.charging_coordinator.get_best_available_charger(
                self.robot.get_pose(), 
                self.world,
                self.robot.path_planner
            )
            if path_to_charger is not None and path_to_charger.poses:  # Check if path has poses
                if self.charging_coordinator.reserve_charger(charger.name, self.robot.name):
                    self.reserved_charger = charger
                    self.enable_scanning()
                    
                    try:
                        path_from_charger = self.robot.path_planner.plan(
                            path_to_charger.poses[-1],
                            goal
                        )
                        if path_from_charger is not None and path_from_charger.poses:
                            self.on_charge_finished_path = path_from_charger
                            self.logger.info('Path to goal after charging found')
                        else:
                            # Try to plan return to pre-charge position
                            return_path = self.robot.path_planner.plan(
                                path_to_charger.poses[-1],
                                self.pre_charge_position
                            )
                            if return_path is not None and return_path.poses:
                                self.on_charge_finished_path = return_path
                                self.logger.info('Will return to pre-charge position after charging')
                            else:
                                self.logger.info('No valid path after charging, will resume exploration')
                        
                        self.is_seeking_charger = True
                        return self.follow_path_with_drain(path_to_charger)
                    except Exception as e:
                        self.logger.error(f"Error planning path from charger: {str(e)}")
                        return False
            else:
                self.logger.warning("Could not find valid path to charger")
                return False

        # Try direct path to goal
        try:
            success, path = self.plan_path_through_charger(goal)
            if not success or path is None or not path.poses:
                self.logger.warning("Could not plan valid path to goal")
                return False
                
            if self.reserved_charger:
                try:
                    path_from_charger = self.robot.path_planner.plan(
                        path.poses[-1],
                        goal
                    )
                    if path_from_charger is None or not path_from_charger.poses:
                        self.charging_coordinator.stop_charging(
                            self.reserved_charger.name,
                            self.robot.name
                        )
                        self.reserved_charger = None
                        return False
                    self.on_charge_finished_path = path_from_charger
                except Exception as e:
                    self.logger.error(f"Error planning path from charger to goal: {str(e)}")
                    return False
                
            result = self.follow_path_with_drain(path)
            return result
        except Exception as e:
            self.logger.error(f"Error executing path: {str(e)}")
            return False

    def plan_path_through_charger(self, goal: Pose) -> tuple[bool, Path]:
        try:
            direct_path = self.robot.path_planner.plan(self.robot.get_pose(), goal)
            
            if direct_path is None or not direct_path.poses:  # Check if path has poses
                self.logger.warning("Could not plan direct path to goal")
                return False, None
                
            if self.battery.can_complete_path(direct_path):
                return True, direct_path
                
            path_to_charger, charger = self.charging_coordinator.get_best_available_charger(
                self.robot.get_pose(), 
                self.world,
                self.robot.path_planner,
                goal
            )
            
            if path_to_charger is None or not path_to_charger.poses or charger is None:  # Check if path has poses
                self.logger.warning("Could not find valid path through charger")
                return False, None
                
            if self.charging_coordinator.reserve_charger(charger.name, self.robot.name):
                self.reserved_charger = charger
                return True, path_to_charger
                
            return False, None
        except Exception as e:
            self.logger.error(f"Path planning error: {str(e)}")
            return False, None