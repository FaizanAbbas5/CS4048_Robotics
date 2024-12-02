from pyrobosim.core import Robot, World, ObjectSpawn, Location
from pyrobosim.utils.pose import Pose
from pyrobosim.utils.motion import Path
from pyrobosim.navigation import ConstantVelocityExecutor, RRTPlanner

import time


class ChargingStation:
    def __init__(self, chargingRate: float, location: Pose):
        self.chargingRate = chargingRate  # How much charge is transferred per second
        self.location = location

    def getLocation(self):
        return self.location

    def charge(self, battery: "Battery"):
        batteryCapacity = battery.getCapacity()
        batteryCurrent = battery.getCharge()
        toCharge = batteryCapacity - batteryCurrent
        while toCharge > 0:
            time.sleep(1)
            toCharge -= self.chargingRate
        battery.charge = batteryCapacity


class Battery:
    def __init__(
        self,
        capacity: float,
        chargingStations: list["ChargingStation"],
        drainPerDistanceUnit: float,
        drainPerRadianRotate: float,
        actionDrains: dict[str:float],
    ):
        self.capacity = capacity
        self.charge = capacity
        self.chargingStations = chargingStations
        self.drainPerDistanceUnit = drainPerDistanceUnit
        self.drainPerRadianRotate = drainPerRadianRotate
        self.actionDrains = actionDrains

    def getCharge(self):
        return self.charge

    def getCapacity(self):
        return self.capacity

    def calculateDrainToChargingStation(
        self, current: Pose, chargingStation: "ChargingStation"
    ):
        drain = 0.0
        drain += (
            current.get_linear_distance(chargingStation.getLocation())
            * self.drainPerDistanceUnit
        )
        drain += abs(
            current.get_angular_distance(chargingStation.getLocation())
            * self.drainPerRadianRotate
        )
        return drain

    def batteryPermission(
        self, path: list[Pose], startingPose: Pose, actions: list[str] = None
    ):
        workingCharge = self.getCharge()
        currentPose = startingPose
        stepCount = 0
        while stepCount < len(path) and workingCharge >= 0:
            workingCharge -= (
                currentPose.get_linear_distance(path[stepCount])
                * self.drainPerDistanceUnit
            )
            workingCharge -= abs(
                currentPose.get_angular_distance(path[stepCount])
                * self.drainPerRadianRotate
            )
            stepCount += 1
            currentPose = path[stepCount]
        if actions and len(actions) > 0:
            for action in actions:
                workingCharge -= self.actionDrains[action]
        if workingCharge < 0.0:
            return False
        availableChargingStation = False
        stationCount = 0
        while availableChargingStation == False and stationCount < len(
            self.chargingStations
        ):
            if (
                workingCharge
                - self.calculateDrainToChargingStation(
                    currentPose, self.chargingStations[stationCount]
                )
                >= 0
            ):
                availableChargingStation = True
        if availableChargingStation == True:
            return True
        else:
            return False

    def completeDrainForPath(
        self, path: list[Pose], startingPose: Pose, actions: list[str] = None
    ):
        workingCharge = self.getCharge()
        currentPose = startingPose
        stepCount = 0
        while stepCount < len(path) - 1 and workingCharge >= 0:
            currentPose = path[stepCount]
            workingCharge -= (
                currentPose.get_linear_distance(path[stepCount + 1])
                * self.drainPerDistanceUnit
            )
            workingCharge -= abs(
                currentPose.get_angular_distance(path[stepCount + 1])
                * self.drainPerRadianRotate
            )
            stepCount += 1
            print("wcharge:", workingCharge)
        if actions and len(actions) > 0:
            for action in actions:
                workingCharge -= self.actionDrains[action]
        self.charge = workingCharge

    def get_optimal_charger(
        self, start: Pose, world: World, planner: RRTPlanner, goal: Pose = None
    ) -> tuple[Path, Location]:
        """Returns the charger that adds the least amount of extra distance to visit."""
        chargers: list[Location] = world.get_locations(["charger"])

        path_length = None
        best_charger = None
        best_charger_pose = None
        path_to_charger = None

        for charger in chargers:
            for docking_pose in charger.nav_poses:
                plan_to_charger = planner.plan(start, docking_pose)

                if plan_to_charger.num_poses == 0:
                    print("couldnt find oath to charger", charger, docking_pose)
                    continue

                plan_to_goal = None
                if goal is not None:
                    plan_to_goal = planner.plan(docking_pose, goal)

                    length = plan_to_charger.length + plan_to_goal.length
                else:
                    length = plan_to_charger.length

                if path_length is None or length < path_length:
                    path_length = length
                    best_charger = charger
                    best_charger_pose = docking_pose
                    path_to_charger = plan_to_charger

        return (path_to_charger, best_charger)

    def can_complete_path(self, path: Path, with_charge: float = None) -> bool:
        """Returns whther battery capacity is enough to complete path."""
        charge = with_charge
        if charge is None:
            charge = self.charge
        return charge >= self.get_drain_for_path(path)

    def get_drain_for_path(self, path: Path) -> float:
        """Return the battery units needed to complete the given path."""
        drain = 0.0
        for i in range(path.num_poses - 1):
            current_pose = path.poses[i]
            next_pose = path.poses[i + 1]

            linear_drain = (
                current_pose.get_linear_distance(next_pose) * self.drainPerDistanceUnit
            )
            angular_drain = abs(
                current_pose.get_angular_distance(next_pose) * self.drainPerRadianRotate
            )

            drain += linear_drain + angular_drain
        # left actions out for now
        return drain
