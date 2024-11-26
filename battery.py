from pyrobosim.utils.pose import Pose
from pyrobosim.utils.path import Path
import time

class ChargingStation:
    def __init__(self, chargingRate: float, location: Pose):
        self.chargingRate = chargingRate #How much charge is transferred per second
        self.location = location

    def getLocation(self):
        return self.location

    def charge(self, battery: "Battery"):
        batteryCapacity = battery.getCapacity()
        batteryCurrent = battery.getCharge()
        toCharge = batteryCapacity - batteryCurrent
        while(toCharge > 0):
            time.sleep(1)
            toCharge -= self.chargingRate
        battery.charge = batteryCapacity


class Battery:
    def __init__(self, capacity: float, chargingStations: list["ChargingStation"], drainPerDistanceUnit: float, drainPerRadianRotate: float,actionDrains: dict[str:float]):
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
    
    def calculateDrainToChargingStation(self, current: Pose, chargingStation:"ChargingStation"):
        drain = 0.0
        drain += current.get_linear_distance(chargingStation.getLocation()) * self.drainPerDistanceUnit
        drain += abs(current.get_angular_distance(chargingStation.getLocation()) * self.drainPerRadianRotate)
        return drain
    
    def batteryPermission(self, path: list[Pose], startingPose: Pose, actions: list[str] = None):
        workingCharge = self.getCharge()
        currentPose = startingPose
        stepCount = 0
        while (stepCount<len(path) and workingCharge >= 0):
            workingCharge -= currentPose.get_linear_distance(path[stepCount]) * self.drainPerDistanceUnit
            workingCharge -= abs(currentPose.get_angular_distance(path[stepCount]) * self.drainPerRadianRotate)
            stepCount += 1
            currentPose = path[stepCount]
        if (actions and len(actions)>0):
            for action in actions:
                workingCharge -= self.actionDrains[action]
        if (workingCharge<0.0):
            return False
        availableChargingStation = False
        stationCount = 0
        while (availableChargingStation == False and stationCount<len(self.chargingStations)):
            if (workingCharge - self.calculateDrainToChargingStation(currentPose, self.chargingStations[stationCount])>=0):
                availableChargingStation = True
        if (availableChargingStation == True):
            return True
        else:
            return False
        
    def completeDrainForPath(self, path: list[Pose], startingPose: Pose, actions: list[str] = None):
        workingCharge = self.getCharge()
        currentPose = startingPose
        stepCount = 0
        while (stepCount<len(path) and workingCharge >= 0):
            workingCharge -= currentPose.get_linear_distance(path[stepCount]) * self.drainPerDistanceUnit
            workingCharge -= abs(currentPose.get_angular_distance(path[stepCount]) * self.drainPerRadianRotate)
            stepCount += 1
            currentPose = path[stepCount]
        if (actions and len(actions)>0):
            for action in actions:
                workingCharge -= self.actionDrains[action]
        self.charge = workingCharge
        
    