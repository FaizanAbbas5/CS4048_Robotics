import os
import random

from pyrobosim.core import Robot, World, Room
from pyrobosim.navigation import ConstantVelocityExecutor, RRTPlanner
from pyrobosim.utils.pose import Pose

def add_locations_to_room(
        world: World,
        room: Room,
        loc_category: str,
        count: int=5,
        random_seed: int=42
):
    random.seed(random_seed)

    locs = []
    minx, miny, maxx, maxy = room.polygon.bounds

    while len(locs) < count:
        x = random.uniform(minx, maxx)
        y = random.uniform(miny, maxy) 

        loc = world.add_location(
            # name=loc_name,
            category=loc_category,
            parent=room,
            pose=Pose(x=x, y=y, yaw=0.0)
        )

        if loc is not None:
            locs.append(loc)

            # try:
            #     # raises valueerror if object doesnt have enough space 
            #     fossil = world.add_object(category=obj_category, parent=fossil_site)
            # except ValueError:
            #     print(f"Fossil couldn't be placed inside location ({fossil_site.name}).")
        
    return locs

def add_fossils_to_room(
        world: World,
        room: Room,
        count: int=5,
        random_seed: int=42
):
    obj_category = "fossil"

    random.seed(random_seed)

    fossil_sites = add_locations_to_room(
        world,
        room,
        "fossil_site_box",
        random_seed=random_seed
    )

    for loc in fossil_sites:
        try:
            # raises valueerror if object doesnt have enough space 
            fossil = world.add_object(category=obj_category, parent=loc)
        except ValueError:
            print(f"Fossil couldn't be placed inside location ({loc.name}).")
        
    
def create_fossil_world(
        width: float=20.0,
        height: float=20.0,
        n_fossils: int=5,
        n_rocks: int=6,
        n_bushes: int=4,
        random_seed: int=42,
):
    world = World()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    world.set_metadata(
        locations=os.path.join(current_dir, "fossil_location_data.yaml"),
        objects=os.path.join(current_dir, "fossil_object_data.yaml"),
    )

    half_width = width / 2
    half_height = height / 2
    exploration_coords = [
        (-half_width, -half_height),
        (half_width, -half_height),
        (half_width, half_height),
        (-half_width, half_height)
    ]
    room = world.add_room(name="exploration_zone", footprint=exploration_coords, color=[0.8, 0.8, 0.8])

    # Add base stationf
    base = world.add_location(
        name="base_station0",  # Explicit name
        category="base_station",
        parent="exploration_zone",
        pose=Pose(x=0.0, y=0.0, yaw=0.0)
    )

    location_list=[[-0.5,0.5,-0.5,0.5]] #Initially only base 

    add_fossils_to_room(
        world,
        room,
        count=n_fossils,
        random_seed=random_seed + 1 * 42
    )
    add_locations_to_room(
        world,
        room,
        "rock",
        count=n_rocks,
        random_seed=random_seed + 2 * 42
    )
    add_locations_to_room(
        world,
        room,
        "bush",
        count=n_rocks,
        random_seed=random_seed + 3 * 42
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
