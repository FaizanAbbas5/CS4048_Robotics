# Fossil collection robots 

## Libraries used
- Pyrobosim
- ROS2
## Running the simulation
'''
python3 fossil_demo. py
''' 

## Changing the environment 
Variables for the environment generation can be changed in the fossil_world_generation.py file 

At line 67, the following variables can be changed:
- width: Width of the environment
- height: Height of the environment
- n_fossils: Number of fossils in the environment 
- n_rocks: Number of rocks in the environment
- n_bushes: Number of bushes in the environment
- random_seed: Seed to allow for reproducability of environment. Change to any other number or random.randint() to generate a new environment 
