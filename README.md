# Fossil collection robots 
This project uses ROS2 and pyrobosim to simulate two autonomous robots that work together to discover fossils in an area and then collect and deposit at a base. 

## Libraries and tools used
- Pyrobosim
- ROS2

## Features 
- Using ROS2 Topics, Services and Actions for communication between robots
- Simulation of sensors for object detection
- Robust path planning and navigation for both robots
- Environment that can be customised and randomly generated
- Battery simulation and management

## Running the simulation
1. Make sure pyrobosim is installed locally follofing [these steps](https://pyrobosim.readthedocs.io/en/latest/setup.html#local-setup)
1. Clone the repository into directory
   ``` pyrobosim_ws/src/pyrobosim/pyrobosim_ros/examples```
2. Run the file ```source source_pyrobosim.bash``` from directory ```pyrobosim_ws/src/pyrobosim/setup```
3. Run the file ```fossil_demo.py```
4. The simulation will then start logging information to the terminal and load the GUI

## Changing the environment 
Variables for the environment generation can be customised. 

At line 1191 of fossil_demo the following can be arguments to the create world method:
- width: Width of the environment
- height: Height of the environment
- n_fossils: Number of fossils to be generated
- n_rocks: Number of rocks to be generated
- n_bushes: Number of bushes to be generated
- n_charegrs: Number of charging stations to be genreated
- random_seed: Seed to allow for reproducability of environment. Change to any other number or random.randint() to generate a new environment
