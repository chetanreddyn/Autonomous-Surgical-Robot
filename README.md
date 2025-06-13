# Autonomous-Surgical-Robot
Project with CHARM and IPRL on the Da Vinci Surgical Robot. The project aims to automate one of the arms as an assistant (using imitation learning) to collaborate with the surgeon.

## Table of Contents
- [Overview](#overview)
- [File Structure](#File-Structure)
- [Teleoperation](#Teleoperation)
- [Data Collection](#Data-Collection)
- [Rollout](#Rollout)
- [Project Structure](#project-structure)
- [Contact](#contact)

## Overview
This github repository has the ROS packages for three functionalities - teleoperation, data collection and model rollout. The instructions have been described in detail below. It is assumed that the commands are executed from the SRC (Stanford Robotics Centre) computer in the Medical Bay connected to the Da Vinci Robot (Si Model). 

## File Structure
```
Autonomous-Surgical-Robot/
├── README.md
├── Models/
│   ├── Adapt-ACT/
│   ├── mlcore/
├── ROS Packages/
    ├── data_collection/
    │   ├── CMakeLists.txt
    │   ├── package.xml
    │   ├── launch/
    │   ├── scripts/
    │   └── ...
    ├── rollout/
    │   ├── CMakeLists.txt
    │   ├── package.xml
    │   ├── launch/
    │   ├── scripts/
    │   └── ...
    ├── teleop/
        ├── CMakeLists.txt
        ├── package.xml
        ├── launch/
        ├── scripts/
        └── ...

```
- **Models/**: Contains ACT model and core utilities.
- **ROS Packages/**: Contains ROS packages for data collection, teleoperation, and rollouts.
## Teleoperation
The `teleop` ROS package has the scripts/nodes to launch the robot and perform teleoperation using the MTMs (Master Tele Manipulator), Phantom Omni and Keyboard.
#### Step 1: Launch the dvrk console 
```bash
roslaunch teleop arms_real.launch
```
This launch files will run the `dvrk_console_json` node from the dVRK package and other static coordinate transformations that are required for the teleoperation.

#### Step 2: Launch the vision pipeline
```bash
roslaunch teleop vision_cart.launch console:=true
```
> Set `console:=false` to suppress surgeon console GUI windows.
This launch file will run the nodes required to process the video stream from the camera and publish them into ROS topics. 

#### Step 3: Launching the Phantom Omni device
```bash
roslaunch teleop phantom_real.launch
```
The `phantom_real.launch` file contains the nodes required to simulate the digital twin and publish the pose of the phantom omni's stylus with respect to it's base

#### Step 4: Run the script to launch phantom omni teleoperation
```bash
rosrun teleop phantom_teleop.py -a PSM1
```
> The -a flag is used to specify the arm to teleoperate

## Data Collection 
The `data_collection` ROS package has the scripts/nodes to record the data during an experiment. It also has the scripts to initialize and replay experiments and also save and check the initial poses of the SUJs and tool tips. Follow these steps to log an experimental run (after completing the steps above):
#### Step 1: Run the launch file that loads and publishes the saved initial pose 
```bash
roslaunch data_collection data_collection_setup.launch
```
> (This launch file can also be edited to include all the steps in Teleoperation if you want everything in a single place but not recommended)

#### Step 2: Check the Initialise poses to ensure the SUJs haven't been moved (done only once per session)
```bash
rosrun data_collection check_initial_pose.py
```
The values corresponding to PSM1_base, PSM2_base, PSM3_base and ECM_base must be less than 0.01. Use the flag --type joint_angles to specify the errors in the joints.

#### Step 2: Specify the Logging Folder (done only once per session)
Open the file `/data_collection/scripts/csv_generator.py` and specify the `LOGGING_FOLDER`. This needs to be done only once per session unless different kinds of experiments are done in the same sessions.

#### Step 3: Initialise the Experiment
```bash
rosrun data_collection initialize_exp.py
```

#### Step 3: Run the csv_generator script to log an experiment
```bash
rosrun data_collection csv_generator.py --loginfo -T 20 -d Demo1
```
Specify the demonstration name in the -d flag and the -T flag is used to specify the duration of an experiment after which the logging automatically stops (default is 15 seconds). In the above command, the experimental run will be saved in LOGGING_FOLDER/Demo1 and the duration is 20 seconds.

#### 
## Rollout 
Steps
1. roslaunch rollout

## Contact




**Teleoperation Demo**


https://github.com/user-attachments/assets/abd87d2c-8bc9-43d4-abea-3149a9075a11

