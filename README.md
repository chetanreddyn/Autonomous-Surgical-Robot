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
    │   ├── CMakeLists.txt
    │   ├── package.xml
    │   ├── launch/
    │   ├── scripts/
    │   └── ...

```
- **Models/**: Contains ACT model and core utilities.
- **ROS Packages/**: Contains ROS packages for data collection, teleoperation, and rollouts.
## Teleoperation
The `teleop` package has the scripts to launch the robot and perform teleoperation using the MTMs (Master Tele Manipulator), Phantom Omni and Keyboard.
### Step 1: Launch the dvrk console 
```bash
roslaunch teleop arms_real.launch
```

### Step 2: Launch the vision pipeline
```bash
roslaunch teleop vision_cart.launch console:=true
```
> Set `console:=false` to suppress surgeon console GUI windows.

### Step 3: Launching the Phantom Omni device
```bash
roslaunch teleop phantom_real.launch
```

### Step 4: Run the script to launch phantom omni teleoperation
```bash
rosrun teleop phantom_teleop.py -a PSM1
```
> The -a flag is used to specify the arm to teleoperate

## Data Collection 

## Rollout 
Steps
1. roslaunch rollout

## Contact




**Teleoperation Demo**


https://github.com/user-attachments/assets/abd87d2c-8bc9-43d4-abea-3149a9075a11

