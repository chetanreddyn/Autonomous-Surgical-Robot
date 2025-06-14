# Autonomous-Surgical-Robot
Project with CHARM and IPRL on the Da Vinci Surgical Robot. The project aims to automate one of the arms as an assistant (using imitation learning) to collaborate with the surgeon.

## Table of Contents
- [Overview](#overview)
- [File Structure](#File-Structure)
- [Teleoperation](#Teleoperation)
- [Data Collection](#Data-Collection)
- [Rollout](#Rollout)
- [Results](#Results)
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
#### Step 0: The setup
Ensure the vision cart is switched on along with the light at 100%. Create a new terminator window. Use `Ctrl+Shift+O` to split it horizontally and `Ctrl+Shift+E` to split vertically. We would be needing a lot of terminal windows, so terminator is recommended.

#### Step 1: Launch the dvrk console 
```bash
roslaunch teleop arms_real.launch
```
The `arms_real.launch` launch files will run the `dvrk_console_json` node from the dVRK package and other static coordinate transformations that are required for the teleoperation.

#### Step 2: Click the Power On button followed by the Home button in the console
Clicking the `Power On` button turns the LED on the arms to blue. Clicking the `Home` button turns them green and you will notice the MTMs moving towards their home position. Wait for all the arms to turn green, sometimes it takes longer for SUJ to turn green. 

If you want to switch on the MTM-PSM teleoperation connection at this point, click on the checkbox under `Tele operation` and it should go from **disabled** (highlighed in red) to **enabled** (highlighted in green) and the MTMs will start aligning their orientation with that of the PSMs.

#### Step 3: Launch the vision pipeline
```bash
roslaunch teleop vision_cart.launch console:=true
```
Set `console:=false` to suppress surgeon console GUI windows.
The `vision_cart.launch` file will run the nodes required to process the video stream from the camera and publish them into ROS topics. Two windows will be displayed corresponding to the left and right camera streams. Maximise the windows and push them into the surgeon console by pressing `Ctrl+Shift+Left Arrow`, press the `Left Arrow` twice for the `camera_left` window and once for the `camera_right` window.

#### Step 4: Launching the Phantom Omni device
```bash
roslaunch teleop phantom_real.launch
```
The `phantom_real.launch` file contains the nodes required to simulate the digital twin and publish the pose of the phantom omni's stylus with respect to it's base. Sometimes, this command can throw permission errors (when the phantom omni is re-plugged or the computer is restarted). Run the following command when that happens: 
```
sudo chmod 777 /dev/ttvACM0
```
and re launch the `phantom_real.launch`

#### Step 5: Run the script to launch phantom omni teleoperation
```bash
rosrun teleop phantom_teleop.py -a PSM3 # Specify the appropriate the PSM
```
The -a flag is used to specify the arm to teleoperate. The `phantom_teleop` script performs the required transformation to ensure the pose of the PSM tool tip with respect to the camera matches that of the stylus with respect to the eyes. It also has the logic to process the button clicks into a continuous jaw angle. 

## Data Collection 
The `data_collection` ROS package has the scripts/nodes to record the data during an experiment, initialize and replay experiments and also save and check the initial poses of the SUJs and tool tips. Follow these steps to log an experimental run (the step 1 commands below are explained in detail above):

#### Step 1: Teleoperation Steps (in different terminals)
```bash
roslaunch teleop arms_real.launch
```
```bash
roslaunch teleop vision_cart.launch console:=true
```
```bash
roslaunch teleop phantom_real.launch 
```
```bash
rosrun teleop phantom_teleop.py -a PSM3 # Specify the appropriate PSM 
```

#### Step 2: Run the launch file that loads and publishes the saved initial pose 
```bash
roslaunch data_collection data_collection_setup.launch
```
(This launch file can also be edited to include all the steps in Teleoperation if you want everything in a single place but not recommended since you have to relaunch the robot everytime there is a small issue)

#### Step 3: Check the Initialise poses to ensure the SUJs haven't been moved (done only once per session)
```bash
rosrun data_collection check_initial_pose.py
```
The values corresponding to PSM1_base, PSM2_base, PSM3_base and ECM_base must be less than 0.01. Use the flag --type joint_angles to display the errors in the joints. In a circumstance where the errors of any of the arm base is not less than 0.01, the SUJs have to be manually moved to the saved initial pose in 3D space, a couple of tools were developed to help with this. The details are inside the `data_collection` package.

#### Step 4: Specify the Logging Folder (done only once per session)
Open the files `ROS Packages/data_collection/scripts/csv_generator.py` and `ROS Packages/data_collection/scripts/replay_exp.py` and specify the `LOGGING_FOLDER`. This needs to be done only once per session unless different kinds of experiments are done in the same sessions.

#### Step 5: Initialize the Experiment
```bash
rosrun data_collection initialize_exp.py
```
The initialization should take less than 10 seconds. If it is stuck at a step, terminate and re-run the script.

#### Step 6: Run the csv_generator script to log an experiment
```bash
rosrun data_collection csv_generator.py --loginfo -T 20 -d Test
```
Specify the demonstration name in the -d flag and the -T flag is used to specify the duration of an experiment after which the logging automatically stops (default is 15 seconds). In the above command, the experimental run will be saved in LOGGING_FOLDER/Demo1 and the duration is 20 seconds.

#### Step 7: Replaying an Experiment
```bash
rosrun data_collection replay_exp.py -d Test
```
This replays the experimental run saved in LOGGING_FOLDER/Test. The script internally calls `initialize_exp.py` and the experiment is first initialized followed by the replay.

#### Sequence of Events to follow during data collection (Only Step 5 and Step 6 are repeated in a loop)
- Switch off MTM teleoperation in the Console
- Initialize the Experiment (Step 5)
- Place the objects
- Switch On MTM teleoperation in the Console
- Run the csv_generator script with the appropriate demo name in -d flag (Step 6). A prompt will be made. Answering `y` will start the logging (do not enter `y` yet)
- Tell "Mono" to suggest the person on the surgeon console to be ready and confirm
- Answering `y` in the csv_generator script will start the logging after 1 second delay. Therefore, start a countdown from 3 as you press enter.
- Perform the task and let the logging finish
- Terminate the logging script
- Repeat


## Rollout 
The `rollout` package is responsible for loading the trained model from a specified folder and using it to control the robot. It also has a logging script to save the generated actions. Run the following steps in a rollout session:

#### Step 1: Teleoperation Steps (in different terminals)
```bash
roslaunch teleop arms_real.launch
```
```bash
roslaunch teleop vision_cart.launch console:=true
```
```bash
roslaunch teleop phantom_real.launch 
```
```bash
rosrun teleop phantom_teleop.py -a PSM3 
```

#### Step 2: Specify the trained model folder path and logging folder path in rollout/launch/rollout.launch
```
<param name = "TRAIN_DIR" value="INSERT trained model path HERE" type="str"/>
<param name = "LOGGING_FOLDER" value="INSERT logging_folder_path HERE" type="str"/>
```
The `LOGGING_DESCRIPTION` will be passed in command line in the next step

#### Step 3: Activate conda environment and make AdaptACT recognisable (Should be done only once in a new terminal)
```bash
conda activate aloha
```
```bash
export PYTHONPATH=$PYTHONPATH:/home/stanford/catkin_ws/src/Autonomous-Surgical-Robot/Models/Adapt-ACT
```
The second command adds the `Adapt-ACT` package to the `PYTHONPATH`, allowing it to be imported in scripts. In other words, `import AdaptACT` will work as expected.

#### Step 4: Launch rollout.launch and specify the arms to be automated
```bash
roslaunch rollout rollout.launch a1:=PSM1 a2:=PSM2 a3:=None d:=Test
```
The arguments a1,a2 and a3 specify the arms to be automated. Specify `None` if an arm needs to be teleoperated. The above command will automate the arms PSM1 and PSM2 while PSM3 will not receive commands from the model. The `phantom_teleop.py` script running from a different terminal will control PSM3. The `d` argument specifies the logging description. The rollout run will be saved in `LOGGING_FOLDER/Test`.

#### Step 5: Run process_logged_folder.py to create videos and the visualizations
```bash
rosrun data_collection process_logged_folder.py
```
This should be run in a different terminal immediately after the rollout is completed. This is because the script relies on reading from the `LOGGING_FOLDER/Test` which is published as a ROS parameter. 


## Results
**Demo of the Two handed and Three Handed Tasks Semi and Fully Automated (Play it with the Music!)**

https://github.com/user-attachments/assets/b8fad08e-2f3c-4888-b93d-a9336c0b85df


**Teleoperation Demo**

https://github.com/user-attachments/assets/abd87d2c-8bc9-43d4-abea-3149a9075a11

## Contact
**Name:** Chetan Reddy Narayanaswamy<Br>
**Email:** chetanrn@stanford.edu


