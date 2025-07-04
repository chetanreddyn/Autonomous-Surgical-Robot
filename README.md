# Autonomous-Surgical-Robot
Project with CHARM and IPRL on the Da Vinci Surgical Robot. The project aims to automate one of the arms as an assistant (using imitation learning) to collaborate with the surgeon.

## Table of Contents
- [Overview](#overview)
- [File Structure](#File-Structure)
- [Teleoperation](#Teleoperation)
- [Data Collection](#Data-Collection)
- [Rollout](#Rollout)
- [Results](#Results)
- [Known Issues](#Known-Issues)
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

#### Step 1: Launch the dvrk console (New Terminal)
```bash
roslaunch teleop arms_real.launch
```
You should see two windows appearing one after another. The first one is an RViz window and the second one is the console (the GUI to control the robot). The `arms_real.launch` launch files will run the `dvrk_console_json` node from the dVRK package and other static coordinate transformations that are required for the teleoperation. 

#### Step 2: Click the Power On button followed by the Home button in the console
Clicking the `Power On` button turns the LED on the arms to blue. Clicking the `Home` button turns them green and you will notice the MTMs moving towards their home position. Wait for all the arms to turn green, sometimes it takes longer for SUJ to turn green. 

If you want to switch on the MTM-PSM teleoperation connection at this point, click on the checkbox under `Tele operation` and it should go from **disabled** (highlighed in red) to **enabled** (highlighted in green) and the MTMs will start aligning their orientation with that of the PSMs. 

#### Step 3: Launch the vision pipeline (New Terminal)
```bash
roslaunch teleop vision_cart.launch console:=true
```
The `vision_cart.launch` file will run the nodes required to process the video stream from the camera and publish them into ROS topics. Two windows will be displayed corresponding to the left and right camera streams. There will be another window in RViz corresponding to the left camera stream. 

Maximise the windows and push them into the surgeon console by pressing `Win+Shift+Left Arrow`, press the `Left Arrow` twice for the `camera_left` window and once for the `camera_right` window.

(Set `console:=false` to suppress surgeon console GUI windows)
#### Step 4: Launching the Phantom Omni device (New Terminal) 
```bash
roslaunch teleop phantom_real.launch
```
The `phantom_real.launch` file contains the nodes required to simulate the digital twin and publish the pose of the phantom omni's stylus with respect to it's base. You should be a simulated model of the phantom omni in RViz.

Sometimes, this command can throw permission errors (when the phantom omni is re-plugged or the computer is restarted). Run the following command when that happens: 
```
sudo chmod 777 /dev/ttyACM0
```
and re launch the `phantom_real.launch` using the command above

#### Step 5: Run the script to launch phantom omni teleoperation (New Terminal)
```bash
rosrun teleop phantom_teleop.py -a PSM3 # Specify the appropriate PSM
```
You should see the message: `Detected Phantom Pose! Hold the Pen in Position, Hold the Grey Button Once to Start`

Some pointers:
- The -a flag is used to specify the arm to teleoperate.
- Controlling the Jaw: The white button is used to open and close the jaw.
- When using the phantom omni, make the initial position of the stylus roughly align with the PSM's tool tip from the video stream
- Always switch off the connection (by pressing the Grey Button) before placing the stylus into the home position (Ink Well) of the Phantom Omni.
- The `phantom_teleop` script performs the required transformation to ensure the pose of the PSM tool tip with respect to the camera matches that of the stylus with respect to the eyes. It also has the logic to process the button clicks into a continuous jaw angle. 

## Data Collection 
The `data_collection` ROS package has the scripts/nodes to record the data during an experiment, initialize and replay experiments and also save and check the initial poses of the SUJs and tool tips. Follow these steps to log an experimental run (the step 1 commands below are explained in detail above):

#### Step 1: Teleoperation Steps (in different terminals, skip this step if you finished it above)
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

#### Step 2: Run the launch file that loads and publishes the saved initial pose (New Terminal)
```bash
roslaunch data_collection data_collection_setup.launch
```
(This launch file can also be edited to include all the steps in Teleoperation (Step 1) if you want everything in a single place but not recommended since you have to relaunch the robot everytime there is a small issue)

#### Step 3: Check the initial poses to ensure the SUJs haven't been moved (done only once per session) (New Terminal)
```bash
rosrun data_collection check_initial_pose.py
```
The values corresponding to `PSM1_base`, `PSM2_base`, `PSM3_base` and `ECM_base` must be less than 0.01. Use the flag --type joint_angles to display the errors in the joints. In a circumstance where the errors of any of the arm base is not less than 0.01, the SUJs have to be manually moved to the saved initial pose in 3D space, a couple of tools were developed to help with this. The details are under **"Correcting the Initial Pose"** inside the `data_collection` package [Link](https://github.com/chetanreddyn/Autonomous-Surgical-Robot/tree/main/ROS%20Packages/data_collection)

#### Step 4: Specify the Logging Folder and the meta file details(done only once per session)
Open the files `ROS Packages/data_collection/scripts/csv_generator.py` and `ROS Packages/data_collection/scripts/replay_exp.py` and specify the `LOGGING_FOLDER`. This needs to be done only once per session unless different kinds of experiments are done in the same sessions. The files can be opened using the command.
##### Step 4.1
```bash
code '/home/stanford/catkin_ws/src/Autonomous-Surgical-Robot/ROS Packages/data_collection/scripts/csv_generator.py'
code '/home/stanford/catkin_ws/src/Autonomous-Surgical-Robot/ROS Packages/data_collection/scripts/replay_exp.py'
```
##### Step 4.2
Ctrl F the following: `Change logging folder here` in both the files. The `LOGGING_FOLDER` path must be the same in both the scripts (as the `replay_exp.py` reads from the same folder the recording is saved using `csv_generator.py`)

##### Step 4.3



#### Step 5: Data Collection - Sequence of Events to follow
- Switch off MTM teleoperation in the Console
- Step 5.1 (described below) - Initialize the Experiment
- Place the objects
- Switch On MTM teleoperation in the Console
- Step 5.2 (described below) - Run the csv_generator script with the appropriate demo name in -d flag. A prompt will be made. Answering `y` will start the logging (do not enter `y` yet)
- Tell "Mono" to suggest the person on the surgeon console to be ready and confirm
- Answering `y` in the csv_generator script will start the logging after 1 second delay. Therefore, start a countdown from 3 as you press enter.
- Perform the task and let the logging finish
- Switch off the phantom omni (click grey button once) and place it in the inkwell.
- Terminate the logging script
- Repeat

##### Step 5.1: Initialize the Experiment 
```bash
rosrun data_collection initialize_exp.py
```
The initialization should take less than 10 seconds. If it is stuck at a step, terminate and re-run the script.

The initial pose is retrieved from a specific file in [Link](https://github.com/chetanreddyn/Autonomous-Surgical-Robot/tree/main/ROS%20Packages/data_collection/utils_config). If you want to change the initial pose, follow the steps under **Saving Initial Pose** described here [Link](https://github.com/chetanreddyn/Autonomous-Surgical-Robot/tree/main/ROS%20Packages/data_collection)

##### Step 5.2: Run the csv_generator script to log an experiment
```bash
rosrun data_collection csv_generator.py --loginfo -T 20 -d Test
```
Specify the demonstration name in the -d flag and the -T flag is used to specify the duration of an experiment after which the logging automatically stops (default is 15 seconds). In the above command, the experimental run will be saved in LOGGING_FOLDER/Test and the duration is 20 seconds.

#### Step 7: Optional Step (Replaying an Experiment)
##### Step 7.1
Make sure to Switch off the phantom omni (click grey button once) and the the MTM teleoperation.

##### Step 7.2
```bash
rosrun data_collection replay_exp.py -d Test
```
This replays the experimental run saved in LOGGING_FOLDER/Test. The script internally calls `initialize_exp.py` and the experiment is first initialized followed by the replay.

#### Step 8: Postprocessing the Data (Done only once per session in the end)
As noted in the Issues section below, it is not possible to achieve negative jaw angles on the robot using the API functions. Therefore, there is an additional step that needs to be done to clip the negative jaw angles (of the arms controlled by MTMs) to zero in the dataset.

##### Step 8.1: Plot the data to check
Open the `plotter.py` script and specify the right `exp_type` and `demo_name` to visualize. You can search for `specify logging folder and demo number here`
```bash
code '/home/stanford/catkin_ws/src/Autonomous-Surgical-Robot/ROS Packages/data_collection/scripts/utils/plotter.py'
```

Once you have filled it, run the following (in the terminal with the `aloha` environment).
```bash
rosrun data_collection plotter.py
```

This should show the plots in the browser. If the angles are not clipped yet, you should see negative values in the curves corresponding to the jaw angles (run the following steps if the angles are not clipped)

##### Step 8.3: Create a copy of the existing data and save it into a new folder

##### Step 8.2: Run the `jaw_angle_corrector.py` script
Open the jaw corrector script
```bash
code '/home/stanford/catkin_ws/src/Autonomous-Surgical-Robot/ROS Packages/data_collection/scripts/utils/jaw_angle_corrector.py'
```

Search for `Specify logging folder and demo number here`, specify `LOGGING_FOLDER`, `demo_start` (name of the first folder to do the post processing) and `demo_end` (name of the last folder to do the post processing)

Once you have filled it, run the script (in terminal with `aloha` environment)
```bash
rosrun data_collection jaw_angle_corrector.py
```

When you run Step 8.1 again (with the appropriate paths and demo name), the jaw angle curve will now be clipped.


## Rollout 
The `rollout` package is responsible for loading the trained model from a specified folder and using it to control the robot. It also has a logging script to save the generated actions. Run the following steps in a rollout session:

#### Step 1: Teleoperation Steps (in different terminals, skip this step if done earlier)
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
##### Step 2.1: Open the file rollout.launch
```bash
code '/home/stanford/catkin_ws/src/Autonomous-Surgical-Robot/ROS Packages/rollout/launch/rollout.launch'
```
##### Step 2.2: Search "specify paths here" by using Ctrl+F and speci
```
<param name = "TRAIN_DIR" value="INSERT trained model path HERE" type="str"/>
<param name = "LOGGING_FOLDER" value="INSERT logging_folder_path HERE" type="str"/>
```
`TRAIN_DIR` is represents the trained imitation learning model folder
`LOGGING_FOLDER` represents the folder where we want to saved the logs of the rollouts.

#### Step 3: Activate conda environment and make AdaptACT recognisable (Should be done only once in a new terminal)
```bash
conda activate aloha
```
```bash
export PYTHONPATH=$PYTHONPATH:/home/stanford/catkin_ws/src/Autonomous-Surgical-Robot/Models/Adapt-ACT
```
The second command adds the `Adapt-ACT` package to the `PYTHONPATH`, allowing it to be imported in scripts. In other words, `import AdaptACT` will work as expected.

#### Step 5: Specify if Temporal Aggregation is required
Open the `train_cfg.yaml` file inside the `TRAIN_DIR` and specify `ROLLOUT.TEMPORAL_AGG` to `true` or `false`.

#### Step 6: Initialize the Experiment
```bash
rosrun data_collection initialize_exp.py
```
#### Step 7: Launch rollout.launch and specify the arms to be automated
Run the following command in the same terminal where the `aloha` environment is activated
```bash
roslaunch rollout rollout.launch a1:=PSM1 a2:=PSM2 a3:=PSM3 d:=Test
```
The arguments a1,a2 and a3 specify the arms to be automated. Specify `None` if an arm needs to be teleoperated. The `d` argument specifies the name of the specific experimental rollout. The rollout run will be saved in `LOGGING_FOLDER/Test`. To enable temporal aggregation. Go to the `TRAIN_DIR/train_cfg.yaml` and set `ROLLOUT.TEMPORAL_AGG` to true.

##### Fully Automated Case
- Specify `a1:=PSM1 a2:=PSM2 a3:=PSM3`
- Make sure to switch off MTM teleop and Phantom Omni

##### Semi-Automated (PSM3 - Teleoperated with Phantom Omni)
- Specify `a1:=PSM1 a2:=PSM2 a3:=None`
- Make sure to switch off MTM teleop and switch on Phantom Omni

##### Semi-Automated (PSM1, PSM2 - Teleoperated with MTMs)
- Specify `a1:=None a2:=None a3:=PSM3`
- Make sure to switch on MTM teleop and switch off Phantom Omni

#### Step 8: Run process_logged_folder.py to create videos and the visualizations
```bash
rosrun data_collection process_logged_folder.py
```
This should be run in a different terminal immediately after the rollout is completed. This is because the script relies on reading from the `LOGGING_FOLDER/Test` which is published as a ROS parameter. 

The plot.html 


## Results
**Demo of the Two handed and Three Handed Tasks Semi and Fully Automated (Play it with the Music!)**

https://github.com/user-attachments/assets/b8fad08e-2f3c-4888-b93d-a9336c0b85df

## Known Issues

#### Jaw Angle Command
The jaw angle can take negative values when using the MTMs but using the dvrk/crtk API functions, we were unable to pass negative values. The stable function used to control the robot is `move_cp` or `move_jp`. When passing a negative jaw angle, the jaws stop at 0 degrees and does not go below that. We need to explore other API functions like servo_jp or interpolate_jp to be able to pass negative jaw values.

#### Unable to Get Setpoint
Sometimes, we get a runtime warning `unable to get setpoint_cp (/PSM3/setpoint_cp)`, Rerun the four Teleoperation Steps when this happens.

**Teleoperation Demo**

https://github.com/user-attachments/assets/abd87d2c-8bc9-43d4-abea-3149a9075a11

## Contact
**Name:** Chetan Reddy Narayanaswamy<Br>
**Email:** chetanrn@stanford.edu


