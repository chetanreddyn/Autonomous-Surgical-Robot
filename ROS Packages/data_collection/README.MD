# Data Collection Package

The `data_collection` package has a set of scripts required in the data collection pipeline. The most commonly used scripts are explained in the main README file. Some additional essential scripts are explained below.

## Saving Initial Pose
If it is decided to move the SUJs to a different location and record data from that pose going forward, use the `save_initial_pose.py` node to save this pose to a designated location. Follow these steps:

#### Step 1
Specify the path of the json file where the transforms will be saved. Open the script `/data_collection/scripts/save_initial_pose.py` and specify the json file path in `output_file`.

#### Step 2
```bash
rosrun data_collection save_initial_pose.py
```

#### Step 3
Specify the path of the json inside `data_collection_setup.launch` in the `value` field
```
      <param 
      name="initial_pose_json" 
      value="/home/stanford/catkin_ws/src/Autonomous-Surgical-Robot/ROS Packages/data_collection/utils_config/initial_pose_with_suj_3_arms.json" # SPECIFY HERE
      type="str"/>
```
After these 3 steps, the new SUJ poses will be loaded and published which will be used by the `check_initial_pose.py` and `initialize_exp.py` scripts

## Correcting the Initial Pose
If the SUJ base has be moved somehow, we have to correct it manually. Follow these steps:

#### Step 1: Check if the base of any of the arms have been moved
```bash 
rosrun data_collection check_initial_pose.py
```
Check the errors corresponding to PSM1_base, PSM2_base, PSM3_base and ECM_base. All the values must be less than 0.01, if any of the values are too high, proceed to the next step.

#### Step 2: Manual Movement using Visualization in RViz
In RViz, under Frames, enable only PSM1_base and PSM1_base_ref (Assuming the error is in PSM1). You should see only two coordinate frames. Try moving the PSM1_base towards PSM1_base_ref. 

#### Step 3: Manual Movement using SUJ joint value errors
To refine further, we can reduce the errors in each of the SUJs. To do this, run the following command:
```bash
rosrun data_collection check_initial_pose.py --type joint_angles
```
You should see 4 values corresponding to SUJ joints 1 to 4. Try bringing the errors close to zero (less than 0.01) starting from the prismatic joint closest to the Base
