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
Specify the path of the json inside data_collection_setup.launch in the value field
```
      <param 
      name="initial_pose_json" 
      value="/home/stanford/catkin_ws/src/Autonomous-Surgical-Robot/ROS Packages/data_collection/utils_config/initial_pose_with_suj_3_arms.json" # SPECIFY HERE
      type="str"/>
```

