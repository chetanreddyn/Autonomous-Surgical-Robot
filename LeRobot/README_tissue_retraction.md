<!--
Open-H Embodiment Dataset README Template (v1.0)
Please fill out this template and include it in the ./metadata directory of your LeRobot dataset.
This file helps others understand the context and details of your contribution.
-->

# Tissue Retraction - README

---

## 📋 At a Glance

*Teleoperated demonstrations of the da Vinci Si robot performing tissue retraction of 2-3 layers (of silicone phantom)*

<!--
**Example:** *Teleoperated demonstrations of a da Vinci robot performing needle passing on a silicone phantom.*
-->

---

## 📖 Dataset Description

<!--
*Briefly describe the purpose and content of this dataset. What key skills or scenarios does it demonstrate?*

**Example:** *This dataset contains 2,500 trajectories of expert surgeons using the dVRK to perform surgical suturing tasks. It includes successful trials, failures, and recovery attempts to provide a robust dataset for training imitation learning policies.*
-->
The dataset comprises 700 dVRK trajectories of tissue retraction performed on a table top phantom, including successful trials, failures, and recovery attempts. It provides synchronized cartesian, joint and video data for training and evaluating robot learning policies.


| | |
| :--- | :--- |
| **Total Trajectories** | `700` |
| **Total Hours** | `2.9` |
| **Data Type** | `[ ] Clinical` `[ ] Ex-Vivo` `[x] Table-Top Phantom` `[ ] Digital Simulation` `[ ] Physical Simulation` `[ ] Other (If checked, update "Other")` |
| **License** | CC BY 4.0 |
| **Version** | `[e.g., 1.0]` |

---

## 🎯 Tasks & Domain

### Domain

*Select the primary domain for this dataset.*

- [x] **Surgical Robotics**
- [ ] **Ultrasound Robotics**
- [ ] **Other Healthcare Robotics** (Please specify: `[Your Domain]`)

### Demonstrated Skills

*List the primary skills or procedures demonstrated in this dataset.*
- Choosing Right Point to Start Retracting
- Second Arm Assisting 
- Retracting Multiple Layers 

<!--
***Example:***
- Needle-passing
- Suture-tying
- ...
-->
---

## 🔬 Collection Procedure

### Collection Method

*How was the data collected?*

- [x] **Human Teleoperation**
- [ ] **Programmatic/State-Machine**
- [ ] **AI Policy / Autonomous**
- [ ] **Other** (Please specify: `[Your Method]`)

### Operator Details

| | Description |
| :--- | :--- |
| **Operator Count** | `1` |
| **Operator Skill Level** | `[ ] Expert (e.g., Surgeon, Sonographer)` <br> `[x] Intermediate (e.g., Trained Researcher)` <br> `[ ] Novice (e.g., ML Researcher with minimal experience)` <br> `[ ] N/A` |
| **Collection Period** | From `[2025-10-01]` to `[2025-01-15]` |

### Recovery Demonstrations

*Does this dataset include examples of recovering from failure?*

- [ ] **Yes**
- [x] **No**

<!--
*Example: For 250 demonstrations, demonstrations are initialized from a failed needle grasp position, the operator re-orients the robotic grippers and attempts to grasp the needle again from a different angle.*
-->
---

## 💡 Diversity Dimensions

*Check all dimensions that were intentionally varied during data collection.*

- [x] **Camera Position / Angle**
- [x] **Lighting Conditions**
- [x] **Target Object** (e.g., different phantom models, suture types)
- [x] **Spatial Layout** (e.g., placing the target suture needle in various locations)
- [ ] **Robot Embodiment** (if multiple robots were used)
- [ ] **Task Execution** (e.g., different techniques for the same task)
- [ ] **Background / Scene**
- [ ] **Other** (Please specify: `[Your Dimension]`)

*If you checked any of the above please briefly elaborate below.*
The camera configuration was adjusted every 50–100 demonstrations by varying the setup height by ±2 cm. In addition, the needle type and phantom base were changed periodically. Lighting conditions were varied between 60% and 100%. Some demonstrations had 2 layers and some had 3 layers.
<!--

**Example:** We adjusted the room camera perspective every 100 demonstrations. The camera angle was varied by panning up and down by +/- 10 degrees, as well as manually adjusting the height of the camera mount by +/- 2 cm. Additionally, we varied the needle used by swapping out various curvatures, including 1/4, 3/8, 1/2, and 5/8.
-->
---

## 🛠️ Equipment & Setup

### Robotic Platform(s)

*List the primary robot(s) used.*

- **Robot 1:** `dVRK (da Vinci Research Kit)`

### Sensors & Cameras

*List the sensors and cameras used. Specify model names where possible. 

| Type | Model/Details |
| :--- | :--- |
| **Primary Camera** | `Endoscopic Camera, 1920x1080 @ 30fps with both left and right video feed` |
| **Joint/Position Encoders** | |

---

## 🎯 Action & State Space Representation

*Describe how actions and robot states are represented in your dataset. This is crucial for understanding data compatibility and enabling effective policy learning.*

### Action Space Representation

**Primary Action Representation:**
- [x] **Absolute Cartesian** (position/orientation relative to camera frame (ECM))
- [ ] **Relative Cartesian** (delta position/orientation from current pose)
- [ ] **Joint Space** (direct joint angle commands)
- [ ] **Other** (Please specify: `[Your Representation]`)

**Orientation Representation:**
- [ ] **Quaternions** (x, y, z, w)
- [x] **Euler Angles** (roll, pitch, yaw)
- [ ] **Axis-Angle** (rotation vector)
- [ ] **Rotation Matrix** (3x3 matrix)
- [ ] **Other** (Please specify: `[Your Representation]`)

**Reference Frame:**
- [ ] **Robot Base Frame**
- [ ] **Tool/End-Effector Frame**
- [ ] **World/Global Frame**
- [x] **Camera Frame**
- [ ] **Other** (Please specify: `[Your Frame]`)

**Action Dimensions:**
*List the action space dimensions and their meanings.*
```
action: [PSM{i}_jaw, PSM{i}_ee_x, PSM{i}_ee_y, PSM{i}_ee_z, PSM{i}_ee_roll, PSM{i}_ee_pitch, PSM{i}_ee_yaw]
- PSM{i} represents the arm (Could be PSM1 or PSM2)
- PSM{i}_jaw: Jaw angle in radians
- PSM{i}_ee_x, PSM{i}_ee_y, PSM{i}_ee_z: Absolute position in camera frame (ECM frame)
- PSM{i}_ee_roll, PSM{i}_ee_pitch, PSM{i}_ee_yaw: Absolute Orientation as Euler Angles
```
<!--
**Example:**
```
action: [x, y, z, qx, qy, qz, qw, gripper]
- x, y, z: Absolute position in robot base frame (meters)
- qx, qy, qz, qw: Absolute orientation as quaternion
- gripper: Gripper opening angle (radians)
```
-->
### State Space Representation

**State Information Included:**
- [x] **Joint Positions** (all articulated joints)
- [ ] **Joint Velocities**
- [x] **End-Effector Pose** (Cartesian position/orientation)
- [ ] **Force/Torque Readings**
- [x] **Gripper State** (position, force, etc.)
- [ ] **Other** (Please specify: `[Your State Info]`)

**State Dimensions:**
*List the state space dimensions and their meanings.*

**Example:**
```
observation.state: [PSM{i}_joint_1, PSM{i}_joint_2, PSM{i}_joint_3, PSM{i}_joint_4, PSM{i}_joint_5, PSM{i}_joint_6, PSM{i}_jaw, PSM{i}_ee_x, PSM{i}_ee_y, PSM{i}_ee_z, PSM{i}_ee_roll, PSM{i}_ee_pitch, PSM{i}_ee_yaw]
- PSM{i} represents the arm (Could be PSM1 or PSM2)
- PSM{i}_joint_1 to PSM{i}_joint_6: Absolute joint positions for the 7-DOF arm (radians)
- PSM{i}_jaw: Jaw angle of the gripper (radians)
- PSM{i}_ee_x, PSM{i}_ee_y, PSM{i}_ee_z: End-effector absolute position in camera (ECM) frame (meters)
- PSM{i}_ee_roll, PSM{i}_ee_pitch, PSM{i}_ee_yaw: End-effector absolute orientation as Euler angles (radians)

```

---

## ⏱️ Data Synchronization Approach

*Describe how you achieved proper data synchronization across different sensors, cameras, and robotic systems during data collection. This is crucial for ensuring temporal alignment of all modalities in your dataset.*

We use the ApproximateTimeSynchronizer [[Link]](https://wiki.ros.org/message_filters/ApproximateTime) from the ROS message_filters package to synchronize all data streams. The queue_size parameter controls the number of incoming messages buffered for each topic, while the slop parameter specifies the maximum allowable time difference between messages for them to be considered synchronized. This approach aligns messages based on their timestamps within a defined tolerance rather than requiring exact matches. 

Data is recorded at 30 Hz, with the camera feed acting as the bottleneck. During data collection, we monitor the inter-frame time difference and ensure it remains close to 33 ms, resulting in approximately 450 frames per 15-second episode. In rare cases, message delays lead to significantly fewer frames (fewer than 435); such episodes are discarded and re-recorded.

<!--
**Example:** *We collect joint kinematics from our Franka Research 3 and RGB-D frames from Intel RealSense D435 cameras, all running in ROS 2 Galactic on the same workstation clocked with ROS Time. Both drivers stamp their outgoing messages’ header.stamp fields with the shared system clock, and we record /joint_states, /camera/*/image_raw, and /camera/*/camera_info in a single rosbag2 session. During export to LeRobot, each data point’s ROS header.stamp is written verbatim into the timestamp attribute. Offline checks show inter-sensor skew stays below ±2 ms across a 2-minute capture.*
-->
---

## 👥 Attribution & Contact

*Please provide attribution for the dataset creators and a point of contact.*

| | |
| :--- | :--- |
| **Dataset Lead** | `Chetan Reddy Narayanaswamy` |
| **Institution** | `Stanford University` |
| **Contact Email** | `chetanrn@stanford.edu` |
| **Personal Website** | [`https://chetanreddyn.github.io/`](https://chetanreddyn.github.io/)|
