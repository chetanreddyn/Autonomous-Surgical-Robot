import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import dash
from dash import dcc, html


class Visualizer:

    def __init__(self,config_dict,df):
        self.df = df.copy()
        self.stanford_computer_path = config_dict["stanford_computer_path"]
        self.local_computer_path = config_dict["local_computer_path"]

        self.arm1_joint_names = config_dict["arm1_joints"]
        self.arm2_joint_names = config_dict["arm2_joints"]

        self.preprocess_data()

    def preprocess_data(self):

        self.df["Epoch Time"] = pd.to_datetime(self.df["Epoch Time"])

        self.df["camera_left_image_path_local"] = df['camera_left_image_path'].apply(lambda x: self.local_computer_path + x[len(self.stanford_computer_path):])
        self.df["camera_right_image_path_local"] = df['camera_right_image_path'].apply(lambda x: self.local_computer_path + x[len(self.stanford_computer_path):])

        self.df = self.df.set_index("Frame Number", inplace=False)


    def plot_camera_images(self, frame_number):
        pass

    def plot_arm_joints(self,joint_names):
        fig = go.Figure()

        for i,col in enumerate(joint_names):
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col))


        fig.update_layout(
            width=600,   # Set figure width (in pixels)
            height=400,   # Set figure height (in pixels)
            margin=dict(t=50, b=0, l=0, r=0)  # Adjust margins if necessary
        )
        return fig

    def plot_all_figs(self):
        fig1 = self.plot_arm_joints(self.arm1_joint_names)
        fig2 = self.plot_arm_joints(self.arm2_joint_names)

        fig1.show()
        fig2.show()
        


csv_file = "/Users/chetan/Desktop/CHARM_IPRL_Project/Autonomous-Surgical-Robot-Data/Initial_Experiments/Sample2/data.csv"
df = pd.read_csv(csv_file)

config_dict = {
    "stanford_computer_path": "/home/stanford/catkin_ws/src/Autonomous-Surgical-Robot-Data/Initial Samples/",
    "local_computer_path": "/home/stanford/catkin_ws/src/Autonomous-Surgical-Robot-Data/Initial Samples/",
    'arm1_joints':['PSM1_joint_1',
       'PSM1_joint_2', 'PSM1_joint_3', 'PSM1_joint_4', 'PSM1_joint_5',
       'PSM1_joint_6', 'PSM1_jaw'],
    'arm2_joints':['PSM2_joint_1', 'PSM2_joint_2', 'PSM2_joint_3', 'PSM2_joint_4',
       'PSM2_joint_5', 'PSM2_joint_6', 'PSM2_jaw']

}

if __name__ == "__main__":
    visualiser = Visualizer(config_dict,df)
    visualiser.plot_all_figs()