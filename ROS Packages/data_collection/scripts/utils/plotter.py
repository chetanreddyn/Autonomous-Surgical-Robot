import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
from plotly.subplots import make_subplots
import os

def get_arms():
    return {
        'PSM1': [f'PSM1_joint_{i+1}' for i in range(6)] + ['PSM1_jaw'],
        'PSM2': [f'PSM2_joint_{i+1}' for i in range(6)] + ['PSM2_jaw'],
        # 'PSM3': [f'PSM3_joint_{i+1}' for i in range(6)] + ['PSM3_jaw'],
    }

def load_csv(csv_path):
    if csv_path is None or not os.path.isfile(csv_path):
        print(f"File not found: {csv_path}, skipping.")
        return None
    df = pd.read_csv(csv_path)
    if 'Epoch Time' in df.columns:
        df['Epoch Time'] = pd.to_datetime(df['Epoch Time'])
    return df

def align_dfs(df1, df2, frequency=30):
    t1 = df1["Epoch Time"].iloc[0]
    t2 = df2["Epoch Time"].iloc[0]
    offset = (t1 - t2)
    offset_idx = int(abs(offset.total_seconds() * frequency)) + 1
    print(offset.total_seconds(), offset_idx)
    if offset.total_seconds() <= 0:
        df1 = df1.iloc[offset_idx:]
        df2 = df2.iloc[:-offset_idx]
    else:
        df1 = df1.iloc[:-offset_idx]
        df2 = df2.iloc[offset_idx:]
    if "Frame Number" in df1.columns:
        df1 = df1.copy()
        df1["Frame Number"] = range(len(df1))
    if "Frame Number" in df2.columns:
        df2 = df2.copy()
        df2["Frame Number"] = range(len(df2))
    print(df1["Frame Number"].head(), df2["Frame Number"].head())
    return df1, df2

def plot_arm_joints(data_csv=None, actions_csv=None, x_axis="Frame Number", title=""):
    """
    x_axis: "Frame Number" or "Epoch Time"
    """
    pio.renderers.default = "browser"
    df_data = load_csv(data_csv)
    df_actions = load_csv(actions_csv)
    arms = get_arms()

    # Only align if both are present
    if df_data is not None and df_actions is not None:
        df_data, df_actions = align_dfs(df_data, df_actions)

    color_palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]

    fig = make_subplots(
        rows=len(arms), cols=1, shared_xaxes=False,
        subplot_titles=[f"{arm} Joint Values" for arm in arms.keys()]
    )

    # Plot data.csv if present
    if df_data is not None and x_axis in df_data.columns:
        for idx, (arm, joints) in enumerate(arms.items(), start=1):
            for j, joint in enumerate(joints):
                if joint in df_data.columns:
                    color = color_palette[j % len(color_palette)]
                    legend_name = f"{joint} (data)"
                    fig.add_trace(
                        go.Scatter(
                            x=df_data[x_axis],
                            y=df_data[joint],
                            mode='lines',
                            name=legend_name,
                            legendgroup=legend_name,
                            showlegend=True,
                            line=dict(dash='solid', color=color)
                        ),
                        row=idx, col=1
                    )

    # Plot actions_csv if present
    if df_actions is not None and x_axis in df_actions.columns:
        for idx, (arm, joints) in enumerate(arms.items(), start=1):
            for j, joint in enumerate(joints):
                if joint in df_actions.columns:
                    color = color_palette[j % len(color_palette)]
                    legend_name = f"{joint} (action)"
                    fig.add_trace(
                        go.Scatter(
                            x=df_actions[x_axis],
                            y=df_actions[joint],
                            mode='lines',
                            name=legend_name,
                            legendgroup=legend_name,
                            showlegend=True,
                            line=dict(dash='dash', color=color)
                        ),
                        row=idx, col=1
                    )
    for idx in range(1, len(arms) + 1):
        fig.update_yaxes(title_text="Joint Value", row=idx, col=1)
        fig.update_xaxes(title_text=x_axis, showticklabels=True, row=idx, col=1)

    fig.update_layout(
        height=600 * len(arms),
        width=1000,
        title_text=title,
        legend_title="Joint"
    )
    pio.show(fig)

def plot_arm_joints_and_xyz(data_csv=None, actions_csv=None, x_axis="Frame Number", title="", logging_folder=None):
    pio.renderers.default = "browser"
    df_data = load_csv(data_csv)
    df_actions = load_csv(actions_csv)
    arms = list(get_arms().keys())
    color_palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]
    xyz_labels = ['x', 'y', 'z']
    xyz_palette = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    # Only align if both are present
    if df_data is not None and df_actions is not None:
        df_data, df_actions = align_dfs(df_data, df_actions)

    fig = make_subplots(
        rows=2, cols=2, shared_xaxes=False,
        subplot_titles=[
            f"{arms[0]} Joint Values", f"{arms[0]} Cartesian XYZ",
            f"{arms[1]} Joint Values", f"{arms[1]} Cartesian XYZ"
        ]
    )

    for arm_idx, arm in enumerate(arms):
        joints = get_arms()[arm]
        row = arm_idx + 1

        # --- JOINTS ---
        for j, joint in enumerate(joints):
            # Data
            if df_data is not None and joint in df_data.columns and x_axis in df_data.columns:
                color = color_palette[j % len(color_palette)]
                legend_name = f"{joint} (data)"
                fig.add_trace(
                    go.Scatter(
                        x=df_data[x_axis],
                        y=df_data[joint],
                        mode='lines',
                        name=legend_name,
                        showlegend=True,
                        line=dict(dash='solid', color=color)
                    ),
                    row=row, col=1
                )
            # Actions
            if df_actions is not None and joint in df_actions.columns and x_axis in df_actions.columns:
                color = color_palette[j % len(color_palette)]
                legend_name = f"{joint} (action)"
                fig.add_trace(
                    go.Scatter(
                        x=df_actions[x_axis],
                        y=df_actions[joint],
                        mode='lines',
                        name=legend_name,
                        showlegend=True,
                        line=dict(dash='dash', color=color)
                    ),
                    row=row, col=1
                )

        # --- XYZ ---
        for j, label in enumerate(xyz_labels):
            col = f"{arm}_ee_{label}"
            if df_data is not None and col in df_data.columns and x_axis in df_data.columns:
                color = xyz_palette[j % len(xyz_palette)]
                legend_name = f"{arm}_ee_{label} (data)"
                fig.add_trace(
                    go.Scatter(
                        x=df_data[x_axis],
                        y=df_data[col],
                        mode='lines',
                        name=legend_name,
                        showlegend=True,
                        line=dict(dash='solid', color=color)
                    ),
                    row=row, col=2
                )

    for row in range(1, 3):
        fig.update_yaxes(title_text="Joint Value", row=row, col=1)
        fig.update_xaxes(title_text=x_axis, showticklabels=True, row=row, col=1)
        fig.update_yaxes(title_text="XYZ", row=row, col=2)
        fig.update_xaxes(title_text=x_axis, showticklabels=True, row=row, col=2)

    fig.update_layout(
        height=1200,
        width=1600,
        title_text=title,
        legend_title="Legend",
        legend=dict(
            x=1.02,
            y=1,
            xanchor='left',
            yanchor='top',
            orientation='v',
            font=dict(size=12),
            bordercolor="Black",
            borderwidth=1
        ),
        margin=dict(r=350)  # Add space on the right for the legend
    )
    # Save as HTML in the logging folder
    if logging_folder is not None:
        html_path = os.path.join(logging_folder, "plot.html")
        fig.write_html(html_path)
        print(f"Plot saved to {html_path}")
    pio.show(fig)

if __name__ == "__main__":
    root_folder = "/home/stanford/catkin_ws/src/Autonomous-Surgical-Robot-Data/Rollouts"
    exp_type = "Rollouts Autonomous"
    # exp_type = "Collaborative Expert Two Handed Object Transfer"
    demo_name = "Test3"
    LOGGING_FOLDER = os.path.join(root_folder, exp_type, demo_name)
    data_csv = os.path.join(LOGGING_FOLDER, "data.csv")
    actions_csv = os.path.join(LOGGING_FOLDER, "rollout_actions.csv")
    # plot_arm_joints(data_csv=data_csv, actions_csv=actions_csv, x_axis="Frame Number", title=f"{exp_type}/{demo_name}")
    # plot_arm_joints(data_csv=data_csv, actions_csv=actions_csv, x_axis="Epoch Time")
    plot_arm_joints_and_xyz(data_csv=data_csv, actions_csv=actions_csv, 
                            x_axis="Frame Number", 
                            title=f"{exp_type}/{demo_name}", 
                            logging_folder=LOGGING_FOLDER)