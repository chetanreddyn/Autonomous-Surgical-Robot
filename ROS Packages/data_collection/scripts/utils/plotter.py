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
    if csv_path is None:
        return None
    df = pd.read_csv(csv_path)
    if 'Epoch Time' in df.columns:
        df['Epoch Time'] = pd.to_datetime(df['Epoch Time'])
    return df
def align_dfs(df1, df2, frequency=30):
    t1 = df1["Epoch Time"].iloc[0]
    t2 = df2["Epoch Time"].iloc[0]

    # Compute the offset
    offset = (t1 - t2)  # In Time
    offset_idx = int(abs(offset.total_seconds() * frequency)) + 1

    print(offset.total_seconds(), offset_idx)
    if offset.total_seconds() <= 0:  # i.e df1 starts late
        df1 = df1.iloc[offset_idx:]
        df2 = df2.iloc[:-offset_idx]
    else:
        df1 = df1.iloc[:-offset_idx]
        df2 = df2.iloc[offset_idx:]

    # Reset Frame Number to start from 0 and increment by 1
    if "Frame Number" in df1.columns:
        df1 = df1.copy()
        df1["Frame Number"] = range(len(df1))
    if "Frame Number" in df2.columns:
        df2 = df2.copy()
        df2["Frame Number"] = range(len(df2))

    print(df1["Frame Number"].head(), df2["Frame Number"].head())
    return df1, df2

def plot_arm_joints(data_csv=None, actions_csv=None, x_axis="Frame Number"):
    """
    x_axis: "Frame Number" or "Epoch Time"
    """
    pio.renderers.default = "browser"
    df_data = load_csv(data_csv)
    df_actions = load_csv(actions_csv)
    arms = get_arms()

    df_data, df_actions = align_dfs(df_data, df_actions)

    # Define a color palette (Plotly default 10 colors)
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
                            legendgroup=legend_name,  # unique for each trace
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
                            legendgroup=legend_name,  # unique for each trace
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
        title_text=f"Joint Values for All Arms ({x_axis} X-Axis)",
        legend_title="Joint"
    )
    pio.show(fig)

if __name__ == "__main__":
    # Set either or both to None as needed
    root_folder = "/home/stanford/catkin_ws/src/Autonomous-Surgical-Robot-Data/"
    exp_type = "Rollouts Autonomous"
    demo_name = "Test"
    LOGGING_FOLDER = os.path.join(root_folder, exp_type, demo_name)
    data_csv = os.path.join(LOGGING_FOLDER, "data.csv")
    actions_csv = os.path.join(LOGGING_FOLDER, "rollout_actions.csv")
    # Choose x_axis: "Frame Number" or "Epoch Time"
    plot_arm_joints(data_csv=data_csv, actions_csv=actions_csv, x_axis="Frame Number")
    # plot_arm_joints(data_csv=data_csv, actions_csv=actions_csv, x_axis="Epoch Time")