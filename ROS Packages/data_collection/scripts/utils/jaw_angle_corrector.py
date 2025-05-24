import pandas as pd
import sys
import os
import time

def correct_jaw_angles(csv_path):
    df = pd.read_csv(csv_path)
    for i in range(1, 4):
        col = f'PSM{i}_jaw'
        if col in df.columns:
            df[col] = df[col].clip(lower=0)
    df.to_csv(csv_path, index=False)
    print(f"Corrected jaw angles in {csv_path}")


if __name__ == "__main__":
    # Specify the path to the CSV file and the output folder
    root_folder = "/home/stanford/catkin_ws/src/Autonomous-Surgical-Robot-Data/"
    exp_type = "Collaborative Object Transfer Corrected"

    LOGGING_FOLDER = os.path.join(root_folder, exp_type)

    if not os.path.exists(LOGGING_FOLDER):
        print("Enter Correct Logging Folder")

    else:
        t0 = time.time()
        t_prev = t0
        for i in range(70,101):
            logging_description = "Demo" + str(i)
            csv_path = os.path.join(LOGGING_FOLDER, logging_description, "data.csv")

            correct_jaw_angles(csv_path)

            t = time.time()
            time_stamp = t-t0
            print(f"Time: {time_stamp:4.3f} | Time Taken: {t-t_prev:4.3} | {logging_description} done")
            t_prev = t
