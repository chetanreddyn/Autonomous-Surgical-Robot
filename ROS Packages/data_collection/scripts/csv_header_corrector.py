import csv
import numpy as np
import os

class CSV_Corrector:
    def __init__(self, config_dict):
        self.arm_names = config_dict["arm_names"]
        self.folder_path = config_dict["folder_path"]
        self.demo_names = config_dict["demo_names"]
        self.csv_name = config_dict["csv_name"]

        self.correct_csv_columns = self.generate_correct_csv_columns()

    def generate_correct_csv_columns(self):
        columns = ["Epoch Time","Time (Seconds)","Frame Number"]

        for arm_name in self.arm_names:
            for k in range(1,4):
                for l in range(1,4):
                    columns.append(f"{arm_name}_orientation_matrix_[{k},{l}]")

            for i in range(1,7):
                columns.append(f"{arm_name}_joint_{i}")
            columns.append(f"{arm_name}_jaw")

        for camera_name in ["camera_right", "camera_left"]:
            columns.append(f"{camera_name}_image_path")

        return columns

    def correct_csv_header(self, demo_name):
        csv_file_path = os.path.join(self.folder_path, demo_name, self.csv_name)

        # Read the CSV file into memory
        with open(csv_file_path, mode='r', newline='') as file:
            reader = csv.reader(file)
            data = list(reader)

        # Modify the first row (header) to the correct columns
        assert len(data[0]) == len(self.correct_csv_columns)
        data[0] = self.correct_csv_columns
    
        # Write the modified data back into the same CSV file
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data)

    def correct_csv_files(self):
        for demo_name in self.demo_names:
            self.correct_csv_header(demo_name)



demo_names = ["Demo25"]

csv_corrector_config_dict = {
        "arm_names": ["PSM1", "PSM2"],
        "folder_path": "/Users/chetan/Desktop/CHARM_IPRL_Project/Autonomous-Surgical-Robot-Data/Initial_Experiments",
        "demo_names": demo_names,
        "csv_name": "data.csv"
    }

csv_corrector = CSV_Corrector(csv_corrector_config_dict)
print("The Script will replace the headers of {} files with {}".format(len(demo_names),csv_corrector.correct_csv_columns))
reply = input("Are you sure you want to permanently change? (Y/N)")

if reply=="Y" or "y":
    csv_corrector.correct_csv_files()
    print("Successfully Changed")
else:
    print("Abort")