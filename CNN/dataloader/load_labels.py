import copy
from glob import glob
import os

import pandas as pd

data = pd.read_csv("labels_motors_explanation.csv")
print(data.shape)
data = data.replace("-", 0.0)
data = data.drop([0], axis=0)
data = data.drop(["ids"], axis=1)
cols = data.columns[6:]
data[cols] = data[cols].apply(pd.to_numeric, errors='raise')
data[cols] = data[cols].astype("float")
print(data.shape)

images = copy.deepcopy(data[["ImageView_0", "ImageView_90", "ImageView_180", "ImageView_270", "ImageView_Aufsicht", "ImageView_Untersicht"]])
BB = data[["BB_0", "BB_90", "BB_180", "BB_270", "BB_Aufsicht", "BB_Untersicht"]]
BK = data[["BK_0", "BK_90", "BK_180", "BK_270", "BK_Aufsicht", "BK_Untersicht"]]
BWH1 = data[["BWH1_0", "BWH1_90", "BWH1_180", "BWH1_270", "BWH1_Aufsicht", "BWH1_Untersicht"]]
BWH2 = data[["BWH2_0", "BWH2_90", "BWH2_180", "BWH2_270", "BWH2_Aufsicht", "BWH2_Untersicht"]]
BANR1 = data[["BA Nr1_0", "BA Nr1_90", "BA Nr1_180", "BA Nr1_270", "BA Nr1_Aufsicht", "BA Nr1_Untersicht"]]
BANR2 = data[["BA Nr2_0", "BA Nr2_90", "BA Nr2_180", "BA Nr2_270", "BA Nr2_Aufsicht", "BA Nr2_Untersicht"]]
NRVNR1 = data[["NRV NR1"]]
NRVNR2 = data[["NRV NR2"]]
NRVNR3 = data[["NRV NR3"]]
NRVNR4 = data[["NRV NR4"]]

images["BB"] = BB.max(axis=1)
images["BK"] = BK.max(axis=1)
images["BWH1"] = BWH1.max(axis=1)
images["BWH2"] = BWH2.max(axis=1)
images["BANR1"] = BANR1.max(axis=1)
images["BANR2"] = BANR2.max(axis=1)
images["NRVNR1"] = NRVNR1.max(axis=1)
images["NRVNR2"] = NRVNR2.max(axis=1)
images["NRVNR3"] = NRVNR3.max(axis=1)
images["NRVNR4"] = NRVNR4.max(axis=1)

cols = images.columns[6:]
images[cols] = images[cols].apply(lambda x: x.apply(lambda x: 1.0 if x >= 0.7 else 0.0))

middle_motors = glob("**/*.jpg", root_dir="Middle_Motors/view_0")
small_motors = glob("**/*.jpg", root_dir="Small_Motors/view_0")
large_motors = glob("**/*.jpg", root_dir="Large_Motors/view_0")

def find_motor_size(x):
    for m in middle_motors:
        if x in m:
            return "Middle"
    for m in small_motors:
        if x in m:
            return "Small"
    for m in large_motors:
        if x in m:
            return "Large"
    return "NA"
        
images["MotorSize"] = images["ImageView_0"].apply(find_motor_size)
print(images.shape)
images = images.loc[images["MotorSize"] != "NA"]
print(images.shape)

def change_image_paths(x):
    motor_size = x["MotorSize"]
    folder = None
    if motor_size == "Middle":
        folder = "Middle_Motors"
    elif motor_size == "Small":
        folder = "Small_Motors"
    elif motor_size == "Large":
        folder = "Large_Motors"
    else:
        raise ValueError(f"Motor_size {motor_size} is not supported.")
    x["ImageView_0"] = os.path.join(folder, "view_0", "images", x["ImageView_0"] + ".jpg")
    x["ImageView_90"] = os.path.join(folder, "view_90", "images", x["ImageView_90"] + ".jpg")
    x["ImageView_180"] = os.path.join(folder, "view_180", "images", x["ImageView_180"] + ".jpg")
    x["ImageView_270"] = os.path.join(folder, "view_270", "images", x["ImageView_270"] + ".jpg")
    x["ImageView_Aufsicht"] = os.path.join(folder, "view_top", "images", x["ImageView_Aufsicht"] + ".jpg")
    x["ImageView_Untersicht"] = os.path.join(folder, "view_bottom", "images", x["ImageView_Untersicht"] + ".jpg")
    return x

cols = images.columns
images[cols] = images[cols].apply(change_image_paths, axis=1)


def check_files_exist(x, base_dir="."):
    assert os.path.exists(os.path.join(base_dir, x))
    return x

cols = images.columns[0:6]
images[cols].apply(lambda x: x.apply(check_files_exist))

images.to_csv("processed_labels_motors.csv", index=False)
