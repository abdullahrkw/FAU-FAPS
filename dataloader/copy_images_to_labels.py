import csv
import os
import shutil
import glob

ROOT_DIR = "/proj/ciptmp/ic33axaq/FAPS/electricMotor"
motor_sizes = ["L2", "M2"]
views = ["Top", "Side"]
classes = {
    "C":0,
    "MC_2":1,
    "MC_O":2,
    "MC_U":3,
    "MS_1":4,
    "MS_2I":5,
    "MS_2X":6,
    "MS_3":7,
    "MS_4":8,
    "NS_1":9,
    "NS_2I":10,
    "NS_2X":11,
    "NS_3":12,
    "NS_4":13
}
four_classes = {
    "C":0,
    "MC_2":1,
    "MC_O":1,
    "MC_U":1,
    "MS_1":2,
    "MS_2I":2,
    "MS_2X":2,
    "MS_3":2,
    "MS_4":2,
    "NS_1":3,
    "NS_2I":3,
    "NS_2X":3,
    "NS_3":3,
    "NS_4":3
}

with open(os.path.join(ROOT_DIR, 'train_img_labels_paths_new.csv'), 'w+', newline='') as labelcsv:
    labelwriter = csv.writer(labelcsv, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for size in motor_sizes:
        for view in views:
            for root,dir_,files in os.walk(os.path.join(ROOT_DIR, size, view)):
                if "0_Out" in root or "New folder" in root or "zu viel" in root:
                    print(root, len(files))
                    continue
                print(root, len(files))
                for file_ in files:
                    if file_.endswith(".JPG"):
                        file_path = os.path.join(root, file_)
                        label = root.split("/")[-1].replace(size + "_", "")
                        label = classes[label]
                        labelwriter.writerow([file_path, label, view, size])

with open(os.path.join(ROOT_DIR, '4_classes_train_multiview_img_labels_paths.csv'), 'w+', newline='') as labelcsv:
    labelwriter = csv.writer(labelcsv, delimiter=',',
                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    
    img_classes = list(glob.glob(ROOT_DIR + "/*/Side/*"))
    for side in img_classes:
        side_files = list(glob.glob(side + "/*.JPG"))
        top = side.replace("Side", "Top")
        top_files = list(glob.glob(top + "/*.JPG"))
        label = top_files[0].split("/")[-2]
        size = top_files[0].split("/")[-4]
        label = label.replace(size + "_", "")
        label = four_classes[label]
        comb = list(zip(side_files, top_files))
        for item in comb:
            labelwriter.writerow([item[0], item[1], label, size])
