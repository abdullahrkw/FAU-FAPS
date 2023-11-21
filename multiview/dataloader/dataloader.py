import csv
import os
import random

import imgaug.augmenters as ia
import torch
import numpy as np
from PIL import Image, ImageDraw


class MultiViewDataset(object):
    def __init__(self, path, views, labels, base_dir=None, transform=True, normalize=True) -> None:
        self.data = []
        self.transform = transform
        self.normalize = normalize
        self.views = views
        self.labels = labels
        self.base_dir = base_dir
        self.initialize_data(path)
        self.img_aug =  ia.Sometimes(0.96, ia.SomeOf(3, [
            ia.Fliplr(0.8),
            ia.Flipud(0.8),
            ia.Multiply((0.7, 1.3)),
            ia.SaltAndPepper(p=(0, 0.03)),
            ia.GaussianBlur(sigma=(0, 1.5)),
            ia.Affine(translate_percent={'x':(-0.05,0.05), 'y':(-0.05,0.05)}),
            ia.Affine(scale=(0.7, 1.1)), 
            ia.Affine(rotate=(-45, 45)),
        ]))
        self.g_blur = ia.GaussianBlur(sigma=40)

    def __getitem__(self, i) -> dict:
        item = self.data[i]
        output = {}
        for view in self.views:
            img = item[view]
            if self.base_dir is not None:
                img = os.path.join(self.base_dir, img)
            output[view] = self._process_view(self._load_img(img))
        
        label = np.asarray(item["label"], dtype=np.float32)
        label = torch.from_numpy(label)
        output["label"] = label
        return output

    def _process_view(self, img):
        new_w, new_h = (256,256)
        # resize but keep aspect ratio
        background = Image.new("RGB", (new_w, new_h))
        img.thumbnail((new_w, new_h))
        background.paste(img, (0,0))
        img = background

        img = np.asarray(img, dtype=np.float32)
        if self.transform:
            img = self.perform_augmentation(img)
        if self.normalize:
            img = self._normalize(img)
        img = np.reshape(img, (3, new_h, new_w))
        img = torch.from_numpy(img)
        return img

    def __len__(self):
        return len(self.data)
    
    def _load_img(self, path) -> Image:
        img = Image.open(path)
        # img = img.convert(mode="L")
        img = img.convert(mode="RGB")
        return img

    def initialize_data(self, path) -> None:
        with open(path, newline='') as csvfile:
            # first row (header) will be treated as field names
            reader = csv.DictReader(csvfile, fieldnames=None)
            for row in reader:
                item = dict()
                item["label"] = []
                for view in self.views:
                    item[view] = row[view]
                for label in self.labels:
                    if label[0] == "~":
                        item["label"].append(1.0 - float(row[label[1:]]))
                        continue
                    item["label"].append(float(row[label]))
                item["id"] = len(self.data)
                self.data.append(item)
        random.shuffle(self.data)
        random.shuffle(self.data)

    def perform_augmentation(self, img):
        img = self.img_aug(image=img)
        return img

    def _normalize(self, item):
        mean = np.mean(item)
        std = np.std(item)
        norm_item = (item - mean)/std
        return norm_item

if __name__=="__main__":
    views = ["file_name"]
    # Order matter for labels
    labels = ["label", "~label"]
    ROOT_DIR = "/home/vault/iwfa/iwfa018h/FAPS/NewMotorsDataset/Classification/Sheet_Metal_Package/"
    csv_path = os.path.join(ROOT_DIR, "train.csv")
    mv_dst = MultiViewDataset(csv_path, views, labels, base_dir=ROOT_DIR, normalize=False)
    print(mv_dst.data[0])
    item = mv_dst[0]
    view1 = item[views[0]]
    view1 = view1.numpy()
    print(view1.shape)
    view1 = view1.reshape((256, 256, 3))
    view1 = view1.astype(np.uint8)

    img = Image.fromarray(view1)

    img = img.convert("RGB")
    img.save("visualisations/view1.jpg")
