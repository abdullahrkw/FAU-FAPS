import csv
import os
import random

import imgaug.augmenters as ia
import torch
import numpy as np
from PIL import Image


class MultiViewDataset(object):
    def __init__(self, 
                    path, 
                    views,
                    labels,
                    num_classes,
                    base_dir=None, 
                    transform=True, 
                    normalize=True,
                    pil_image_mode="L") -> None:
        self.data = []
        self.transform = transform
        self.normalize = normalize
        self.views = views
        self.labels = labels
        self.base_dir = base_dir
        self.num_classes = num_classes
        self.pil_image_mode = pil_image_mode
        self.initialize_data(path)
        self.img_aug =  ia.Sometimes(0.96, ia.SomeOf(3, [
            ia.Fliplr(0.8),
            ia.Flipud(0.8),
            ia.SaltAndPepper(p=(0, 0.03)),
            ia.GaussianBlur(sigma=(0, 1.5)),
            ia.Affine(translate_percent={'x':(-0.05,0.05), 'y':(-0.05,0.05)}),
            ia.Affine(scale=(0.7, 1.1)), 
            ia.Affine(rotate=(-45, 45)),
        ]))

    def __getitem__(self, i) -> dict:
        item = self.data[i]
        output = {}
        for view in self.views:
            img = item[view]
            if self.base_dir is not None:
                img = os.path.join(self.base_dir, img)
            output[view] = self._process_view(self._load_img(img))
        
        label = np.zeros((self.num_classes), dtype=np.float32)
        label[item["label"]] = 1
        label = torch.from_numpy(label)
        output["label"] = label
        return output

    def _process_view(self, img):
        new_w, new_h = (224,224)
        # resize but keep aspect ratio
        background = Image.new(self.pil_image_mode, (new_w, new_h))
        img.thumbnail((new_w, new_h))
        background.paste(img, (0,0))
        img = background

        img = np.asarray(img, dtype=np.float32)
        if self.transform:
            img = self.perform_augmentation(img)
        if self.normalize:
            img = self._normalize(img)
        if self.pil_image_mode == "L":
            img = np.expand_dims(img, 0).repeat(3, axis=0)
        img = np.reshape(img, (3, new_h, new_w))
        img = torch.from_numpy(img)
        return img

    def __len__(self):
        return len(self.data)
    
    def _load_img(self, path) -> Image:
        img = Image.open(path)
        img = img.convert(mode=self.pil_image_mode)
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
                        item["label"].append(1.0 - int(row[label[1:]]))
                        continue
                    item["label"].append(int(row[label]))
                item["id"] = len(self.data)
                self.data.append(item)
        random.seed(1000)
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
    views = ["view1", "view2"]
    # Order matter for labels
    labels = ["label"]
    num_classes = 14
    ROOT_DIR = "/home/hpc/iwfa/iwfa018h/FAU-FAPS/multiview/oldElectricMotorData/"
    csv_path = os.path.join(ROOT_DIR, "train.csv")
    mv_dst = MultiViewDataset(csv_path, views, labels, num_classes, base_dir=None, normalize=False, pil_image_mode="L")
    print(mv_dst.data[0])
    item = mv_dst[0]
    view1 = item[views[1]]
    view1 = view1.numpy()
    view1 = np.transpose(view1, (1, 2, 0))
    view1 = view1.astype(np.uint8)

    img = Image.fromarray(view1)

    img = img.convert("RGB")
    img.save("view1.jpg")
