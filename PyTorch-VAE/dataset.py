import csv
import os
import random

import numpy as np
from PIL import Image

import torch
from typing import List, Optional, Sequence, Union
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision import transforms


class MultiViewDataset(object):
    def __init__(self, path, views, labels, base_dir=None, transform=None) -> None:
        self.data = []
        self.transform = transform
        self.views = views
        self.labels = labels
        self.base_dir = base_dir
        self.initialize_data(path)

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
        return tuple(output.values())

    def _process_view(self, img):
        new_w, new_h = (512, 512)
        # resize but keep aspect ratio
        # background = Image.new("RGB", (new_w, new_h))
        # img.thumbnail((new_w, new_h))
        # background.paste(img, (0,0))
        # img = background
        img = img.resize((new_w, new_h))
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.data)
    
    def _load_img(self, path) -> Image:
        img = Image.open(path)
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
 

class VAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module 

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        problem : str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.views = ["file_name"]

        # Order matter for labels
        self.labels = ["label", "~label"]

        self.data_dir = os.path.join(data_path, problem)
        print(self.data_dir)
        self.train_csv_path = os.path.join(self.data_dir, "train-one-class.csv")
        self.test_csv_path = os.path.join(self.data_dir, "test.csv")
        self.val_csv_path = os.path.join(self.data_dir, "val.csv")

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
#       =========================  Motors Dataset  =========================
    
        train_transforms = transforms.Compose([
                                                transforms.RandomHorizontalFlip(p=0.5),
                                                transforms.RandomVerticalFlip(p=0.5),
                                                transforms.RandomRotation(degrees=(-10, 10)),
                                                # transforms.ColorJitter(brightness=(0.7, 1.1), contrast=(0.8, 1.2)),
                                                transforms.Resize(self.patch_size),
                                                transforms.RandomCrop(self.patch_size, padding=4, padding_mode='reflect'),
                                                transforms.ToTensor(),])
        
        val_transforms = transforms.Compose([transforms.Resize(self.patch_size),
                                            transforms.ToTensor(),])
        
        self.train_dataset = MultiViewDataset(self.train_csv_path, views=self.views, labels=self.labels, base_dir=self.data_dir, transform=train_transforms)
        self.val_dataset = MultiViewDataset(self.val_csv_path, views=self.views, labels=self.labels, base_dir=self.data_dir, transform=val_transforms)
        self.test_dataset = MultiViewDataset(self.test_csv_path, views=self.views, labels=self.labels, base_dir=self.data_dir, transform=val_transforms)

        
        
#       ===============================================================
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_dataset,
            batch_size=16,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
