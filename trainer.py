import os

import numpy as np
from pytorch_lightning import loggers, Trainer
from torch.utils.data import DataLoader
import torch

from dataloader.dataloader import Dataset, MultiViewDataset
from network import ResNet, LateFusionNetwork
from torchvision.models import resnet18, resnet50
from utils.data_splitting import random_split_dataset
from eval import evaluate

ROOT_DIR = "/proj/ciptmp/ic33axaq/FAPS/electricMotor/"
num_classes = 14

if num_classes == 14:
    target_classes = ["C", "MC_2", "MC_O", "MC_U", "MS_1", "MS_2I", "MS_2X", "MS_3", "MS_4", "NS_1", "NS_2I", "NS_2X", "NS_3", "NS_4"]
    csv_file = "train_multiview_img_labels_paths.csv"
    four_classes = False
elif num_classes == 4:
    target_classes = ["C", "MC", "MS", "NS"]
    csv_file = "4_classes_train_multiview_img_labels_paths.csv"
    four_classes = True

# use resnet18, resnet34, resnet50
resnet_model = resnet50(weights="IMAGENET1K_V1")

# making last FC layer identity
resnet_model.fc = torch.nn.Identity()

# Freeze weights
for param in resnet_model.parameters():
    param.requires_grad = False

# Unfreeze partial layers
unfreeze_layers = ["layer4", "avgpool", "fc"]
for layer in unfreeze_layers:
    for param in getattr(resnet_model, layer).parameters():
        param.requires_grad = True

multiview_data_csv_path = ROOT_DIR + csv_file

mv_train, mv_val, mv_test = random_split_dataset(MultiViewDataset(multiview_data_csv_path, four_classes=four_classes), [0.7, 0.2, 0.1])

mv_train_loader = DataLoader(mv_train, shuffle=True, batch_size=32, num_workers=4, drop_last=True)
mv_val_loader = DataLoader(mv_val, shuffle=False, batch_size=32, num_workers=4, drop_last=True)
mv_test_loader = DataLoader(mv_test, shuffle=False, batch_size=1, num_workers=4)

tb_late_fusion = loggers.TensorBoardLogger(save_dir="/proj/ciptmp/ic33axaq/FAPS/experiments/",
                             version=None,
                             prefix="late_fusion",
                             name='lightning_logs')

wandb_late_fusion = loggers.WandbLogger(project="FAPS AI",
                             save_dir="/proj/ciptmp/ic33axaq/FAPS/experiments/",
                             version=None,
                             prefix="late_fusion",
                             name='lightning_logs_wandb')

trainer = Trainer(
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,
    max_epochs = 1,
    log_every_n_steps=15,
    logger=[tb_late_fusion, wandb_late_fusion])

fusion_model = LateFusionNetwork(backbone=resnet_model, num_classes=num_classes,lr=1e-4)
trainer.fit(fusion_model, mv_train_loader, mv_val_loader)
trainer.test(dataloaders=mv_test_loader, ckpt_path="last")

preds = trainer.predict(dataloaders=mv_test_loader, ckpt_path="last")
evaluate(mv_test, preds, target_classes)
