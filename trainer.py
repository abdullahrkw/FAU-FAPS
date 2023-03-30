import os

import numpy as np
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning import loggers, Trainer, callbacks
from torch.utils.data import DataLoader
import torch

from dataloader.dataloader import MultiViewDataset
from network import ResNet, LateFusionNetwork
from torchvision.models import resnet18, resnet50, resnet101
from torchvision.models import densenet121
from utils.data_splitting import random_split_dataset
from eval import evaluate

ROOT_DIR = "/home/vault/iwfa/iwfa018h/FAPS/NewMotorsDataset/"
dataset_path = ROOT_DIR + "processed_labels_motors.csv"

views = ["ImageView_0", "ImageView_90", "ImageView_180", "ImageView_270", "ImageView_Aufsicht", "ImageView_Untersicht"]
# Order matter for labels
labels = ["BB", "BK", "BWH1", "BWH2", "BANR1", "BANR2", "NRVNR1", "NRVNR2", "NRVNR3", "NRVNR4"]

num_classes = 10
epochs = 400
lr = 2.2e-4
batch_size = 16
loss_func = torch.nn.BCELoss()
output_activation = torch.nn.Sigmoid()
# use resnet18, resnet34, resnet50, resnet101, densenet121
model = densenet121(weights="IMAGENET1K_V1")
model.name = "densenet121"
# fc for resnet, classifier for densenet
backbone_out_features = model.classifier.in_features
model.classifier = torch.nn.Identity()

# Freeze partial layers
# freeze_layers = ["conv0", "denseblock1"]
# for layer in freeze_layers:
#     for param in getattr(getattr(model, "features"), layer).parameters():
#         param.requires_grad = False

multiview_data_csv_path = dataset_path

mv_train, mv_val, mv_test = random_split_dataset(MultiViewDataset2(multiview_data_csv_path), [0.8, 0.1, 0.1])

mv_train_loader = DataLoader(mv_train, shuffle=True, batch_size=batch_size, num_workers=4, drop_last=True)
mv_val_loader = DataLoader(mv_val, shuffle=False, batch_size=batch_size, num_workers=4, drop_last=True)
mv_test_loader = DataLoader(mv_test, shuffle=False, batch_size=1, num_workers=4)

tb_late_fusion = loggers.TensorBoardLogger(save_dir=ROOT_DIR + "experiments/",
                             version=None,
                             prefix="late_fusion",
                             name='lightning_logs',
                             log_graph=False)


def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-3)
    early_stop = callbacks.EarlyStopping('val_loss', patience=4)
    trainer = Trainer(
        logger=True,
        enable_checkpointing=False,
        max_epochs=10,
        accelerator="gpu",
        devices=1 if torch.cuda.is_available() else None,
        callbacks=[early_stop],
    )
    fusion_model = LateFusionNetwork(backbone=model, backbone_out=backbone_out_features, num_classes=num_classes, lr=lr)
    trainer.fit(fusion_model, mv_train_loader, mv_val_loader)
    return trainer.callback_metrics["val_acc"].item()

# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=100)

# print("Number of finished trials: {}".format(len(study.trials)))
# print("Best trial:")

tb_late_fusion.log_hyperparams({"model": model.name,
                                    "epochs": epochs,
                                    "batch_size": batch_size,
                                    "lr": lr,
                                    "loss_func": loss_func,
                                    "output_activation": output_activation,
                                    "num_classes": num_classes})

early_stop = callbacks.EarlyStopping('val_acc', mode="max", stopping_threshold=0.93)
trainer = Trainer(
    accelerator="auto",
    devices=1 if torch.cuda.is_available() else None,
    max_epochs = epochs,
    log_every_n_steps=15,
    # callbacks=[early_stop],
    logger=[tb_late_fusion])

fusion_model = LateFusionNetwork(backbone=model,
                                    backbone_out=backbone_out_features,
                                    num_classes=num_classes,
                                    lr=2.2e-4,
                                    output_activation=output_activation,
                                    loss_func=loss_func
                                    views=views,
                                    labels=labels)
trainer.fit(fusion_model, mv_train_loader, mv_val_loader)
trainer.test(dataloaders=mv_test_loader, ckpt_path="last")

preds = trainer.predict(dataloaders=mv_test_loader, ckpt_path="last")
result = evaluate(mv_test, preds, labels)
tb_late_fusion.log_metrics(result)
