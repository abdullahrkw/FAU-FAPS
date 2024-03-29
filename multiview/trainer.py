import os

import matplotlib.pyplot as plt
import numpy as np
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning import loggers, Trainer, callbacks
from torch.utils.data import DataLoader
import torch
from torch.utils.data import WeightedRandomSampler
from torchvision.models import resnet18, resnet50, resnet101
from torchvision.models import densenet121

from dataloader.dataloader import MultiViewDataset
from focal_loss import FocalLoss
from network import LateFusionNetwork
from utils.data_splitting import random_split_dataset
from utils.visualisations import visualize_dataloader_for_class_balance
from eval import evaluate

ROOT_DIR = "/home/hpc/iwfa/iwfa018h/FAU-FAPS/multiview/oldElectricMotorData/"
print(ROOT_DIR)
train_csv_path = os.path.join(ROOT_DIR, "train.csv")
test_csv_path = os.path.join(ROOT_DIR, "test.csv")
val_csv_path = os.path.join(ROOT_DIR, "val.csv")

views = ["view1", "view2"]
labels = ["label"]

num_classes = 14
epochs = 100
lr = 0.0001
batch_size = 32
loss_func = torch.nn.CrossEntropyLoss(weight=None)
output_activation = torch.nn.Softmax(dim=1)
# use resnet18, resnet34, resnet50, resnet101, densenet121
model = densenet121(weights="IMAGENET1K_V1")
model.name = "densenet121"
# fc for resnet, classifier for densenet
backbone_out_features = model.classifier.in_features
model.classifier = torch.nn.Identity()

# Freeze partial layers
freeze_layers = []
for layer in freeze_layers:
    for param in getattr(getattr(model, "features"), layer).parameters():
        param.requires_grad = False

train_dataset = MultiViewDataset(train_csv_path, views, labels, num_classes, transform=True)
val_dataset = MultiViewDataset(val_csv_path, views, labels, num_classes, transform=False)
test_dataset = MultiViewDataset(test_csv_path, views, labels, num_classes, transform=False)


# Sampler to for oversampling/undersampling to counter class imbalance
class_counts = np.ones(num_classes)
for i, val in enumerate(train_dataset.data):
    label = np.asarray(val["label"])
    # assuming one-hot encoding
    class_ = label[0]
    class_counts[class_] += 1

sample_weights = np.zeros(len(train_dataset.data))
for i, val in enumerate(train_dataset.data):
    label = np.asarray(val["label"])
    # assuming one-hot encoding
    class_ = label[0]
    sample_weights[i] = 1/class_counts[class_]
print(class_counts)
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_dataset), replacement=True)
mv_train_loader = DataLoader(train_dataset, sampler=sampler, batch_size=batch_size, num_workers=4, drop_last=True)
mv_val_loader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=4, drop_last=True)
mv_test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1, num_workers=4)


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
                                    lr=lr,
                                    output_activation=output_activation,
                                    loss_func=loss_func,
                                    views=views,
                                    labels=labels)
trainer.fit(fusion_model, mv_train_loader, mv_val_loader)
trainer.test(dataloaders=mv_test_loader, ckpt_path="last")
# Prediction on validation dataset
mv_val_loader = DataLoader(val_dataset, shuffle=False, batch_size=1, num_workers=4, drop_last=True)
preds = trainer.predict(dataloaders=mv_val_loader, ckpt_path="last")
result = evaluate(val_dataset, preds, np.arange(0, num_classes))

# prediction on test dataset
preds = trainer.predict(dataloaders=mv_test_loader, ckpt_path="last")
result = evaluate(test_dataset, preds, np.arange(0, num_classes))
tb_late_fusion.log_metrics(result)
