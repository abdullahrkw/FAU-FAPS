import os

import numpy as np
from matplotlib import pyplot
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from dataloader.dataloader import Dataset
from network import ResNet


def evaluate(dataset, preds, classes):
    gts = np.zeros((len(preds), len(classes)))
    predsn = np.zeros((len(preds), len(classes)))

    for id in range(len(preds)):
        gt = dataset[id]["label"]
        pred = preds[id]
        # pred = torch.sigmoid(pred)
        gts[id] = gt.numpy()
        predsn[id] = pred.numpy()


    print(predsn[:10])
    y_true = gts.argmax(axis=1)
    y_pred = predsn.argmax(axis=1)
    print(y_pred)
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    print('\nAccuracy: {:.2f}\n'.format(accuracy_score(y_true, y_pred)))

    print('Micro Precision: {:.2f}'.format(precision_score(y_true, y_pred, average='micro', zero_division=0)))
    print('Micro Recall: {:.2f}'.format(recall_score(y_true, y_pred, average='micro', zero_division=0)))
    print('Micro F1-score: {:.2f}\n'.format(f1_score(y_true, y_pred, average='micro', zero_division=0)))

    print('\nClassification Report\n')
    print(classification_report(y_true, y_pred, zero_division=0, target_names=classes))
