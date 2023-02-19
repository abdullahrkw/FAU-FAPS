import os

import numpy as np
from matplotlib import pyplot
from sklearn.metrics import (confusion_matrix,
                                accuracy_score,
                                precision_score,
                                recall_score,
                                f1_score,
                                classification_report)
import torch


def eval_for_hyperparam_tuning(dataset, preds):
    gts = np.zeros((len(preds), len(preds[0])))
    predsn = np.zeros((len(preds), len(preds[0])))

    for id in range(len(preds)):
        gt = dataset[id]["label"]
        pred = preds[id]
        # pred = torch.sigmoid(pred)
        gts[id] = gt.numpy()
        predsn[id] = pred.numpy()


    y_true = gts.argmax(axis=1)
    y_pred = predsn.argmax(axis=1)
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy

def evaluate(dataset, preds, classes):
    result = {}
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
    result["accuracy"] = accuracy_score(y_true, y_pred)

    result["micro_precision"] = precision_score(y_true, y_pred, average='micro', zero_division=0)
    result["micro_recall"] = recall_score(y_true, y_pred, average='micro', zero_division=0)
    result["micro_f1"] = f1_score(y_true, y_pred, average='micro', zero_division=0)

    result["macro_precision"] = precision_score(y_true, y_pred, average='macro', zero_division=0)
    result["macro_recall"] = recall_score(y_true, y_pred, average='macro', zero_division=0)
    result["macro_f1"] = f1_score(y_true, y_pred, average='macro', zero_division=0)

    print('\nClassification Report\n')
    print(classification_report(y_true, y_pred, zero_division=0, target_names=classes))
    return result
