import matplotlib.pyplot as plt
import numpy as np

import torch

def visualize_dataloader_for_class_balance(dataloader, labels, path):
    class_0 = []
    class_1 = []
    ids = []
    batch_size = 8
    
    for i,v in enumerate(dataloader):
        label = v["label"]
        classes = label.argmax(dim=1)
        freq = torch.bincount(classes)
        class_0.append(freq[0])
        class_1.append(freq[1])
        ids.append(i)

    ids = np.asarray(ids)
    fig, ax = plt.subplots(1, figsize=(15, 15))
    width = 0.35
    ax.bar(
        ids[:25],
        class_0[:25],
        width,
        align="edge"
    )
    ax.bar(
        ids[:25] + width,
        class_1[:25],
        width,
        align="edge"
    )
    ax.set_xlabel("Batch index", fontsize=12)
    ax.set_ylabel("No. of samples in batch", fontsize=12)
    ax.set_ylim(top = batch_size)

    plt.savefig(path)
