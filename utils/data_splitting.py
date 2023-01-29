import numpy as np
import torch
from torch.utils.data import random_split

from dataloader.dataloader import Dataset


def random_split_dataset(dataset: Dataset, splits):
    if len(splits) != 3:
        raise ValueError(f"Specify 3 splits not {len(splits)}")
    [train, val, test] = random_split(dataset, splits, generator=torch.Generator().manual_seed(42))
    return train, val, test

def k_fold_cross_validation(k: int, ids: list) -> list:
    uniq_ids = list(set(ids))
    folds = np.array_split(uniq_ids, k)
    return folds
