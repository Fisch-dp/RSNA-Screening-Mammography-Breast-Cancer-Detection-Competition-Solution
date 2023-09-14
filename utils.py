import argparse
import gc
import importlib
import os
import sys
import shutil
import pydicom as dcm
import dicomsdl
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import json
import cv2
from sklearn.metrics import roc_auc_score
import timm
import matplotlib.pyplot as plt
from monai.utils import set_determinism
from sklearn.model_selection import StratifiedGroupKFold
from config import *

def set_seed(seed):
    set_determinism(seed=seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_train_dataloader(train_dataset, cfg, sampler=None):
    shu = True
    dL = False
    if sampler is not None: 
        shu = False
        dL = True

    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        shuffle=shu,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=False,
        collate_fn=None,
        drop_last=dL,
    )

    return train_dataloader


def get_val_dataloader(val_dataset, cfg):

    val_dataloader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=cfg.val_batch_size,
        num_workers=cfg.num_workers,
        pin_memory=False,
        collate_fn=None,
    )

    return val_dataloader


def create_checkpoint(model, optimizer, epoch, scheduler=None, scaler=None):
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
    }

    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()

    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()
    return checkpoint

def pfbeta(labels, predictions, beta):
    y_true_count = 0
    ctp = 0
    cfp = 0
    c_precision = 0.00
    c_recall = 0.00
    result = 0.00
    for idx in range(len(labels)):
        prediction = min(max(predictions[idx], 0), 1)
        if (labels[idx]):
            y_true_count += 1
            ctp += prediction
        else:
            cfp += prediction
    beta_squared = beta * beta

    try: c_precision = ctp / (ctp + cfp)
    except: c_precision = 0.00
    try: c_recall = ctp / y_true_count
    except: c_recall = 0.00
    try: result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
    except: result = 0.00
    
    return float(result), float(c_recall), float(c_precision)

def apply_StratifiedGroupKFold(X, y, groups, n_splits, random_state=42):

    df_out = X.copy(deep=True)

    # split
    cv = StratifiedGroupKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    for fold_index, (train_index, val_index) in enumerate(cv.split(X, y, groups)):

        df_out.loc[val_index, "fold"] = fold_index

        # check
        train_groups, val_groups = groups[train_index], groups[val_index]
        assert len(set(train_groups) & set(val_groups)) == 0

    df_out = df_out.astype({"fold": 'int64'})

    return df_out