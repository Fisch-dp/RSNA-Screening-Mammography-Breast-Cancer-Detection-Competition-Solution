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
from sklearn.model_selection import StratifiedGroupKFold
from config import *
import warnings
import seaborn as sns
from sklearn import metrics
from matplotlib.collections import LineCollection
from sklearn.metrics import PrecisionRecallDisplay

def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class MultiImageBatchSampler(torch.utils.data.Sampler):
    def __init__(self, df, batch_size):
        self.batch_size = batch_size
        self.df = df

    def __iter__(self):
        batch = []
        for id in self.df['prediction_id'].unique():
            item = list(self.df[self.df['prediction_id'] == id].index)
            if len(item) + len(batch) >= self.batch_size:
                yield batch
                batch = []
                batch.extend(item)
            else:
                batch.extend(item)
        if len(batch) > 0:
            yield batch

    def __len__(self):
        return len(self.df['prediction_id'].unique()) // 50
    
def triplet_loss(y_pred, prediction_id_list, margin=10.0):
        loss =[torch.tensor(0.0).to(cfg.device), torch.tensor(0.0).to(cfg.device), torch.tensor(0.0).to(cfg.device)]# [positive, negative, triplet]
        margin = torch.tensor(margin).to(cfg.device)
        for prediction_id in prediction_id_list:
            pos_indices = torch.tensor([index for index, element in enumerate(prediction_id_list) if element == prediction_id]).to(cfg.device)
            neg_indices = torch.tensor([index for index, element in enumerate(prediction_id_list) if element != prediction_id]).to(cfg.device)
            loss[0] += torch.norm(y_pred[pos_indices].unsqueeze(1) - y_pred[pos_indices].unsqueeze(0), dim=2).mean()
            loss[1] += torch.norm(y_pred[pos_indices].unsqueeze(1) - y_pred[neg_indices].unsqueeze(0), dim=2).mean()
            loss[2] += torch.max(loss[0] - loss[1] + margin, torch.tensor(0.0).to(cfg.device))#only hard triplets
            
        return loss[2] / len(prediction_id_list)


def get_train_dataloader(train_dataset, cfg, sampler=None, batch_sampler=None):
    shu = True
    if sampler is not None or batch_sampler is not None: 
        shu = False
    bs = cfg.batch_size
    dl = True
    if batch_sampler is not None:
        bs = 1
        dl = False

    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        batch_sampler=batch_sampler,
        shuffle=shu,
        batch_size=bs,
        num_workers=cfg.num_workers,
        pin_memory=False,
        collate_fn=None,
        drop_last=dl,
    )

    return train_dataloader


def get_val_dataloader(val_dataset, cfg, sampler=None):

    val_dataloader = DataLoader(
        val_dataset,
        sampler=sampler,
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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try: c_precision = ctp / (ctp + cfp)
        except: pass
        try: c_recall = ctp / y_true_count
        except: pass
        try: result = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall)
        except: pass
        
    return float(result), float(c_recall), float(c_precision)

def get_f1score(truth, probability, threshold = np.linspace(0, 1, 50)):
    f1score = []
    precision=[]
    recall=[]
    for t in threshold:
        predict = (probability > t).astype(np.float32)

        tp = ((predict >= 0.5) & (truth >= 0.5)).sum()
        fp = ((predict >= 0.5) & (truth < 0.5)).sum()
        fn = ((predict < 0.5) & (truth >= 0.5)).sum()

        r = tp / (tp + fn + 1e-3)
        p = tp / (tp + fp + 1e-3)
        f1 = 2 * r * p / (r + p + 1e-3)
        f1score.append(f1)
        precision.append(p)
        recall.append(r)
    f1score = np.array(f1score)
    precision = np.array(precision)
    recall = np.array(recall)
    return f1score, precision, recall, threshold

def pfbeta_thres(labels, predictions, beta):
    f1score, precision, recall, thresholds = get_f1score(np.array(labels), np.array(predictions))
    i = f1score.argmax()
    return f1score[i], recall[i], precision[i], thresholds[i]

def apply_StratifiedGroupKFold(X, y, groups, n_splits, random_state=42):

    df_out = X.copy(deep=True)

    # split
    cv = StratifiedGroupKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    for fold_index, (train_index, val_index) in enumerate(cv.split(X, y, groups)):

        df_out.loc[val_index, "fold"] = fold_index
        train_groups, val_groups = groups[train_index], groups[val_index]
        assert len(set(train_groups) & set(val_groups)) == 0

    df_out = df_out.astype({"fold": 'int64'})

    return df_out

def get_probability_hist(df_list, df_names=["Train", "Val"], threshold=None):
    fig, axes = plt.subplots(len(df_list), max(len(cfg.out_classes) + 2, 2), figsize=(20,10))
    plt.subplots_adjust(hspace=0.2, wspace=0.2)
    based_on = "Label"
    if threshold is not None: based_on = "Thres_Output"
    fig.suptitle(f'Based on {based_on}', fontsize=16)
    
    for i, df in enumerate(df_list):
        for j, cls in enumerate(cfg.out_classes):
            if threshold is not None:
                df[f"{cls}"] = (df[f"{cls}_outputs"] > threshold[i][j])
            class0 = df[df[f"{cls}"] == 0][f"{cls}_outputs"].tolist()
            class1 = df[df[f"{cls}"] == 1][f"{cls}_outputs"].tolist() 

            axes[i,j].hist(class0, bins=10, alpha=0.5, label='Negative', weights=np.ones_like(class0)/len(class0))
            axes[i,j].hist(class1, bins=10, alpha=0.5, label='Positive', weights=np.ones_like(class1)/len(class1))
            axes[i,j].legend()
            axes[i,j].set_xlabel("Output Probabilities")
            axes[i,j].set_ylabel("Distribution of samples")
            axes[i,j].set_title(f"{df_names[i]} {cls.capitalize()}")
        for k, site in enumerate([0,1]):
            k += len(cfg.out_classes)
            class0 = df.query(f'site_id == {site} & {cfg.out_classes[0]} == 0')[f"{cfg.out_classes[0]}_outputs"].tolist()
            class1 = df.query(f'site_id == {site} & {cfg.out_classes[0]} == 1')[f"{cfg.out_classes[0]}_outputs"].tolist()

            axes[i,k].hist(class0, bins=10, alpha=0.5, label='Negative', weights=np.ones_like(class0)/len(class0))
            axes[i,k].hist(class1, bins=10, alpha=0.5, label='Positive', weights=np.ones_like(class1)/len(class1))
            axes[i,k].legend()
            axes[i,k].set_xlabel("Output Probabilities")
            axes[i,k].set_ylabel("Distribution of samples")
            axes[i,k].set_title(f"Site{site+1} {df_names[i]} {cfg.out_classes[0].capitalize()}")
    plt.show()
    
def get_corr_matrix(df_list, df_names=["Train", "Val"]):
    fig, axes = plt.subplots(len(df_list), 3, figsize=(30, 15))
    plt.subplots_adjust(hspace=0.2, wspace=0.3)
    for i, df in enumerate(df_list):
        sns.heatmap(df.corr(), ax=axes[i,0])
        axes[i,0].set_title(f"{df_names[i]}")
        for j in [0,1]:
            sns.heatmap(df[df["site_id"] == i].corr(), ax=axes[i,j+1])
            axes[i,j+1].set_title(f"Site{j+1} {df_names[i]}")
    plt.show()

def color_map(data, cmap):
    
    dmin, dmax = np.nanmin(data), np.nanmax(data)
    cmo = plt.cm.get_cmap(cmap)
    cs, k = list(), 256/cmo.N
    
    for i in range(cmo.N):
        c = cmo(i)
        for j in range(int(i*k), int((i+1)*k)):
            cs.append(c)
    cs = np.array(cs)
    data = np.uint8(255*(data-dmin)/(dmax-dmin))
    
    return cs[data]

def get_PR_curve(df_list, best_metric, df_names=["Train", "Val"], by="prediction_id"):
    fig, axes = plt.subplots(len(df_list), max(len(cfg.out_classes),2), figsize=(10, 10))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    for i, df in enumerate(df_list):
        for j, cls in enumerate(cfg.out_classes):
            all_labels = np.array(df.groupby([by]).agg({f"{cls}": "max"})[f"{cls}"])
            site1_labels = np.array(df[df["site_id"]==0].groupby([by]).agg({f"{cls}": "max"})[f"{cls}"])
            site2_labels = np.array(df[df["site_id"]==1].groupby([by]).agg({f"{cls}": "max"})[f"{cls}"])
            selectedp = best_metric[f"Result/{cls.capitalize()[:3]} {df_names[i]} SelectedP"]
            if selectedp==100:
                all_outputs = np.array(df.groupby([by]).agg({f"{cls}_outputs": "max"})[f"{cls}_outputs"])
                site1_outputs = np.array(df[df["site_id"]==0].groupby([by]).agg({f"{cls}_outputs": "max"})[f"{cls}_outputs"])
                site2_outputs = np.array(df[df["site_id"]==1].groupby([by]).agg({f"{cls}_outputs": "max"})[f"{cls}_outputs"])
            else:
                def gem(x):
                    return np.power(np.mean(x.pow(selectedp)), 1.0/selectedp)
                all_outputs = np.array(df.groupby([by]).agg({f"{cls}_outputs": gem})[f"{cls}_outputs"])
                site1_outputs = np.array(df[df["site_id"]==0].groupby([by]).agg({f"{cls}_outputs": gem})[f"{cls}_outputs"])
                site2_outputs = np.array(df[df["site_id"]==1].groupby([by]).agg({f"{cls}_outputs": gem})[f"{cls}_outputs"])
            
            _, precision, recall, thresholds = get_f1score(all_labels, all_outputs)
            display = metrics.PrecisionRecallDisplay(recall=recall, precision=precision)
            display.plot(ax=axes[i,j], label = "All")
            site_1_display = PrecisionRecallDisplay.from_predictions(site1_labels, site1_outputs, ax=axes[i,j], label=f"site_1")
            site_2_display = PrecisionRecallDisplay.from_predictions(site2_labels, site2_outputs, ax=axes[i,j], label=f"site_2")
            
            f_scores = np.linspace(0.1, 0.7, num=4)
            for f_score in f_scores:
                x = np.linspace(0.01, 1)
                y = f_score * x / (2 * x - f_score)
                (l,) = axes[i,j].plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
                axes[i,j].annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

            f1score_max, recall_max, precision_max, threshold_max = pfbeta_thres(all_labels, all_outputs, 1.0)
            auc = metrics.roc_auc_score(df[f"{cls}"], df[f"{cls}_outputs"])
            text=''
            text+=f'MAX f1score {f1score_max: 0.5f} @ th = {threshold_max: 0.5f}\n'
            text+=f'prec {precision_max: 0.5f}, recall {recall_max: 0.5f}, pr-auc {auc: 0.5f}\n'
            text+=f"{df_names[i]} {cls.capitalize()}\n"
            axes[i,j].set_title(text)
            axes[i,j].set_xlabel('Recall')
            axes[i,j].set_ylabel('Precision')
            
            x = recall
            y = precision
            ps = np.stack((x,y), axis=1)
            segments = np.stack((ps[:-1], ps[1:]), axis=1)

            cmap = 'hsv'
            colors = color_map(thresholds, cmap)
            colors = color_map(y[:-1], cmap)
            line_segments = LineCollection(segments, colors=colors, linewidths=3, linestyles='solid', cmap=cmap)

            axes[i,j].set_xlim(np.min(x)-0.1, np.max(x)+0.1)
            axes[i,j].set_ylim(np.min(y)-0.1, np.max(y)+0.1)
            axes[i,j].add_collection(line_segments)
            _ = fig.colorbar(line_segments, cmap='hsv', ax=axes[i][j], label='threshold')
            
            lims = [
                np.min([axes[i,j].get_xlim(), axes[i,j].get_ylim()]),  # min of both axes
                np.max([axes[i,j].get_xlim(), axes[i,j].get_ylim()]),  # max of both axes
            ]
            axes[i,j].plot(lims, lims, '-', alpha=0.3, zorder=0, color="gray")
    plt.show()
