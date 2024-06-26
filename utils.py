import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
import warnings
import seaborn as sns
from sklearn import metrics
from matplotlib.collections import LineCollection
from sklearn.metrics import PrecisionRecallDisplay
from albumentations import *
import random
sys.path.append('./')
from config import *
from numpy.random import default_rng

from warmup_scheduler import GradualWarmupScheduler

##### Dataset 
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

def soft_label(df, soften_by, col=["cancer"]):
    for i,c in enumerate(col):
        if soften_by[i] is not None:
            df.loc[df[c]==0, c] += soften_by[i]
            df.loc[df[c]>0, c] -= soften_by[i]
    return df
    
def sampling_df_with_replace(df):
    rng = default_rng()
    train_pos = np.array(df[df["cancer"] == 1].index)
    train_neg = np.array(df[df["cancer"] == 0].index)
    arranged_data = []

    while len(train_pos) > 0:
        # Randomly select 1-2 positive samples
        num_positive_samples = cfg.num_positive_samples

        selected_positive_index = rng.choice(np.arange(0, len(train_pos)), size=num_positive_samples, replace=False, shuffle=False)
        selected_positive_samples = train_pos[selected_positive_index]

        # Randomly select negative samples to fill the remaining slots
        num_negative_samples = cfg.batch_size - num_positive_samples
        if(len(train_neg) < num_negative_samples):
            selected_negative_samples = train_neg
        else: 
            selected_negative_index = rng.choice(np.arange(0, len(train_neg)), size=num_negative_samples, replace=False, shuffle=False)
            selected_negative_samples = train_neg[selected_negative_index]

        # Concatenate positive and negative samples into a group of 64 items
        group = np.concatenate([selected_positive_samples, selected_negative_samples])
        np.random.shuffle(group)
        
        if(len(train_neg) < num_negative_samples):
            break
        # Add the group to the arranged data
        arranged_data.append(group)

        # Remove the selected samples from the negative samples
        train_neg = np.setdiff1d(train_neg, selected_negative_samples, True)
        
    np.random.shuffle(arranged_data)
    group = np.concatenate([selected_positive_samples, train_neg])
    np.random.shuffle(group)
    arranged_data.append(group)
    arranged_data = np.concatenate(arranged_data)
    return df.iloc[arranged_data]

def sampling_df(df):
    rng = default_rng()
    train_pos = np.array(df[df["cancer"] == 1].index)
    train_pos_backup = np.array(df[df["cancer"] == 1].index)
    train_neg = np.array(df[df["cancer"] == 0].index)
    arranged_data = []
    backupused = False

    while len(train_pos) > 0 and len(train_neg) > cfg.batch_size:
        # Randomly select 1-2 positive samples
        if len(train_pos) > len(train_neg) / cfg.batch_size + cfg.batch_offset and not backupused:
            num_positive_samples = 2
        else: 
            num_positive_samples = 1
        selected_positive_index = rng.choice(np.arange(0, len(train_pos)), size=num_positive_samples, replace=False, shuffle=False)
        selected_positive_samples = train_pos[selected_positive_index]

        # Randomly select negative samples to fill the remaining slots
        num_negative_samples = cfg.batch_size - num_positive_samples
        if(len(train_neg) < num_negative_samples):
            break
        else: 
            selected_negative_index = rng.choice(np.arange(0, len(train_neg)), size=num_negative_samples, replace=False, shuffle=False)
        selected_negative_samples = train_neg[selected_negative_index]

        # Concatenate positive and negative samples into a group of 64 items
        group = np.concatenate([selected_positive_samples, selected_negative_samples])
        np.random.shuffle(group)

        # Add the group to the arranged data
        arranged_data.append(group)

        # Remove the selected samples from the positive and negative samples
        train_pos = np.setdiff1d(train_pos, selected_positive_samples, True)
        train_neg = np.setdiff1d(train_neg, selected_negative_samples, True)
        if len(train_pos) < 1 and len(train_neg) > cfg.batch_size and cfg.further_sample_pos:
            train_pos = train_pos_backup
            backupused = True
    
    np.random.shuffle(arranged_data)
    group = np.concatenate([train_pos, train_neg])
    np.random.shuffle(group)
    arranged_data.append(group)
    arranged_data = np.concatenate(arranged_data)
    return df.iloc[arranged_data]

def get_train_dataloader(train_dataset, cfg, sampler=None, batch_sampler=None, shuffle=False, drop_last=False):
    shu = shuffle
    if sampler is not None or batch_sampler is not None: 
        shu = False
    bs = cfg.batch_size
    dl = drop_last
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

def get_val_dataloader(val_dataset, cfg, sampler=None, batch_sampler=None):
    bsize = cfg.val_batch_size
    if batch_sampler is not None:
        bsize = 1
    val_dataloader = DataLoader(
        val_dataset,
        sampler=sampler,
        batch_sampler=batch_sampler,
        shuffle=False,
        batch_size=bsize,
        num_workers=cfg.num_workers,
        pin_memory=False,
        collate_fn=None,
    )

    return val_dataloader

##### Training Metrics
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

##### Evaluation Metrics
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

##### Evaluation 
def get_probability_hist(df_list, writer, df_names=["Train", "Val"], threshold=None, bins=10):
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

            axes[i,j].hist(class0, bins=bins, alpha=0.5, label='Negative', weights=np.ones_like(class0)/len(class0))
            axes[i,j].hist(class1, bins=bins, alpha=0.5, label='Positive', weights=np.ones_like(class1)/len(class1))
            axes[i,j].legend()
            axes[i,j].set_xlabel("Output Probabilities")
            axes[i,j].set_ylabel("Distribution of samples")
            axes[i,j].set_title(f"{df_names[i]} {cls.capitalize()}")
        for k, site in enumerate([0,1]):
            k += len(cfg.out_classes)
            class0 = df.query(f'site_id == {site} & {cfg.out_classes[0]} == 0')[f"{cfg.out_classes[0]}_outputs"].tolist()
            class1 = df.query(f'site_id == {site} & {cfg.out_classes[0]} == 1')[f"{cfg.out_classes[0]}_outputs"].tolist()

            axes[i,k].hist(class0, bins=bins, alpha=0.5, label='Negative', weights=np.ones_like(class0)/len(class0))
            axes[i,k].hist(class1, bins=bins, alpha=0.5, label='Positive', weights=np.ones_like(class1)/len(class1))
            axes[i,k].legend()
            axes[i,k].set_xlabel("Output Probabilities")
            axes[i,k].set_ylabel("Distribution of samples")
            axes[i,k].set_title(f"Site{site+1} {df_names[i]} {cfg.out_classes[0].capitalize()}")
    plt.savefig(fname=f"{cfg.output_dir}/fold{cfg.fold}/histogram.png")
    plt.show()
    writer.save_Image(name=f'Images/Based on {based_on}', image_path=f"{cfg.output_dir}/fold{cfg.fold}/histogram.png")
    
def get_corr_matrix(df_list, writer, df_names=["Train", "Val"]):
    fig, axes = plt.subplots(len(df_list), 3, figsize=(30, 15))
    plt.subplots_adjust(hspace=0.2, wspace=0.3)
    for i, df in enumerate(df_list):
        df = df.copy()
        df = df.drop(labels=["patient_id", "image_id", "prediction_id"], axis=1)
        sns.heatmap(df.corr(), ax=axes[i,0])
        axes[i,0].set_title(f"{df_names[i]}")
        for j in [0,1]:
            sns.heatmap(df[df["site_id"] == i].corr(), ax=axes[i,j+1])
            axes[i,j+1].set_title(f"Site{j+1} {df_names[i]}")
    plt.savefig(fname=f"{cfg.output_dir}/fold{cfg.fold}/corr_matrix.png")
    plt.show()
    writer.save_Image(name=f'Images/Confusion Matrix', image_path=f"{cfg.output_dir}/fold{cfg.fold}/corr_matrix.png")

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

def get_PR_curve(df_list, best_metric, mode, writer, df_names=["Train", "Val"], by="prediction_id"):
    fig, axes = plt.subplots(len(df_list), max(len(cfg.out_classes),2), figsize=(10, 10))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)
    for i, df in enumerate(df_list):
        for j, cls in enumerate(cfg.out_classes):
            all_labels = np.array(df.groupby([by]).agg({f"{cls}": "max"})[f"{cls}"])
            site1_labels = np.array(df[df["site_id"]==0].groupby([by]).agg({f"{cls}": "max"})[f"{cls}"])
            site2_labels = np.array(df[df["site_id"]==1].groupby([by]).agg({f"{cls}": "max"})[f"{cls}"])
            if mode != "multi":
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
            else: 
                all_outputs = np.array(df[f"{cls}_outputs"])
                site1_outputs = np.array(df[df["site_id"]==0][f"{cls}_outputs"])
                site2_outputs = np.array(df[df["site_id"]==1][f"{cls}_outputs"])

            precision, recall, thresholds = metrics.precision_recall_curve(all_labels, all_outputs)
            display = metrics.PrecisionRecallDisplay(recall=recall, precision=precision)
            display.plot(ax=axes[i,j], label = "All")
            site_1_display = PrecisionRecallDisplay.from_predictions(site1_labels, site1_outputs, ax=axes[i,j], label=f"site_1")
            site_2_display = PrecisionRecallDisplay.from_predictions(site2_labels, site2_outputs, ax=axes[i,j], label=f"site_2")
        
            f1score_max, recall_max, precision_max, threshold_max = pfbeta_thres(all_labels, all_outputs, 1.0)
            auc = metrics.roc_auc_score(df[f"{cls}"], df[f"{cls}_outputs"])
            text=''
            text+=f'MAX f1score {f1score_max: 0.5f} @ th = {threshold_max: 0.5f}\n'
            text+=f'prec {precision_max: 0.5f}, recall {recall_max: 0.5f}, pr-auc {auc: 0.5f}\n'
            text+=f"{df_names[i]} {cls.capitalize()}\n"
            axes[i,j].set_title(text)
            axes[i,j].legend(loc='upper right')
            axes[i,j].set_xlabel('Recall')
            axes[i,j].set_ylabel('Precision')
            axes[i,j].set_xlim([0.0, 1.0])
            axes[i,j].set_ylim([0.0, 1.05])
            
            axes[i,j].plot([0,1],[0,1], color="gray", alpha=0.2)
            f_scores = np.linspace(0.1, 0.7, num=4)
            for f_score in f_scores:
                x = np.linspace(0.01, 1)
                y = f_score * x / (2 * x - f_score)
                (l,) = axes[i,j].plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
                axes[i,j].annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))

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
    plt.savefig(fname=f"{cfg.output_dir}/fold{cfg.fold}/PR_curve.png")
    plt.show()
    writer.save_Image(name=f'Images/PR Curve', image_path=f"{cfg.output_dir}/fold{cfg.fold}/PR_curve.png")

##### Util
def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.use_deterministic_algorithms(False)
    torch.cuda.manual_seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    os.environ['PYTHONHASHSEED'] = str(cfg.seed)

def get_hparams(cfg, dataset):
    hparams = {
        "img_size": cfg.img_size[0],
        "img_size_W": cfg.img_size[1],
        "backbone": f"{cfg.backbone}",
        "pos_weight": cfg.pos_weight,
        "in_channels": cfg.in_channels,
        "drop_rate": cfg.drop_rate,
        "drop_path_rate": cfg.drop_path_rate,
        "Augmentation": cfg.Aug,
        "Batch_size": cfg.batch_size,
        "Num_Folds": cfg.num_folds,
        "seed": cfg.seed,
        "initial LR": cfg.lr,
        "OneCycleLR_div_factor": cfg.lr_div,
        "OneCycleLR_final_div_factor": cfg.lr_final_div,
        "Optimizer": cfg.optimizer,
        "LR_Scheduler": cfg.scheduler,
        "weight_decay": cfg.weight_decay,
        "grad_clip": cfg.grad_clip,
        "Lookahead": str(cfg.Lookahead),
        "TTA": str(cfg.tta is not None),
        "Train_AUG": str(cfg.Trans is not None),
        "Aux_input": str(cfg.aux_input != [])
    }
    if cfg.out_classes != ["cancer"] and dataset == "RSNA": 
        hparams.update({"Auxiliary Training": "True"})
    elif cfg.out_classes != ["BIRADS"] and dataset == "VinDr": 
        hparams.update({"Auxiliary Training": "True"})
    else: hparams.update({"Auxiliary Training": "False"})
    return hparams

##### Training Util
def get_optimizer(cfg, params):
    if cfg.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "Adam":
        optimizer = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "SGD":
        optimizer = torch.optim.SGD(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "RAdam":
        optimizer = torch.optim.RAdam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise NotImplementedError(f"Optimizer {cfg.optimizer} not implemented")
    return optimizer

def get_scheduler(cfg, train_loader_len, optimizer):
    if cfg.scheduler == "OneCycleLR":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.lr,
            epochs=cfg.epochs,
            steps_per_epoch=train_loader_len,
            pct_start=0.1,
            anneal_strategy="cos",
            div_factor=cfg.lr_div,
            final_div_factor=cfg.lr_final_div,
        )
    elif cfg.scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.epochs,
            eta_min=cfg.lr_min,
        )
    elif cfg.scheduler == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.StepLR_step_size,
            gamma=cfg.StepLR_gamma
        )
    elif cfg.scheduler == "warmupOneCycleLR":
        oneCycleLR = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg.lr,
            epochs=cfg.epochs,
            steps_per_epoch=train_loader_len,
            pct_start=0.1,
            anneal_strategy="cos",
            div_factor=cfg.lr_div,
            final_div_factor=cfg.lr_final_div,
        )
        scheduler = GradualWarmupScheduler(optimizer,
                                           multiplier=1,
                                           total_epoch=train_loader_len * cfg.warmup, 
                                           after_scheduler=oneCycleLR
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.StepLR_step_size,
            gamma=cfg.StepLR_gamma
        )
    return scheduler

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

def mean(item):
    return torch.stack(item, dim=0).mean()