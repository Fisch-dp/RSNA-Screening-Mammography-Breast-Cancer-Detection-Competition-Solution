import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from prettytable import PrettyTable
from sklearn import metrics

from utils import *
from config import *
from Model import *
from Lookahead import *
from Dataset import *

class trainer:
    def __init__(self, 
                 cfg,
                 df,
                 model,
                 scaler = GradScaler(),
                 loss_calculation = torch.mean,
                 loss_functions = [ torch.nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor([cfg.pos_weight])),
                                    torch.nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor([cfg.pos_weight])),
                                    ], 
                 mode = "single",# "triplet", "crossAttention", "multi", "multiScale"
                 dataset = "RSNA",
                 ):
        
        set_seed(cfg.seed)
        assert len(loss_functions) == len(cfg.out_classes)
        self.cfg = cfg

        self.df = df
        self.mode = mode
        self.dataset = dataset

        self.val_df = self.df[self.df["fold"] == cfg.fold].reset_index(drop=True)
        self.train_df = self.df[self.df["fold"] != cfg.fold].reset_index(drop=True)
        if dataset == "VinDr":
            self.train_dataset = VinDrDataset(df=self.train_df, cfg=cfg, Train=True)
            self.val_dataset = VinDrDataset(df=self.val_df, cfg=cfg, Train=False)
            self.val_for_train_dataset = VinDrDataset(df=self.train_df, cfg=cfg, Train=False)
        else:
            self.train_dataset = CustomDataset(df=self.train_df, cfg=cfg, Train=True)
            self.val_dataset = CustomDataset(df=self.val_df, cfg=cfg, Train=False)
            self.val_for_train_dataset = CustomDataset(df=self.train_df, cfg=cfg, Train=False)
        if self.mode == "multi" and self.dataset == "RSNA":
            self.train_dataloader = get_train_dataloader(self.train_dataset, cfg, sampler=None, batch_sampler=MultiImageBatchSampler(self.train_df, cfg.batch_size))
            self.val_dataloader = get_val_dataloader(self.val_dataset, cfg, batch_sampler=MultiImageBatchSampler(self.val_df, cfg.val_batch_size))
            self.val_for_train_dataloader = get_val_dataloader(self.val_for_train_dataset, cfg, batch_sampler=MultiImageBatchSampler(self.train_df, cfg.val_batch_size))
        else: 
            self.train_dataloader = get_train_dataloader(self.train_dataset, cfg, sampler=None, batch_sampler=None)
            self.val_dataloader = get_val_dataloader(self.val_dataset, cfg)
            self.val_for_train_dataloader = get_val_dataloader(self.val_for_train_dataset, cfg)
        
        print("train: ", len(self.train_df), " val: ", len(self.val_df))
        print("Train Pos: ", self.train_df['cancer'].sum(), "Val_Pos: ", self.val_df['cancer'].sum())

        self.model = model.to(cfg.device)

        if cfg.optimizer == "AdamW": self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        elif cfg.optimizer == "Adam": self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        elif cfg.optimizer == "SGD": self.optimizer = torch.optim.SGD(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        elif cfg.optimizer == "RAdam": self.optimizer = torch.optim.RAdam(self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        else: raise NotImplementedError

        if cfg.Lookahead:
            self.optimizer = Lookahead(self.optimizer, k=5, alpha=0.5)
        
        if cfg.scheduler == "OneCycleLR": self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                                                            self.optimizer,
                                                            max_lr=cfg.lr,
                                                            epochs=cfg.epochs,
                                                            steps_per_epoch=int(len(self.train_dataloader)),
                                                            pct_start=0.1,
                                                            anneal_strategy="cos",
                                                            div_factor=cfg.lr_div,
                                                            final_div_factor=cfg.lr_final_div,
                                                        )
        elif cfg.scheduler == "CosineAnnealingLR": self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                                                                    self.optimizer,
                                                                    T_max=cfg.epochs,
                                                                    eta_min=cfg.lr_min,
                                                                )
        elif cfg.scheduler == "StepLR": self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=cfg.StepLR_step_size, gamma=cfg.StepLR_gamma)
        else: self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10000000, gamma=0.1,)

        self.loss_functions = [i.to(cfg.device) for i in loss_functions]
        self.scaler = scaler
        self.writer = SummaryWriter(str(cfg.output_dir + f"/fold{cfg.fold}/"))
        self.loss_calculation = loss_calculation
        self.out_classes = cfg.out_classes
        self.aux_input = cfg.aux_input
        self.grad_clip = cfg.grad_clip

        #Saving Best Model Config
        self.best_score = -1.1
        self.best_model = self.model
        self.best_metric = {}

        self.best_loss = 100000000.1
        self.best_Loss_model = self.model
        self.best_Loss_metric = {}

        self.hparams = {
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
        }
        if cfg.Lookahead: self.hparams.update({"Lookahead": "True"})
        else: self.hparams.update({"Lookahead": "False"})

        if self.aux_input != []: self.hparams.update({"Aux_input": "True"})
        else: self.hparams.update({"Aux_input": "False"})

        if self.out_classes != ["cancer"] and dataset == "RSNA": 
            self.hparams.update({"Auxiliary Training": "True"})
        elif self.out_classes != ["BIRADS"] and dataset == "VinDr": 
            self.hparams.update({"Auxiliary Training": "True"})
        else: self.hparams.update({"Auxiliary Training": "False"})

        if self.cfg.tta is not None: self.hparams.update({"TTA": "True"})
        else: self.hparams.update({"TTA": "False"})

        if self.cfg.Trans is not None: self.hparams.update({"Train_AUG": "True"})
        else: self.hparams.update({"Train_AUG": "False"})

    def run_train(self, epoch):
        self.model.train()
        torch.set_grad_enabled(True)
        progress_bar = tqdm(range(len(self.train_dataloader)))
        tr_it = iter(self.train_dataloader)

        label_dic = {f'{i}': [] for i in self.out_classes}
        out_dic = {f'{i}': [] for i in self.out_classes}
        loss_dic = {f'{i}': [] for i in self.out_classes}
            
        for i,itr in enumerate(progress_bar):
            if self.cfg.test_iter is not None:
                if i == self.cfg.test_iter: break
            self.writer.add_scalar('Learning_Rate', self.scheduler.get_last_lr()[-1], epoch * len(self.train_dataloader) + itr)
            batch = next(tr_it)
            if self.mode == "triplet" and self.dataset == "RSNA":   
                loss, label_dic, out_dic, loss_dic, out_print = self.tripletTrain(batch, epoch * len(self.train_dataloader) + itr, label_dic, out_dic, loss_dic, "")
            elif self.mode == "multi" and self.dataset == "RSNA":   
                loss, label_dic, out_dic, loss_dic, out_print = self.MultiTrain(batch, label_dic, out_dic, loss_dic, "")
            elif self.mode == "multiScale" and self.dataset == "RSNA":   
                loss, label_dic, out_dic, loss_dic, out_print = self.multiScaleTrain(batch, label_dic, out_dic, loss_dic, "")
            else:   
                loss, label_dic, out_dic, loss_dic, out_print = self.train(batch, label_dic, out_dic, loss_dic, "")
            
            self.scaler.scale(loss).backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.scheduler.step()
            
            for cls in self.out_classes: out_print += f"{cls}_loss: {np.mean(loss_dic[cls]):.2f}, "
            out_print += f"lr: {self.scheduler.get_last_lr()[-1]:.6f}"
            progress_bar.set_description(out_print)

        save_list = ["F1", "AUC", "Loss", "Pos Loss", "Neg Loss", "Recall", "Precision"]
        table = PrettyTable(["Method", "F1", "AUC", "Loss", "Pos Loss", "Neg Loss", "Recall", "Precision"])
        for cls in self.out_classes: 
            table = self.train_write(label_dic[cls], out_dic[cls], cls, epoch, save_list, table)
            
        print(table.get_string())

    def read_data(self, batch):
        inputs = batch["image"].float().to(self.cfg.device)
        labels_list = [batch[cls].float().to(self.cfg.device) for cls in self.out_classes]
        aux_input_list = [batch[cls].float().to(self.cfg.device) for cls in self.aux_input]
        prediction_id = batch["prediction_id"]
        return inputs, labels_list, aux_input_list, prediction_id
    
    def calculate_save_loss(self, loss_dic, out_dic, label_dic, labels_list, outputs_list):
        loss = []
        for i in range(len(self.out_classes)):
            loss.append(self.loss_functions[i](outputs_list[i], labels_list[i]))
            loss_dic[self.out_classes[i]].append(loss[i].item())
            out_dic[self.out_classes[i]].extend(torch.sigmoid(outputs_list[i]).detach().cpu().numpy()[:,0])
            temp_labels = labels_list[i].detach().cpu().numpy()[:,0] 
            temp_labels[temp_labels != 0.0] = np.expand_dims(np.array(1, dtype=np.float32), axis=0)
            label_dic[self.out_classes[i]].extend(temp_labels) 
        return loss, loss_dic, out_dic, label_dic
    
    def train(self, batch, label_dic, out_dic, loss_dic, out_print):
        inputs, labels_list, aux_input_list, prediction_ids = self.read_data(batch)
        with autocast():
            outputs_list = self.model(inputs, aux_input_list, prediction_ids)
            loss, loss_dic, out_dic, label_dic = self.calculate_save_loss(loss_dic, out_dic, label_dic, labels_list, outputs_list)
        return self.loss_calculation(loss), label_dic, out_dic, loss_dic, out_print

    def multiScaleTrain(self, batch, label_dic, out_dic, loss_dic, out_print):
        inputs, labels_list, aux_input_list, prediction_ids = self.read_data(batch)
        with autocast():
            outputs_list = self.model(inputs, aux_input_list, prediction_ids)
            losses = []
            for i in range(len(outputs_list[0])):
                indexed_outputs_list = [outputs_list[j][i] for j in range(len(outputs_list))]
                loss, loss_dic, out_dic, label_dic = self.calculate_save_loss(loss_dic, out_dic, label_dic, labels_list, indexed_outputs_list)
                losses.extend(loss)
        return self.loss_calculation(loss), label_dic, out_dic, loss_dic, out_print

    def MultiTrain(self, batch, label_dic, out_dic, loss_dic, out_print):
        inputs, labels_list, aux_input_list, prediction_id_list = self.read_data(batch)
        lists_of_labels = [[], []]
        for prediction_id in pd.unique(prediction_id_list):
            indices = [index for index, element in enumerate(prediction_id_list) if element == prediction_id]
            for i, cls in enumerate(self.out_classes):
                lists_of_labels[i].append(labels_list[i][indices].argmax())
        labels_list = [torch.index_select(labels_list[i], 0, torch.LongTensor(lists_of_labels[i]).to(cfg.device)) for i in range(len(self.out_classes))]
        with autocast():
            outputs_list = self.model(inputs, aux_input_list, prediction_id_list)
            loss, loss_dic, out_dic, label_dic = self.calculate_save_loss(loss_dic, out_dic, label_dic, labels_list, outputs_list)
        return self.loss_calculation(loss), label_dic, out_dic, loss_dic, out_print
    
    def tripletTrain(self, batch, iteration, label_dic, out_dic, loss_dic, out_print):
        inputs, labels_list, aux_input_list, prediction_id = self.read_data(batch)
        with autocast():
            outputs_list, intermediate = self.model(inputs, aux_input_list)
            loss, loss_dic, out_dic, label_dic = self.calculate_save_loss(loss_dic, out_dic, label_dic, labels_list, outputs_list)

        tri_loss = triplet_loss(intermediate, prediction_id)
        out_print += f"Triplet Loss: {tri_loss.item():.2f}, "
        self.writer.add_scalar(f"Triplet Loss", tri_loss.item(), iteration)

        loss = self.loss_calculation(loss) + tri_loss
        return loss, label_dic, out_dic, loss_dic, out_print

    def train_metrics(self, all_labels, all_outputs, cls):
        score, recall, precision = pfbeta(all_labels, all_outputs, 1.0)
        loss = F.binary_cross_entropy(torch.tensor(all_outputs).to(torch.float32), torch.tensor(all_labels).to(torch.float32),reduction="none")
        loss_1 = float((loss * torch.tensor(all_labels)).mean())
        loss_0 = float((loss * (1-torch.tensor(all_labels))).mean())
        loss = float(loss.mean())
        auc = float(roc_auc_score(all_labels, all_outputs))
        return [f"{cls[:3]} Train", score, auc, loss, loss_1, loss_0, recall, precision]

    def train_write(self, all_labels, all_outputs, cls, epoch, save_list, table):
        metrics = self.train_metrics(all_labels, all_outputs, cls)
        cls = cls[:3].capitalize() + "/"
        for i in range(len(save_list)):
            self.writer.add_scalar(f"{cls}Train {save_list[i]}", metrics[i + 1], epoch)
            metrics[i+1] = round(metrics[i+1], 3)
        table.add_row(metrics)
        return table
        
    def predict(self, train="Val", best=False):
        if best: model = self.best_model
        else: model = self.model
        model.eval()
        torch.set_grad_enabled(False)
        if train == "Train": 
            dataloader = self.val_for_train_dataloader
            df = self.train_df.copy()
        elif train == "Val": 
            dataloader = self.val_dataloader
            df = self.val_df.copy()
        if self.mode == "multi" and self.dataset == "RSNA":
            df = df[["site_id", "prediction_id", "cancer", "biopsy", "invasive", "BIRADS", "implant", "density", "machine_id", "difficult_negative_case"]]
            df = df.groupby(['prediction_id'], as_index=False).max()
        progress_bar = tqdm(range(len(dataloader)))
        tr_it = iter(dataloader)

        out_dic = {f'{i}': [] for i in self.out_classes}
        all_image_ids = []
        all_prediction_ids = []
        
        for i, _ in enumerate(progress_bar):
            if self.cfg.test_iter is not None:#Testing
                if i == self.cfg.test_iter: break

            #Loading Data
            batch = next(tr_it)
            inputs = batch["image"].float().to(self.cfg.device)
            aux_input_list = [batch[item].float().to(self.cfg.device) for item in self.aux_input]
            all_image_ids.extend(batch["image_id"])
            all_prediction_ids.extend(batch["prediction_id"])

            #Evaluation
            outputs_list = model(inputs, aux_input_list, batch["prediction_id"])
            if self.mode == "multiScale" and self.dataset == "RSNA":
                outputs_list = [outputs_list[j][-1] for j in range(len(outputs_list))]
            if self.cfg.tta:
                outputs_list = [(x + y) / 2 for x, y in zip(outputs_list, model(torch.flip(inputs, dims=[3, ])[0], aux_input_list, batch["prediction_id"]))]

            #Saving Data
            for i in range(len(self.out_classes)):
                out_dic[self.out_classes[i]].extend(torch.sigmoid(outputs_list[i]).detach().cpu().numpy()[:,0])

        all_image_ids = [k.item() for k in all_image_ids]
        if self.cfg.test_iter is not None:#Testing
            if self.mode == "multi" and self.dataset == "RSNA":
                df = df[df["prediction_id"].isin(all_prediction_ids)]
            else:
                df = df[df["image_id"].isin(all_image_ids)]

        #Save Data to DF
        for i in range(len(self.out_classes)):
            df[f"{self.out_classes[i]}_outputs"] = out_dic[self.out_classes[i]]
            
        return df

    def run_eval(self, epoch, train="Val", best=False):
        df = self.predict(train, best=best)
        table = PrettyTable(["Method", "F1", "Bin F1", "AUC", "Loss", "Pos Loss", "Neg Loss", "Recall", "Precision", "Bin Recall", "Bin Precision", "Threshold", "SelectedP"])
        for cls in self.out_classes:
            for k in self.cfg.evaluation_by:
                if k == self.cfg.evalSaveID:
                    if cls == self.out_classes[0]: 
                        BINSCORE, LOSS, data_lib, table = self.eval_write(df, epoch, cls, table, train, by=k)
                        if self.dataset == "RSNA":
                            for s in [0, 1]: _, _, _, table = self.eval_write(df, epoch, cls, table, train, by=k, site_id=s)
                    elif cls != self.out_classes[0]:
                        _, _, lib, table = self.eval_write(df, epoch, cls, table, train, by=k)
                        data_lib.update(lib)
                else:
                    if cls == self.out_classes[0]:
                        _, _, _, table = self.eval_write(df, epoch, cls, table, train, by=k)
        print(table)
        return BINSCORE, LOSS, data_lib
    
    def eval_metrics(self, df, cls, by="prediction_id"):
        if self.mode == "multi" and self.dataset == "RSNA":
            all_labels = np.array(df[f"{cls}"])
            all_outputs = np.array(df[f"{cls}_outputs"])
            bin_score, bin_recall, bin_precision = pfbeta(all_labels, all_outputs, 1.0)
            threshold = -0.01
            selectedp = -0.01
        else:
            all_labels = np.array(df.groupby([by]).agg({f"{cls}": "max"})[f"{cls}"])
            all_outputs, bin_score, bin_recall, bin_precision, threshold, selectedp = self.optimize(df, all_labels, cls, by)

        score, recall, precision = pfbeta(all_labels, all_outputs, 1.0)
        loss = F.binary_cross_entropy(torch.tensor(all_outputs).to(torch.float32), torch.tensor(all_labels).to(torch.float32),reduction="none")
        loss_1 = float((loss * torch.tensor(all_labels)).mean())
        loss_0 = float((loss * (1-torch.tensor(all_labels))).mean())
        loss = float(loss.mean())
        auc = float(roc_auc_score(all_labels, all_outputs))
        return [score, bin_score, auc, loss, loss_1, loss_0, recall, precision, bin_recall, bin_precision, threshold, selectedp]

    def eval_write(self, df, epoch, cls, table, train="Val", by="prediction_id", site_id=None):
        save_list = ["F1", "Bin F1", "AUC", "Loss", "Pos Loss", "Neg Loss", "Recall", "Precision", "Bin Recall", "Bin Precision", "Threshold", "SelectedP"]
        method = ""
        if site_id is not None:
            df = df[df["site_id"] == site_id]
            method = f"site{site_id+1} "
        metrics = self.eval_metrics(df, cls, by)
        cls = cls.capitalize()
        method += f"{cls[:3]} {train} by {by}"

        if by != "prediction_id": by += "/"
        elif by == "prediction_id": 
            if site_id is not None:
                by = f"site{site_id+1}/"
                cls = cls[:3] + " "
            else:
                by = ""
                cls = cls[:3] + "/"

        if train == "Val":
            for i in range(len(save_list[:-2])):
                self.writer.add_scalar(f"{by}{cls}{train} {save_list[i]}", metrics[i], epoch)
        data_lib = {}
        for i in range(len(save_list)):
            data_lib[f"Result/{cls[:3]} {train} {save_list[i]}"] = metrics[i]
            metrics[i] = round(metrics[i], 3)
        metrics = [method] + metrics
        table.add_row(metrics)
        return metrics[2], metrics[4], data_lib, table

    def optimize(self, df, all_labels, cls, by="prediction_id"):
        bin_score = -0.01
        threshold = 0.0
        selectedp = 1.0
        bin_recall = 0.0
        bin_precision = 0.0

        iter = [1,2,3,4,5,6,7,8,9,10,100]
        if cls != "cancer": iter = [1]
        for p in iter:
            if p==100: all_outputs = np.array(df.groupby([by]).agg({f"{cls}_outputs": "max"})[f"{cls}_outputs"])
            else:
                def funct(x):
                    return np.power(np.mean(x.pow(p)), 1.0/p)
                all_outputs = np.array(df.groupby([by]).agg({f"{cls}_outputs": funct})[f"{cls}_outputs"])
                temp_f1, temp_recall, temp_precision, temp_threshold = pfbeta_thres(all_labels, all_outputs, 1.0)
                if temp_f1 > bin_score:
                    bin_score = temp_f1
                    bin_recall = temp_recall
                    bin_precision = temp_precision
                    threshold = temp_threshold
                    selectedp = p
        
        if selectedp==100:
            all_outputs = np.array(df.groupby([by]).agg({f"{cls}_outputs": "max"})[f"{cls}_outputs"])
        else:
            def gem(x):
                return np.power(np.mean(x.pow(selectedp)), 1.0/selectedp)
            all_outputs = np.array(df.groupby([by]).agg({f"{cls}_outputs": gem})[f"{cls}_outputs"])
        
        return all_outputs, bin_score, bin_recall, bin_precision, threshold, selectedp

    def saving_best(self, score, loss, val_metric, epoch):
        if score > self.best_score:
            print(f"SAVING CHECKPOINT: val_metric {self.best_score} -> {score}")
            self.best_score = score
            self.best_metric = val_metric
            self.best_metric['Result/Stop_Epoch'] = epoch
            self.best_model = self.model
            checkpoint = create_checkpoint(
                            self.best_model,
                            self.optimizer,
                            epoch,
                            scheduler=self.scheduler,
                            scaler=self.scaler,
                        )
            torch.save(
                checkpoint,
                f"{self.cfg.output_dir}/fold{self.cfg.fold}/checkpoint_best_metric.pth",
            )

        if loss < self.best_loss:
            print(f"SAVING CHECKPOINT: Loss_val_metric {self.best_loss} -> {loss}")
            self.best_loss = loss
            self.best_Loss_metric = val_metric
            self.best_Loss_metric['Result/Stop_Epoch'] = epoch
            self.best_Loss_model = self.model
            checkpoint = create_checkpoint(
                self.best_Loss_model,
                self.optimizer,
                epoch,
                scheduler=self.scheduler,
                scaler=self.scaler,
            )
            torch.save(
                checkpoint,
                f"{self.cfg.output_dir}/fold{self.cfg.fold}/checkpoint_best_Loss_metric.pth",
            )
                
    def fit(self):
        for epoch in range(self.cfg.epochs):
            print("EPOCH:", epoch)
            self.run_train(epoch)
            score, loss, val_metric = self.run_eval(epoch, train="Val")
            self.saving_best(score, loss, val_metric, epoch)

        _, _, train_metric = self.run_eval(epoch=self.best_metric['Result/Stop_Epoch'], train="Train", best=True)

        self.best_Loss_metric = {f'Loss_{i}': self.best_Loss_metric[i] for i in self.best_Loss_metric}

        self.best_metric.update(self.best_Loss_metric)
        self.best_metric.update(train_metric)
        self.writer.add_hparams(self.hparams, self.best_metric)
