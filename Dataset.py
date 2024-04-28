import albumentations as A
from albumentations import *
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
import os
from config import *
from torch.utils.data import DataLoader, Dataset
import torch
import random
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    RepeatChanneld,
    Transposed,
    Resized,
    Lambdad
)

class CustomDataset(Dataset):
    def __init__(
        self,
        df,
        cfg,
        Train,
        dataset = "RSNA"
    ):
        super().__init__()
        self.cfg = cfg
        self.df = df.reset_index(drop=True)
        self.epoch_len = self.df.shape[0]
        self.Train = Train
        self.dataset = dataset
        self.aug = Compose([
            LoadImaged(keys="image", image_only=True),
            EnsureChannelFirstd(keys="image"),
            Transposed(keys="image", indices=(0, 2, 1)),
            Resized(keys="image", spatial_size=cfg.img_size, mode="bilinear"),
            Lambdad(keys="image", func=lambda x: x / 255.0),
        ])

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        # Read Data
        if self.dataset == "RSNA":
            data = readTest(sample, self.cfg) if self.Train == "Test" else read(sample, self.cfg)
        else:
            data = readVinDr(sample, self.cfg)
        data = self.aug(data)
        
        # Apply Transformation
        if (self.cfg.Trans is not None and self.Train == "Train"):
            data['image'] = data['image'].permute(1,2,0) * 255
            data["image"] = cfg.Trans(image=np.array(data['image'].to(torch.uint8)))['image']
            Trans2 = ToTensorV2(transpose_mask=False, always_apply=True, p=1.0)
            data['image'] = Trans2(image=data['image'])['image']
            data['image'] = data['image'].to(torch.float32) / 255


        # Mixing
        if self.dataset == "RSNA" and self.Train == "Train" and random.random() < self.cfg.invert_difficult:
            if sample.difficult_negative_case == 1 and sample.biopsy == 1:
                mask = self.df.query(f"cancer == 1 & view == {sample['view']}")
                if len(mask) > 0:
                    sample = self.df.iloc[np.random.choice(mask.index)]
                    supp_data = read(sample, self.cfg)
                    supp_data = self.aug(supp_data)
                    if self.cfg.mixFunction == "simple":
                        data = simple_invert(data, supp_data, self.cfg)
                    elif self.cfg.mixFunction == "Mixup":
                        data = Mixup(data, supp_data, force_label = self.cfg.force_label)
                    elif self.cfg.mixFunction == "CutMix":
                        data = CutMix(data, supp_data, force_label = self.cfg.force_label)
        return data

    def __len__(self):
        return self.epoch_len

def read(sample, cfg):
    data = {
            "image": os.path.join(cfg.root_dir, f"{sample.patient_id}_{sample.image_id}.png"),
            "prediction_id": sample.prediction_id,
            "patient_id": sample.patient_id,
            "image_id": sample.image_id,
            "cancer": np.expand_dims(np.array(sample.cancer, dtype=np.float32), axis=0),
            "biopsy": np.expand_dims(np.array(sample.biopsy, dtype=np.float32), axis=0),
            "invasive": np.expand_dims(np.array(sample.invasive, dtype=np.float32), axis=0),
            "age": np.expand_dims(np.array(sample.age, dtype=np.float32), axis=0),
            "implant": np.expand_dims(np.array(sample.implant, dtype=np.float32), axis=0),
            "machine": np.expand_dims(np.array(sample.machine_id, dtype=np.float32), axis=0),
            "site": np.expand_dims(np.array(sample.site_id, dtype=np.float32), axis=0),
            "view": np.expand_dims(np.array(sample['view'], dtype=np.float32), axis=0),
        }
    return data

def readTest(sample, cfg):
    data = {
            "image": os.path.join(cfg.root_dir, f"{sample.patient_id}_{sample.image_id}.png"),
            "prediction_id": sample.prediction_id,
            "patient_id": sample.patient_id,
            "image_id": sample.image_id,
            "age": np.expand_dims(np.array(sample.age, dtype=np.float32), axis=0),
            "implant": np.expand_dims(np.array(sample.implant, dtype=np.float32), axis=0),
            "machine": np.expand_dims(np.array(sample.machine_id, dtype=np.float32), axis=0),
            "site": np.expand_dims(np.array(sample.site_id, dtype=np.float32), axis=0),
            "view": np.expand_dims(np.array(sample['view'], dtype=np.float32), axis=0),
        }
    return data 
    
def readVinDr(sample, cfg):
    data = {
            "image": os.path.join(cfg.root_dir, f"{sample.laterality}{sample.view_n}/{sample.image_id}.png"),
            "prediction_id": sample.prediction_id,
            "patient_id": sample.patient_id,
            "image_id": sample.image_id,
            "BIRADS": np.array(sample['BIRADS'], dtype=np.long),
            "density": np.array(sample['density'], dtype=np.long),
            "view": np.expand_dims(np.array(sample['view'], dtype=np.float32), axis=0),
        }
    return data


### Mixing Functions ###
def simple_invert(data, supp_data, cfg):
    mix_std = np.random.normal(0, cfg.mix_distr_std) if cfg.mix_distr_std > 0 else 0
    mix_strength = cfg.posMixStrength * (1 + mix_std)
    for key in ["cancer", "invasive", "implant"]:
        data[key] = np.maximum(data[key], supp_data[key])
    for key in ["site", "view", "image"]:
        data[key] = data[key] * (1 - mix_strength) + supp_data[key] * mix_strength
    return data

def Mixup(data, supp_data, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    for key in ["cancer", "invasive", "implant"]:
        data[key] = np.maximum(data[key], supp_data[key])
    for key in ["image"]:
        data[key] = lam * data[key] + (1 - lam) * supp_data[key]
    return data

def CutMix(data, supp_data, alpha=1.0):
    if alpha > 0: 
        lam = np.random.beta(alpha, alpha)
        bbx1, bby1, bbx2, bby2 = rand_bbox(data['image'].size(), lam)
        data['image'][:, bbx1:bbx2, bby1:bby2] = supp_data['image'][:, bbx1:bbx2, bby1:bby2]
        # adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data['image'].size()[-1] * data['image'].size()[-2]))
        for key in ["cancer", "invasive", "implant"]:
            data[key] = np.maximum(data[key], supp_data[key])
        for key in ["site", "view"]:
            data[key] = lam * data[key] + (1 - lam) * supp_data[key]
    return data

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2