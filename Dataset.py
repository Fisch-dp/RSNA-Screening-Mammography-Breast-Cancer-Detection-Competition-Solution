import albumentations as A
from albumentations import *
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
import os
from config import *
from utils import *
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
    ):
        super().__init__()
        self.cfg = cfg
        self.df = df.reset_index(drop=True)
        self.epoch_len = self.df.shape[0]
        self.Train = Train
        self.aug = Compose([
            LoadImaged(keys="image", image_only=True),
            EnsureChannelFirstd(keys="image"),
            RepeatChanneld(keys="image", repeats=3),
            Transposed(keys="image", indices=(0, 2, 1)),
            Resized(keys="image", spatial_size=cfg.img_size, mode="bilinear"),
            Lambdad(keys="image", func=lambda x: x / 255.0),
        ])

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        data = read(sample, self.aug, self.cfg, self.Train)
        if self.Train and random.random() < self.cfg.invert_difficult:
            if sample.difficult_negative_case == 1 and sample.biopsy == 1:
                mask = self.df.query(f"cancer == 1 & view == {sample['view']}")
                sample = self.df.iloc[np.random.choice(mask.index)]
                supp_data = read(sample, self.aug, self.cfg, self.Train)
                if self.cfg.mixFunction == "simple":
                    data = simple_invert(data, supp_data, self.cfg)
                elif self.cfg.mixFunction == "Mixup":
                    data = Mixup(data, supp_data, force_label = self.cfg.force_label)
                elif self.cfg.mixFunction == "CutMix":
                    data = CutMix(data, supp_data, force_label = self.cfg.force_label)
                
        return data

    def __len__(self):
        return self.epoch_len
    
class VinDrDataset(Dataset):
    def __init__(
        self,
        df,
        cfg,
        Train,
    ):
        super().__init__()
        self.cfg = cfg
        self.df = df.reset_index(drop=True)
        self.epoch_len = self.df.shape[0]
        self.Train = Train
        self.aug = Compose([
            LoadImaged(keys="image", image_only=True),
            EnsureChannelFirstd(keys="image"),
            RepeatChanneld(keys="image", repeats=3),
            Transposed(keys="image", indices=(0, 2, 1)),
            Resized(keys="image", spatial_size=cfg.img_size, mode="bilinear"),
            Lambdad(keys="image", func=lambda x: x / 255.0),
        ])

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        data = readVinDr(sample, self.aug, self.cfg, self.Train)
        return data

    def __len__(self):
        return self.epoch_len
    
def readVinDr(sample, aug, cfg, Train):
    data = {
            "image": os.path.join(cfg.root_dir, f"{sample.patient_id}/{sample.image_id}.png"),
            "prediction_id": sample.prediction_id,
            "patient_id": sample.patient_id,
            "image_id": sample.image_id,
            "BIRADS": np.expand_dims(np.array(sample['BIRADS'], dtype=np.float32), axis=0),
            "density": np.expand_dims(np.array(sample['density'], dtype=np.float32), axis=0),
            "view": np.expand_dims(np.array(sample['view'], dtype=np.float32), axis=0),
        }
    data = aug(data)

    if (cfg.Trans is not None and Train):
            data['image'] = data['image'].permute(1,2,0) * 255
            data["image"] = cfg.Trans(image=np.array(data['image'].to(torch.uint8)))['image']
            Trans2 = ToTensorV2(transpose_mask=False, always_apply=True, p=1.0)
            data['image'] = Trans2(image=data['image'])['image']
            data['image'] = data['image'].to(torch.float32) / 255
    return data