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
        mixFunction = "none"
    ):
        super().__init__()
        self.cfg = cfg
        self.df = df.reset_index(drop=True)
        self.epoch_len = self.df.shape[0]
        self.Train = Train
        self.mixFunction = cfg.mixFunction
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
        if self.Train and random.random() < cfg.invert_difficult:
            if sample.difficult_negative_case == 1 and sample.biopsy == 1:
                if self.mixFunction == "none":
                    data['cancer'] = np.ones_like(data['cancer']) * self.cfg.valueForInvert
                    data['invasive'] = np.ones_like(data['invasive']) * random.randint(0,1)
                else:
                    mask = self.df.query(f'cancer == 1')
                    sample = self.df.iloc[np.random.choice(mask.index)]
                    supp_data = read(sample, self.aug, self.cfg, self.Train)
                    if self.mixFunction == "simple":
                        data = simple_invert(data, supp_data, self.cfg)
                    elif self.mixFunction == "Mixup":
                        data = Mixup(data, supp_data, force_label = self.cfg.force_label)
                    elif self.mixFunction == "CutMix":
                        data = CutMix(data, supp_data, force_label = self.cfg.force_label)
                    
        return data

    def __len__(self):
        return self.epoch_len