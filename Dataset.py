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
        data = {
            "image": os.path.join(self.cfg.root_dir, f"{sample.patient_id}_{sample.image_id}.png"),
            "prediction_id": sample.prediction_id,
            "patient_id": sample.patient_id,
            "image_id": sample.image_id,
            "cancer": np.expand_dims(np.array(sample.cancer, dtype=np.float32), axis=0),
            "biopsy": np.expand_dims(np.array(sample.biopsy, dtype=np.float32), axis=0),
            "invasive": np.expand_dims(np.array(sample.invasive, dtype=np.float32), axis=0),
            "age": np.expand_dims(np.array(sample.age, dtype=np.float32), axis=0) / 100,
            "implant": np.array(sample.implant, dtype=np.int8),
            "machine": np.array(sample.machine_id, dtype=np.int8),
            "site": np.array(sample.site_id, dtype=np.int8),
            "view": np.array(sample['view'], dtype=np.int8)  
        }
        data = self.aug(data)

        if (cfg.Trans is not None and self.Train):
            data['image'] = data['image'].permute(1,2,0) * 255
            data["image"] = cfg.Trans(image=np.array(data['image'].to(torch.uint8)))['image']
            Trans2 = ToTensorV2(transpose_mask=False, always_apply=True, p=1.0)
            data['image'] = Trans2(image=data['image'])['image']
            data['image'] = data['image'].to(torch.float32) / 255
        
        if sample.biopsy ==1 and self.Train and random.random() < cfg.invert_difficult:
            mask = self.df.query(f'cancer == 1')
            sample = self.df.iloc[np.random.choice(mask.index)]
            data['cancer'] = np.expand_dims(np.array(cfg.valueForInvert, dtype=np.float32), axis=0)
            data['invasive'] = np.expand_dims(np.array(cfg.valueForInvert, dtype=np.float32), axis=0)
            image_data = self.aug({"image": os.path.join(self.cfg.root_dir, f"{sample.patient_id}_{sample.image_id}.png")})
            if (cfg.Trans is not None and self.Train):
                image_data['image'] = image_data['image'].permute(1,2,0) * 255
                image_data["image"] = cfg.Trans(image=np.array(image_data['image'].to(torch.uint8)))['image']
                Trans2 = ToTensorV2(transpose_mask=False, always_apply=True, p=1.0)
                image_data['image'] = Trans2(image=image_data['image'])['image']
                image_data['image'] = image_data['image'].to(torch.float32) / 255
            data['image'] = data['image'] * (1 - cfg.posMixStrength) + image_data['image'] * cfg.posMixStrength

        return data

    def __len__(self):
        return self.epoch_len