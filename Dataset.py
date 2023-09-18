import albumentations as A
from albumentations import *
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
import os
from config import *
from utils import *
from torch.utils.data import DataLoader, Dataset
import torch

class CustomDataset(Dataset):
    def __init__(
        self,
        df,
        cfg,
        Train,
    ):
        super().__init__()
        self.cfg = cfg
        self.df = df
        self.epoch_len = self.df.shape[0]
        self.Train = Train

    def __getitem__(self, idx):
        sample = self.df.iloc[idx]
        data = {
            "image": os.path.join(self.cfg.root_dir, f"{sample.patient_id}_{sample.image_id}.png"),
            "prediction_id": sample.prediction_id,
            "patient_id": sample.patient_id,
            "image_id": sample.image_id,
            "cancer": np.expand_dims(np.array(sample.cancer, dtype=np.int8), axis=0),
            "biopsy": np.expand_dims(np.array(sample.biopsy, dtype=np.int8), axis=0),
            "invasive": np.expand_dims(np.array(sample.invasive, dtype=np.int8), axis=0),
            "age": np.expand_dims(np.array(sample.age, dtype=np.int8), axis=0) / 100,
            "implant": np.array(sample.implant, dtype=np.int8),
            "machine": np.array(sample.machine_id, dtype=np.int8),
            "site": np.array(sample.site_id, dtype=np.int8),
            "view": np.array(sample['view'], dtype=np.int8)  
        }
        
        data['image'] = cv2.imread(data['image'],cv2.IMREAD_GRAYSCALE)
        if (cfg.Trans is not None and self.Train):
            data['image'] = data['image'].permute(1,2,0) * 255
            data["image"] = cfg.Trans(image=np.array(data['image'].to(torch.uint8)))['image']
            data['image'] = ToTensorV2(transpose_mask=False, always_apply=True, p=1.0)(image=data['image'])['image']
            data['image'] = data['image'].to(torch.float32) / 255
        return data

    def __len__(self):
        return self.epoch_len