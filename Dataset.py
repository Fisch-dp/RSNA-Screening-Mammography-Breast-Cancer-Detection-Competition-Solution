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
        
        data['image'] = cv2.imread(data['image'])
        data['image'] = cv2.resize(data['image'], 
                                   (cfg.img_size[0], cfg.img_size[1]), 
                                   interpolation = cv2.INTER_LINEAR)
        
        if (cfg.Trans is not None and self.Train):
            data["image"] = cfg.Trans(image=data['image'])['image']
        data['image'] = data['image'].transpose(2,1,0) / 255
        
        if sample.difficult_negative_case == 1 and sample.biopsy == 1 and self.Train and random.random() < cfg.invert_difficult:
            mask = self.df.query(f'cancer == 1 & implant == {sample.implant} & site_id == {sample.site_id} & view == {sample["view"]}')
            sample = self.df.iloc[np.random.choice(mask.index)]
            data['cancer'] = np.expand_dims(np.array(sample.cancer, dtype=np.int8), axis=0)
            data['invasive'] = np.expand_dims(np.array(sample.invasive, dtype=np.int8), axis=0)
            image = cv2.imread(os.path.join(self.cfg.root_dir, f"{sample.patient_id}_{sample.image_id}.png"))
            image = cv2.resize(image, 
                               (cfg.img_size[0], cfg.img_size[1]), 
                               interpolation = cv2.INTER_LINEAR)
            if (cfg.Trans is not None and self.Train):
                image = cfg.Trans(image=image)['image']
            data['image'] += (image.transpose(2,1,0) / 255)
            data['image'] /= 2

        return data

    def __len__(self):
        return self.epoch_len