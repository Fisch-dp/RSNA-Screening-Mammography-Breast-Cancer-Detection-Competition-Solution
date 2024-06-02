from types import SimpleNamespace
import os
import albumentations as A
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Transposed,
    Lambdad,
    Resized,
)
import torch
import cv2
cfg = SimpleNamespace(**{})

# General
cfg.trial_name = " " 
cfg.root_dir = "/kaggle/input/rsna512x256/output" # Your preprocessed Dataset
cfg.weights = None # Your Weight location
cfg.img_size = (512, 256)
cfg.batch_size = 64
cfg.val_batch_size = 256
cfg.fold = 3
cfg.lr = 3e-4
cfg.weight_decay = 5e-2
cfg.epochs = 10
cfg.Lookahead = False
cfg.optimizer = "AdamW"
cfg.scheduler = "OneCycleLR"
cfg.grad_clip = 10.0  # None
cfg.warmup = 0  
cfg.p_pool = 2
cfg.aux_input = ["age", "implant", "site"]
cfg.soften_label_by = [0.1]  # None
cfg.soften_columns = ["cancer"]
cfg.dataset = "RSNA"
cfg.out_classes = ["cancer", "invasive"]
cfg.mode = "single",# "triplet", "multi"

# Logging
cfg.write_to="wandb"
cfg.project= "RSNA Second Attempt"
cfg.group = "Initial Tests"
cfg.notes = ""

# Resources
cfg.mixed_precision = True
cfg.gpu = 0
cfg.device = "cuda:%d" % cfg.gpu
cfg.num_workers = 2
cfg.train_cache_rate = 0.0
cfg.val_cache_rate = 0.0
cfg.gpu_cache = False
cfg.val_gpu_cache = False

# Paths and Directories
cfg.data_dir = "/kaggle/input/rsna-breast-cancer-detection" # main dataset directory
cfg.data_df = cfg.data_dir + "train.csv"
cfg.test_df = cfg.data_dir + "sample_submission.csv"
cfg.output_dir = f"./folds{0}/"
cfg.model_dir = f"{cfg.output_dir}/fold{cfg.fold}"
os.makedirs(f"{cfg.output_dir}/fold{cfg.fold}", exist_ok=True)

# Dataset
## Augmentation
cfg.Aug = True
## Dataset Split
cfg.num_folds = 4
cfg.df_y = "cancer"
## Dataset Invert and Mixing
cfg.invert_difficult = 0.8  # larger -> more inverts
cfg.valueForInvert = 1  # invert difficult negative cases and biopsy == 1 with this value
cfg.posMixStrength = 0.5
cfg.mixFunction = "simple"
## Dataloader
cfg.random_append = True
cfg.shuffle = True
cfg.drop_last = False
## preprocess df in trainer
cfg.reinitailize_train_every_epoch = False  # whether to call sampling_df
cfg.sample_train_every_epoch = False
cfg.batch_offset = 6  # 6 for batch 64, 4neg disposed
cfg.further_sample_pos = False
cfg.sample_df_with_replace = False
cfg.num_positive_samples = None  # 2
## preprocessing images
cfg.pad = False
cfg.pad_mode = "center"
cfg.validPxRange = [16, 160]

# Model
cfg.backbone = f"tf_efficientnetv2_s"
cfg.num_classes = 1
cfg.pos_weight = 2
cfg.in_channels = 3
cfg.drop_rate = 0.3
cfg.drop_path_rate = 0.2
cfg.pretrained = True

# Evaluation
cfg.evaluation_by = ["image_id", "prediction_id", "patient_id"]
cfg.tta = False
cfg.evalSaveID = "prediction_id"
cfg.log_dir = str(cfg.output_dir + f"/fold{cfg.fold}/")

# Data Augmentation
cfg.Trans = A.Compose([
    # flip
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),

    # contrast
    A.OneOf([
        A.RandomToneCurve(scale=0.3, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.2), contrast_limit=(-0.4, 0.5), brightness_by_max=True, always_apply=False, p=0.5),
        A.RandomGamma(gamma_limit=(80, 120), eps=None, always_apply=False, p=0.5),
    ], p=0.5),

    # Noise
    A.GaussNoise(var_limit=(10.0, 50.0), mean=0, per_channel=True, always_apply=False, p=0.5),

    # geometric
    A.OneOf([
        A.ShiftScaleRotate(shift_limit=None, scale_limit=[-0.15, 0.15], rotate_limit=[-30, 30], interpolation=cv2.INTER_LINEAR,
                           border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=None, shift_limit_x=[-0.1, 0.1],
                           shift_limit_y=[-0.2, 0.2], rotate_method='largest_box', p=0.6),
        A.ElasticTransform(alpha=1, sigma=20, alpha_affine=10, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT,
                           value=0, mask_value=None, approximate=False, same_dxdy=False, p=0.2),
        A.GridDistortion(num_steps=5, distort_limit=0.3, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT,
                         value=0, mask_value=None, normalized=True, p=0.2),
    ], p=0.5),

    # random erase
    A.CoarseDropout(max_holes=6, max_height=0.15, max_width=0.25, min_holes=1, min_height=0.05, min_width=0.1,
                    fill_value=0, mask_fill_value=None, p=0.25),
], p=0.9)

# Image Preprocessing
cfg.img_preprocess = Compose([
    LoadImaged(keys="image", image_only=True),
    EnsureChannelFirstd(keys="image"),
    Transposed(keys="image", indices=(0, 2, 1)),
    Lambdad(keys="image", func=lambda x: x / 255.0),
])

# Training Parameters
cfg.train = True
cfg.eval = True
cfg.start_eval_epoch = 0
cfg.amp = True
cfg.val_amp = False
cfg.num_workers = 2
cfg.min_lr = 1e-5
cfg.lr_div = 1.0
cfg.lr_final_div = 10000.0
cfg.seed = 42
cfg.eval_epochs = 1
cfg.start_cal_metric_epoch = 1
cfg.run_tta_val = False
cfg.restart_epoch = 100