from types import SimpleNamespace

cfg = SimpleNamespace(**{})

# resources
cfg.mixed_precision = True  
cfg.gpu = 0
cfg.device = "cuda:%d" % cfg.gpu
cfg.num_workers = 2
cfg.weights = None

# data
cfg.data_dir = "/kaggle/input/rsna-breast-cancer-detection"
cfg.data_df = "/kaggle/input/rsna-breast-cancer-detection/train.csv"
cfg.root_dir = "/kaggle/input/rsna-breast-cancer-detection-poi-images/bc_1280_train_lut"
cfg.test_df = cfg.data_dir + "sample_submission.csv"
cfg.output_dir = f"./folds{1}/"

# train  
cfg.train = True
cfg.eval = True
cfg.start_eval_epoch = 0
cfg.amp = True
cfg.val_amp = False
cfg.num_workers = 2
cfg.Lookahead = False
cfg.lr = 3e-4
cfg.min_lr = 1e-5
cfg.weight_decay = 5e-2
cfg.epochs = 10
cfg.seed = -1
cfg.eval_epochs = 1
cfg.start_cal_metric_epoch = 1
cfg.warmup = 1
cfg.run_tta_val = False
cfg.lr_final_div = 10000.0
#cfg.warmup_epoch = 1
cfg.restart_epoch = 100

# dataset
cfg.img_size = (512,256)  
cfg.batch_size = 64
cfg.val_batch_size = 128
cfg.train_cache_rate = 0.0
cfg.val_cache_rate = 0.0
cfg.gpu_cache = False
cfg.val_gpu_cache = False
cfg.Aug = True
cfg.AugStrength = 1
cfg.num_folds = 4
cfg.seed = 42
cfg.df_y = "cancer"
cfg.invert_difficult = 0.5#larger -> more inverts
cfg.valueForInvert = 1#invert difficult negative cases and biopsy == 1 with this value
cfg.posMixStrength = 0.5#mix positive cases with this strength
cfg.mixFunction = "simple"#Mixup, CutMix, simple

# model
cfg.backbone = f"tf_efficientnetv2_s"
cfg.num_classes = 1
cfg.pos_weight = 2
cfg.in_channels = 3
cfg.drop_rate = 0.3
cfg.drop_path_rate = 0.2
cfg.aux_input = ["age", "implant", "view", "site", "machine"]

# Eval
cfg.evaluation_by = ["image_id", "prediction_id", "patient_id"]
cfg.tta = False #True
cfg.evalSaveID = "prediction_id"