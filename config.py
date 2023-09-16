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
cfg.eval_epochs = 1
cfg.start_eval_epoch = 0
cfg.amp = True
cfg.val_amp = False
cfg.num_workers = 2
cfg.seed = -1
cfg.eval_epochs = 1
cfg.start_cal_metric_epoch = 1
cfg.warmup = 1
cfg.Lookahead = False

# dataset
cfg.img_size = (512,256)  
cfg.batch_size = 64
cfg.val_batch_size = 128
cfg.Aug = True
cfg.AugStrength = 1
cfg.num_folds = 4
cfg.seed = 42
cfg.df_y = "cancer"

# model
cfg.backbone = f"tf_efficientnetv2_s"
cfg.num_classes = 1
cfg.pos_weight = 2
cfg.in_channels = 3
cfg.drop_rate = 0.3
cfg.drop_path_rate = 0.2

# Eval
cfg.evaluation_by = ["image_id", "prediction_id", "patient_id"]