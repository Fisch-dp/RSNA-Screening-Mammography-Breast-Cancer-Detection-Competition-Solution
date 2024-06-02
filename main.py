import wandb
import sys
import pandas as pd
import torch
from torch.cuda.amp import GradScaler

sys.path.append("./")
from Dataset import *
from Lookahead import *
from Model import *
from config import *
from utils import *
from trainer import *

def main():
    set_seed(cfg.seed)
    ############ Read dataset ############
    df = pd.read_csv(cfg.data_df)

    df["prediction_id"] = df.patient_id.apply(str) + "_" + df.laterality
    df['site_id'] -= 1
    df["laterality"] = df["laterality"].map({"L":0,"R":1})
    df["density"] = df["density"].map({"A":0,"B":1, "C":2,"D":3})
    df['view'] = df['view'].map({machine_id: idx for idx, machine_id in enumerate(sorted(df['view'].unique()))})
    df['age'] = df['age'].fillna(df['age'].mean())
    df['machine_id'] = df['machine_id'].map({machine_id: idx for idx, machine_id in enumerate(sorted(df['machine_id'].unique()))})
    num_bins = 5
    df["age_bin"] = pd.cut(df['age'].values.reshape(-1), bins=num_bins, labels=False)
    df["age"] = df['age'] / 100

    strat_cols = [
        'laterality', 'view', 'biopsy','invasive', 'BIRADS', 'age_bin',
        'implant', 'density','machine_id', 'difficult_negative_case',
        'cancer',
    ]

    df['stratify'] = ''
    for col in strat_cols:
        df['stratify'] += df[col].astype(str)

    df = apply_StratifiedGroupKFold(
            X=df,
            y=df["stratify"].values,
            groups=df["patient_id"].values,
            n_splits=cfg.num_folds,
            random_state=cfg.seed)

    original_df = df.reset_index(drop=True)
    original_df = original_df[["image_id","fold"]]
    # df = df.drop(labels=["patient_id", "image_id", "prediction_id", "age_bin"], axis=1)

    ############ Training ############
    ter = trainer(cfg,
        df,
        model = MultiView(cfg),
        scaler = GradScaler(),
        loss_calculation = mean,
        loss_functions = [ torch.nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor([cfg.pos_weight])),
                        torch.nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor([cfg.pos_weight])),
                        ], 
        mode = "single",
        name = " ",
        write_to="wandb",
        notes="",
        project= "RSNA Second Attempt", 
        group= "Initial Tests"
    )
    best_metrics = ter.fit()

    ############ Evaluation ############
    train_df = ter.predict("Train", best=True)
    val_df = ter.predict("Val", best=True)
    train_df.to_csv(f"train.csv", index=False)
    val_df.to_csv(f"val.csv", index=False)

    train_wandb_table = wandb.Table(dataframe=train_df)
    val_wandb_table = wandb.Table(dataframe=val_df)
    wandb.run.log({"Eval on Train": train_wandb_table})
    wandb.run.log({"Eval on Val": val_wandb_table})

    get_probability_hist([train_df, val_df], ter.writer, bins=30)
    get_corr_matrix([train_df, val_df], ter.writer)
    get_PR_curve([train_df, val_df], ter.writer, mode="single", best_metric=ter.best_metric)

    ############ End ############
    wandb.finish()

if __name__ == "__main__":
    main()