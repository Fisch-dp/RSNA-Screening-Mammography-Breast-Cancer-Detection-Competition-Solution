from torch.utils.tensorboard import SummaryWriter
import wandb
from utils import *
import warnings
import pandas as pd
from PIL import Image
import torch.nn.functional as F

class summaryWriter:
    def __init__(self, cfg, name, write_to = "wandb", notes="", project= "RSNA Second Attempt", group= "Initial Tests"):
        self.cfg = cfg
        self.write_to = write_to
        if write_to == "wandb":
            wandb.init(
            project= project,
            save_code = True,
            config= vars(cfg),
            group= group,
            name= name,
            notes= " ",
            )
            wandb.define_metric("batch")
            wandb.define_metric("epoch")
            wandb.define_metric("Can/*", step_metric="epoch")
            wandb.define_metric("Inv/*", step_metric="epoch")
            wandb.define_metric("image_id/*", step_metric="epoch")
            wandb.define_metric("patient_id/*", step_metric="epoch")
            wandb.define_metric("site1/*", step_metric="epoch")
            wandb.define_metric("site2/*", step_metric="epoch")
            wandb.define_metric("Learning_Rate", step_metric="batch")
            
        elif write_to == "tensorboard":
            self.writer = SummaryWriter(cfg.log_dir)

    ## writing to summaryWriter
    def add_scalar(self, name, value, step, x_axis = None):
        if self.write_to == "wandb":
            wandb.log({f"{name}": value}, step=step)
            if x_axis is not None:
                wandb.log({name: value, x_axis: step})
        elif self.write_to == "tensorboard":
            self.writer.add_scalar(name, value, step)
    
    def save_Image(self, name, image_path):
        # only Implemeted for wandb
        if self.write_to == "wandb":
            wandb.log({name : wandb.Image(Image.open(image_path))})
    
    def add_hparams(self, hparam_dict, metric_dict):
        if self.write_to == "tensorboard":
            self.writer.add_hparams(hparam_dict, metric_dict)
    
    ## Saving
    def save_output_csv(self, epoch, df=None, image_ids=None, out_dic=None, label_dic=None, mode="val"):
        if df is not None:
            df.to_csv(f"{self.cfg.output_dir}/{mode}{epoch}.csv", index=False)
            return
        if image_ids is None or out_dic is None or label_dic is None:
            warnings.warn("Nothing is passed, please pass df or image_ids, out_dic, label_dic, Nothing is saved.", UserWarning)
            return
            
        df = pd.DataFrame({"image_id": image_ids})
        for cls in self.cfg.out_classes:
            df[f"{cls}_outputs"] = out_dic[cls]
            df[f"{cls}_loss"] = F.binary_cross_entropy(torch.tensor(out_dic[cls]).to(torch.float32), torch.tensor(label_dic[cls]).to(torch.float32),reduction="none")
        df = df.merge(self.train_df, on="image_id", how="left")
        df = df.sort_values(by=[f"{self.cfg.out_classes[0]}_loss"], ascending=False).reset_index(drop=True)
        df.to_csv(f"{self.cfg.output_dir}/{mode}{epoch}.csv", index=False)