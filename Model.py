import timm 
import torch
import torch.nn as nn
from config import *
from utils import *
import copy
class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.model = timm.create_model(
        cfg.backbone,
        pretrained=False,
        num_classes=cfg.num_classes,
        drop_rate=cfg.drop_rate,
        drop_path_rate=cfg.drop_path_rate,
        )
        if cfg.in_channels != 3:
            self.model.conv_stem = nn.Conv2d(
                cfg.in_channels,
                self.model.conv_stem.out_channels,
                kernel_size=self.model.conv_stem.kernel_size,
                stride=self.model.conv_stem.stride,
                padding=self.model.conv_stem.padding,
                bias=self.model.conv_stem.bias,
            )
        self.SelfAttention = nn.TransformerEncoderLayer(d_model=1280, nhead=8, dim_feedforward=1280, dropout=cfg.drop_rate)
        self.CrossAttention = nn.TransformerEncoderLayer(d_model=8, nhead=2, dim_feedforward=1280, dropout=cfg.drop_rate)
        self.classifier = nn.Linear(in_features=1280 + len(cfg.aux_input), out_features=1, bias=True)
        self.model.classifier = nn.Identity()
        self.auxclassifier1 = nn.Linear(in_features=1280 + len(cfg.aux_input) + 1, out_features=1, bias=True)

    def forward(self, x, aux_input, prediction_id_list):
        x = self.model(x)
        x = self.SelfAttention(x)
        output_list = []
        for prediction_id in prediction_id_list:
            indices = torch.tensor([index for index, element in enumerate(prediction_id_list) if element == prediction_id]).to(cfg.device)
            pad = (0, 0,
                   0, 0,
                   0, 0,
                   0, 8 - len(indices))
            output = F.pad(x[indices], pad, "constant", 0).view(1280, -1)
            output = self.CrossAttention(output)
            output = output.sum(axis=1).view(1, 1280)
            output_list.append(output)
        x = [torch.cat(output_list, axis=0)]
        for i in range(len(aux_input)):
            x.append(aux_input[i].view(-1,1))
        x = torch.cat(x, axis=1)
        cancer = self.classifier(x)
        x = torch.cat([x, cancer], axis=1)
        invasive = self.auxclassifier1(x)
        return [cancer, invasive]