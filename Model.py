import timm 
import torch
import torch.nn as nn
from config import *

class Model(nn.Module):
    def __init__(self, cfg):
        super(Model, self).__init__()
        self.model = timm.create_model(
        cfg.backbone,
        pretrained=cfg.pretrained,
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
        self.classifier = nn.Linear(in_features=1280 + 5, out_features=1, bias=True)
        self.model.classifier = nn.Identity()
        self.auxclassifier1 = nn.Linear(in_features=1280 + 5 + 1, out_features=1, bias=True)

    def forward(self, x, aux_input, prediction_id):
        x = self.model(x)
        x = [x]
        for i in aux_input: x.append(i.view(-1,1))
        x = torch.cat(x, axis=1)
        cancer = self.classifier(x)
        x = torch.cat([x, cancer], axis=1)
        invasive = self.auxclassifier1(x)
        
        return [cancer, invasive]