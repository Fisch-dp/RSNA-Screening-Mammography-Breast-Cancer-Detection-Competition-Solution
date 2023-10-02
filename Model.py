import timm 
import torch
import torch.nn as nn
from config import *
from utils import *
from timm.models.layers import BatchNormAct2d
import copy
class Model(nn.Module):
    def __init__(self):
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
        sequence = copy.deepcopy(self.model.blocks[-1])
        sequence[0].conv_pw = nn.Conv2d(256, 960, kernel_size=(1, 1), stride=(1, 1), bias=False)
        sequence[0].conv_pwl = nn.Conv2d(960, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
        sequence[0].bn3 = BatchNormAct2d(320, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        sequence[1].conv_pw = nn.Conv2d(320, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
        sequence[1].se.conv_reduce = nn.Conv2d(1536, 128, kernel_size=(1, 1), stride=(1, 1))
        sequence[1].se.conv_expand = nn.Conv2d(128, 1536, kernel_size=(1, 1), stride=(1, 1))
        sequence[1].conv_pwl = nn.Conv2d(1536, 320, kernel_size=(1, 1), stride=(1, 1), bias=False)
        sequence[1].bn3 = BatchNormAct2d(320, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        for i in range(2, len(sequence)):
            sequence[i] = sequence[1]
        self.model.blocks.append(sequence)
        self.model.conv_head = nn.Conv2d(320, 1536, kernel_size=(1, 1), stride=(1, 1), bias=False)
        self.model.bn2 = BatchNormAct2d(1536, eps=0.001, momentum=0.1, affine=True, track_running_stats=True)
        self.model.classifier = nn.Linear(in_features=1536, out_features=1, bias=True)

        self.classifier = nn.Linear(in_features=1536 + len(cfg.aux_input), out_features=1, bias=True)
        self.model.classifier = nn.Identity()
        self.auxclassifier1 = nn.Linear(in_features=1536 + len(cfg.aux_input) + 1, out_features=1, bias=True)

    def forward(self, x, aux_input):
        x = self.model(x)
        x = [x]
        for i in range(len(aux_input)):
            x.append(aux_input[i].view(-1,1))
        x = torch.cat(x, axis=1)
        cancer = self.classifier(x)
        x = torch.cat([x, cancer], axis=1)
        invasive = self.auxclassifier1(x)
        
        return [cancer, invasive]