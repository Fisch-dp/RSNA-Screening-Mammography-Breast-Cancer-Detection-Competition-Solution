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
        in_chans = 1
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
        self.model.global_pool = nn.Identity()
        #self.global_pool = nn.LPPool2d(cfg.p_pool, (16,8), stride=2)#nn.AdaptiveMaxPool2d(1, return_indices=False)
        self.classifier = nn.Linear(in_features=1280 + 3, out_features=1, bias=True)
        self.model.classifier = nn.Identity()
        self.auxclassifier1 = nn.Linear(in_features=1280 + 3 + 1, out_features=1, bias=True)
        

    def forward(self, x, aux_input, prediction_id):
        x = self.model(x)
#         x = self.global_pool(x)[:,:,0,0]
        x = [x]
        for i in aux_input: x.append(i.view(-1,1))
        x = torch.cat(x, axis=1)
        cancer = self.classifier(x)
        x = torch.cat([x, cancer], axis=1)
        invasive = self.auxclassifier1(x)
        
        return [cancer, invasive]
class MultiView(nn.Module):
    def __init__(self, cfg):
        super(MultiView, self).__init__()
        self.model = Model(cfg)
        self.model = self.model.model
        self.classifier = nn.Linear(in_features=2560 + 3, out_features=1, bias=True)
        self.auxclassifier1 = nn.Linear(in_features=2560 + 3 + 1, out_features=1, bias=True)
            
    def forward(self, x, aux_input, prediction_id):
        x = self.model(x)
#         x = self.global_pool(x)[:,:,0,0]
        x = torch.cat([x[::2,:], x[1::2,:]], axis = 1)
        x = [x]
        for i in aux_input: x.append(i.view(-1,1)[::2,:])
        x = torch.cat(x, axis=1)
        cancer = self.classifier(x)
        x = torch.cat([x, cancer], axis=1)
        invasive = self.auxclassifier1(x)
        
        return [cancer.repeat_interleave(2, dim=0), invasive.repeat_interleave(2, dim=0)]