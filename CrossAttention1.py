import timm 
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import *
from utils import *
import copy

class CrossAttention(nn.Module):
    def __init__(self, cfg):
        super(CrossAttention, self).__init__()
        self.W_k = nn.Conv2d(1280, 1280, 1)
        self.W_q = nn.Conv2d(1280, 1280, 1)
        self.W_v = nn.Conv2d(1280, 1280, 1)
        self.drop_rate = cfg.drop_rate
    def forward(self, Qvector, KVvector):
        Q = self.W_q(Qvector)
        V = self.W_v(KVvector)
        K = self.W_k(KVvector)
        return F.scaled_dot_product_attention(Q, K, V, dropout_p=self.drop_rate)