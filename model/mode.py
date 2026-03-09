
import torch

import torch.nn as nn

from model.build_model import *

from einops import rearrange


class Model(nn.Module):
    def __init__(self,
                 config,
                 ):
        super().__init__()
        self.config = config


    def forward(self, x, rots, trans, intrins, distorts, post_rot, pos_tran, theta_mats, T_ego_his2curs=None):
        b, m, t, n, c, h, w = x.shape
        x = rearrange(x, 'b,m,t,n,c,h,w -> (b,m,t,n),c,h,w')
        rots = rearrange(rots, 'b,m,t,n,h,w -> (b,m),t,n,h,w')
        trans = rearrange(trans, 'b,m,t,n,h,w -> (b,m),t,n,h,w')
        intrins = rearrange(intrins, 'b,m,t,n,h,w -> (b,m),t,n,h,w')
        distorts = rearrange(distorts, 'b,m,t,n,h,w -> (b,m),t,n,h,w')
        post_rot = rearrange(post_rot, 'b,m,t,n,h,w -> (b,m),t,n,h,w')
        pos_tran = rearrange(pos_tran, 'b,m,t,n,h,w -> (b,m),t,n,h,w')
        theta_mats = rearrange(theta_mats, 'b,m,t,h,w -> b,m,t,h,w')
        T_ego_his2curs = torch.eye(4).view(1, 1, 1, 4, 4).expand(b, m, t, 4, 4)

        return self.model(x)
