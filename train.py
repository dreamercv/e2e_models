# -*- encoding: utf-8 -*-
'''
@File         :train.py
@Date         :2026/03/07 13:31:48
@Author       :Binge.Van
@E-mail       :1367955240@qq.com
@Version      :V1.0.0
@Description  :

'''
import os
import sys

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.utils.data.distributed as dist
import torch.nn.functional as F
import torch.nn.init as init
import torch.nn.utils.rnn as rnn_utils
import torch.nn.utils.spectral_norm as spectral_norm
import torch.backends.cudnn as cudnn

from config.config import configs
from dataset.dataset import build_dataloader

from model.models import Model

cudnn.deterministic = True
cudnn.benchmark = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.backends.cuda.matmul.allow_tf32=True
torch.backends.cudnn.allow_tf32=True
torch.multiprocessing.set_sharing_strategy('file_system')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default = -1, type=int)
    args = parser.parse_args()

    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device=torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')

    model = Model(configs).to(configs["device"])

    # # BN层多卡数据共享均值方差
    # num_gpus = torch.cuda.device_count()
    
    # # print("num_gpus:",num_gpus)
    # if num_gpus > 1:
    #     # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    #     model.to(device)
    #     model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    # else:
    #     model.to(device)



    # 配置参数
    epochs = configs["epoch"]
    load_types = configs["load_types"]
    types_names = sorted(list(load_types.keys()))
    #
    active = []
    for i,type_name in enumerate(types_names):
        if load_types[type_name]:
            dataloader,sampler = build_dataloader(configs,mode=type_name)
            active.append((type_name, dataloader, sampler))
        
    steps_per_epoch = max(len(dl) for _, dl, _ in active)
    print(f"steps_per_epoch: {steps_per_epoch}")

    for epoch in range(epochs):
        for _, _, sp in active:
            if sp is not None and hasattr(sp, "set_epoch"):
                sp.set_epoch(epoch)
        iters = [iter(dl) for _, dl, _ in active]
        
        for batch_idx in range(steps_per_epoch):
            batches = [] 
            for i, (type_name, dl, _) in enumerate(active):
                try:
                    batch = next(iters[i])
                except StopIteration:
                    iters[i] = iter(dl)
                    batch = next(iters[i])
                batches.append((type_name, batch))
            inputs_tensor = {} #bs, 2, 5, 8, 3, 128, 384
            gts_values = {}
            gt_names = configs["gt_names"]
            
            input_names = configs["input_names"]  #["x","rots","trans","intrins","distorts","post_rots","post_trans","theta_mats"]
            for type_name, batch in batches:
                for input_name in input_names:
                    if input_name not in inputs_tensor.keys():
                        inputs_tensor[input_name] = batch[input_name]
                    else:
                        inputs_tensor[input_name] = torch.stack([inputs_tensor[input_name],batch[input_name]],1)
                names = gt_names[type_name]
                gts_value = {}
                for name in names:
                    if name not in batch.keys():continue
                    gts_value[name] = batch[name]
                gts_values[type_name] = gts_value
            print()
            # "x",              3, 2, 5, 8, 3, 128, 384
            # "rots",           3, 2, 5, 8, 3, 3
            # "trans",          3, 2, 5, 8, 3       --> 1*3
            # "intrins",        3, 2, 5, 8, 3, 3
            # "distorts",       3, 2, 5, 8, 8       --> 1*8
            # "post_rots",      3, 2, 5, 8, 2, 2    --> 3*3
            # "post_trans",     3, 2, 5, 8, 2       --> 1*3
            # "theta_mats"      3, 2, 5,    2, 3
# 




main()
