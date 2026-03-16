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
import torch.backends.cudnn as cudnn
from torch import nn

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


from collections.abc import Mapping, Sequence

def recursive_to_device(data, device):
    """
    递归地将嵌套数据结构中的所有 torch.Tensor 转移到指定设备。
    支持 dict、list、tuple 以及它们的任意组合。
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, Mapping):  # 处理字典
        return {k: recursive_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, Sequence) and not isinstance(data, str):  # 处理列表/元组（排除字符串）
        return [recursive_to_device(item, device) for item in data]
    else:
        # 其他类型（int, float, str, None, np.ndarray 等）直接返回
        # 如果你希望将 numpy 数组也转为 Tensor 并移到设备，可以在这里添加相应逻辑
        return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default = -1, type=int)
    args = parser.parse_args()

    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device=torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')

    device = configs["device"]
    model = Model(configs).to(device)

    # BN层多卡数据共享均值方差
    num_gpus = torch.cuda.device_count()
    
    # print("num_gpus:",num_gpus)
    if num_gpus > 1:
        # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
        model.to(device)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    else:
        model.to(device)



    # 配置参数
    epochs = configs["epoch"]
    load_types = configs["load_types"]
    types_names = sorted(list(load_types.keys()))
    # 优化器（可按需改成 AdamW 等）
    lr = configs.get("lr", 1e-4)
    weight_decay = configs.get("weight_decay", 1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    active = []
    for i,type_name in enumerate(types_names):
        if load_types[type_name]:
            dataloader,sampler = build_dataloader(configs,mode=type_name)
            active.append((type_name, dataloader, sampler))
        
    steps_per_epoch = max(len(dl) for _, dl, _ in active)
    print(f"steps_per_epoch: {steps_per_epoch}")

    for epoch in range(epochs):
        model.train()
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
            inputs_tensor = {} # bs, M, T, 8, 3, 128, 384
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
                    if name not in batch.keys():
                        continue
                    gts_value[name] = batch[name]
                gts_values[type_name] = gts_value
            
            # 将输入搬到 device
            # device = configs["device"]
            for k, v in inputs_tensor.items():
                if isinstance(v, torch.Tensor):
                    inputs_tensor[k] = v.to(device)

            # 组装 metas（把 dynamic / static 的真值合并）
            metas = recursive_to_device(gts_values,device)
            # for type_name, gts in gts_values.items():
            #     for k, v in gts.items():
            #         # 将张量或张量列表搬到 device
            #         if isinstance(v, torch.Tensor):
            #             metas[k] = v.to(device)
            #         elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
            #             metas[k] = [t.to(device) for t in v]
            #         else:
            #             metas[k] = v

            outputs = model(
                inputs_tensor["x"],
                inputs_tensor["rots"],
                inputs_tensor["trans"],
                inputs_tensor["intrins"],
                inputs_tensor["distorts"],
                inputs_tensor["post_rots"],
                inputs_tensor["post_trans"],
                inputs_tensor["theta_mats"],
                metas=metas,
            )

            total_loss = outputs.get("total_loss", None)
            if total_loss is None:
                continue

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if batch_idx % configs.get("log_print_interval", 10) == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Step [{batch_idx+1}/{steps_per_epoch}] "
                      f"loss: {float(total_loss.detach().cpu()):.4f}")
            # "x",              3, 2, 5, 8, 3, 128, 384
            # "rots",           3, 2, 5, 8, 3, 3
            # "trans",          3, 2, 5, 8, 3       --> 1*3
            # "intrins",        3, 2, 5, 8, 3, 3
            # "distorts",       3, 2, 5, 8, 8       --> 1*8
            # "post_rots",      3, 2, 5, 8, 2, 2    --> 3*3
            # "post_trans",     3, 2, 5, 8, 2       --> 1*3
            # "theta_mats"      -    2, 3
# 




if __name__ == "__main__":
    main()
