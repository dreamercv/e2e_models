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
import json
import argparse


import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch import nn
from torch.optim import lr_scheduler

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

from tensorboardX import SummaryWriter


from collections.abc import Mapping, Sequence
import logging
import time

# 创建日志目录
# log_dir = configs["log_dir"]
# os.makedirs(log_dir, exist_ok=True)


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

    script_path = os.path.abspath(__file__)
    

    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        device=torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')

    device = configs["device"]
    model = Model(configs).to(device)

    # BN层多卡数据共享均值方差
    num_gpus = torch.cuda.device_count()
    
    # # print("num_gpus:",num_gpus)
    # if num_gpus > 1:
    #     # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    #     model.to(device)
    #     model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    # else:
    #     model.to(device)

    path_checkpoint = configs.get("pretrain",None)
    strict=False
    if path_checkpoint is not None:
        checkpoint = torch.load(path_checkpoint, map_location=device)  # 加载断
        if "state_dict" not in checkpoint.keys():
            model.load_state_dict(checkpoint, strict=False)
        else:
            try:
                model.load_state_dict(checkpoint["state_dict"], strict=True)
                strict = True
            except:
                model.load_state_dict(checkpoint["state_dict"], strict=False)

    # 配置参数
    epochs = configs["epoch"]
    load_types = configs["load_types"]
    types_names = sorted(list(load_types.keys()))
    # 优化器（可按需改成 AdamW 等）
    lr = configs.get("lr", 1e-3)
    weight_decay = configs.get("weight_decay", 1e-4)
    max_grad_norm = configs.get("max_grad_norm", 5)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay,betas=(0.9,0.95))
    
    log_dir = os.path.join(os.path.dirname(script_path),configs["log_dir"])
    os.makedirs(log_dir,exist_ok=True)
    # 日志文件名（带时间戳）
    log_file = os.path.join(log_dir, f"train_{time.strftime('%Y%m%d_%H%M%S')}.log")

    # 配置 logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        handlers=[
            logging.FileHandler(log_file),  # 输出到文件
            logging.StreamHandler()         # 输出到控制台
        ]
    )

    active = []
    useful_type_names = []
    for i,type_name in enumerate(types_names):
        if load_types[type_name]:
            dataloader,sampler = build_dataloader(configs,mode=type_name)
            active.append((type_name, dataloader, sampler))
            useful_type_names.append(type_name)
        
    steps_per_epoch = max(len(dl) for _, dl, _ in active)
    logging.info(f"steps_per_epoch: {steps_per_epoch}")

    scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=steps_per_epoch*epochs,eta_min=0)

    iteration = -1
    if  strict:      
        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        scheduler.load_state_dict(checkpoint['lr_schedule'])  # 加载优化器参数
        iteration = checkpoint['iteration'] + 1

    
    log_print_interval = configs.get("log_print_interval", 10)
    log_save_interval = configs.get("log_save_interval", 100)
    ckpt_save_interval = configs.get("ckpt_save_interval", 100)
    
    config_path = os.path.join(os.path.dirname(script_path),"config/config.py")
    os.system(f"cp {config_path} {log_dir}")
    writer = SummaryWriter(logdir=log_dir)
    
    logging.info("Config:\n"+json.dumps(configs,indent=4))
    model.train()
    for epoch in range(epochs):
        np.random.seed()
        for _, _, sp in active:
            if sp is not None and hasattr(sp, "set_epoch"):
                sp.set_epoch(epoch)
        iters = [iter(dl) for _, dl, _ in active]
        
        for batch_idx in range(steps_per_epoch):
            torch.cuda.empty_cache()

            batches = [] 
            type_names = []
            for i, (type_name, dl, _) in enumerate(active):
                try:
                    batch = next(iters[i])
                except StopIteration:
                    iters[i] = iter(dl)
                    batch = next(iters[i])
                batches.append((type_name, batch))
                type_names.append(type_name)
            inputs_tensor = {} # bs, M, T, 8, 3, 128, 384
            gts_values = {}
            gt_names = configs["gt_names"]
            
            input_names = configs["input_names"]  #["x","rots","trans","intrins","distorts","post_rots","post_trans","theta_mats"]
            for type_name, batch in batches:
                for input_name in input_names:
                    if input_name not in inputs_tensor.keys():
                        inputs_tensor[input_name] = batch[input_name][:,None]
                    else:
                        inputs_tensor[input_name] = torch.cat([inputs_tensor[input_name],batch[input_name][:,None]],1)
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
            metas = recursive_to_device(gts_values,device) #metas["dynamic"]["gt_labels_det3D"] =  list(2, 5, n*d)


            
            iteration += 1
            outputs = model(
                inputs_tensor["x"],     # torch.Size([2, 1, 5, 8, 3, 128, 384])
                inputs_tensor["rots"],
                inputs_tensor["trans"], # torch.Size([2, 1, 5, 8, 1, 3])
                inputs_tensor["intrins"],
                inputs_tensor["distorts"],
                inputs_tensor["post_rots"],
                inputs_tensor["post_trans"],
                inputs_tensor["theta_mats"],
                metas=metas,
                decoder= True if iteration % log_save_interval == 0 else False,
                task_names=useful_type_names
            )

            


            total_loss = outputs.get("total_loss", None)
            if total_loss is None:
                continue

            # optimizer.zero_grad()
            # total_loss.backward()
            # optimizer.step()

            total_loss.backward()
                
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            


            if iteration % log_print_interval == 0:
                writer.add_scalar('epoch', epoch, iteration)

                logging.info(f"Iteration [{iteration}] Epoch [{epoch+1}/{epochs}] Step [{batch_idx+1}/{steps_per_epoch}] "
                      f"loss: {float(total_loss.detach().cpu()):.4f}")
                message = "/t"
                writer.add_scalar('all_loss/loss', total_loss, iteration)
                losses = outputs.get("losses", None)
                if losses is not None:
                    for k,v in losses.items():
                        if k.startswith("det3d_"):
                            if "dn" in k:
                                writer.add_scalar(k.replace("det3d_","det3d/dn/"), v, iteration)
                            else:
                                writer.add_scalar(k.replace("det3d_","det3d/ori/"), v, iteration)
                        elif k.startswith("map3d_"):
                            writer.add_scalar(k.replace("map3d_","map3d/"), v, iteration)
                        message += f"{k}:{float(v.detach().cpu()):.4f}; "
                logging.info(message)


            if iteration % log_save_interval == 0:
                # 可视化真值
                seq_len = configs["seq_len"]
                batchidx  = configs["batch_size"]-1

                cur_idx = seq_len-1

                from dataset.dataset import denormalize_img
                import cv2
                imgs = inputs_tensor["x"][batchidx,:,cur_idx]
                for j, type_name in enumerate(type_names):
                    for i, img in enumerate(imgs[j]) :
                        img_np =  denormalize_img(img)
                        img_cv = cv2.cvtColor(np.array(img_np), cv2.COLOR_RGB2BGR)
                        cv2.imwrite(f"{j}_{i}.jpg", img_cv)
                        writer.add_image(f'{type_name}/input_img/{i}', img_cv, global_step=iteration, dataformats='HWC')



                camera_names = configs["camera_names"]
                if "dynamic" in metas.keys():
                    meta = metas["dynamic"]
                    label_path = meta["label_path"][batchidx][cur_idx]# 最后一个batch的最后一帧(当前在)
                    gt_labels_det3D = meta["gt_labels_det3D"][batchidx][cur_idx].cpu().numpy()
                    gt_bboxes_det3D = meta["gt_bboxes_det3D"][batchidx][cur_idx].cpu().numpy()
                    gt_labels_det3D_mask = meta["gt_labels_det3D_mask"][batchidx][cur_idx].cpu().numpy()
                    
                    from utils.vis_gt import vis_dynamic_gt,vis_dynamic_pred
                    gt_dynamic_canvas,gt_dynamic_images = vis_dynamic_gt(camera_names,label_path, gt_labels_det3D, gt_bboxes_det3D, gt_labels_det3D_mask, None)
                    
                    # 预测值
                    det3d_result = outputs["det3d_result"][-1]
                    boxes_3d = det3d_result["boxes_3d"].detach().cpu().numpy()
                    scores_3d = det3d_result["scores_3d"].detach().cpu().numpy()
                    labels_3d = det3d_result["labels_3d"].detach().cpu().numpy()
                    cls_scores = det3d_result["cls_scores"].detach().cpu().numpy()

                    pt_dynamic_canvas,pt_dynamic_images = vis_dynamic_pred(
                                                                camera_names=camera_names,  # 来自 config/config.py
                                                                label_path=label_path,
                                                                boxes_3d=boxes_3d,
                                                                scores_3d=scores_3d,
                                                                labels_3d=labels_3d,
                                                                score_thresh=0.3,  # 可调
                                                            )
                    for k,v in gt_dynamic_images.items():
                        writer.add_image(f'dynamic/gt/{k}', v, global_step=iteration, dataformats='HWC')
                    for k,v in pt_dynamic_images.items():  
                        writer.add_image(f'dynamic/pt/{k}', v, global_step=iteration, dataformats='HWC')



                if "static" in metas.keys():
                    meta = metas["static"]
                    label_path = meta["label_path"][batchidx][cur_idx]
                    gt_labels_map3D = meta["gt_labels_map3D"][batchidx][cur_idx].cpu().numpy()
                    gt_pts_map3D = meta["gt_pts_map3D"][batchidx][cur_idx].cpu().numpy()

                    from utils.vis_gt import vis_static_gt
                    gt_static_canvas,gt_static_images = vis_static_gt(camera_names,label_path,gt_labels_map3D,gt_pts_map3D)


                    # 2) 画预测
                    # map_polylines 是 decode 出的结果，通常是 list 长度 B*T
                    map_polylines = outputs["map3d_out"]["map_polylines"]
                    poly = map_polylines[batchidx * seq_len + cur_idx]
                    pred_labels = poly["labels"].detach().cpu().numpy()
                    if poly["pts"].dim() == 3:
                            pred_pts = poly["pts"].detach().cpu().numpy()
                    else:
                        # (N, num_orders, num_pts, 2)，取 order=0
                        pred_pts = poly["pts"][:, 0].detach().cpu().numpy()

                    pt_static_canvas,pt_static_images = vis_static_gt(
                                                            camera_names=camera_names,
                                                            label_path=label_path,
                                                            labels=pred_labels,
                                                            pts=pred_pts,
                                                        )

                    for k,v in gt_static_images.items():
                        writer.add_image(f'static/gt/{k}', v, global_step=iteration, dataformats='HWC')
                    for k,v in pt_static_images.items():
                        writer.add_image(f'static/pt/{k}', v, global_step=iteration, dataformats='HWC')


            if iteration % ckpt_save_interval == 0:
                model.eval()
                torch.save(model.state_dict(), os.path.join(log_dir,f"iter{iteration}.pth"))
                model.train()

        model.eval()
        checkpoint = {
            "state_dict": model.state_dict(),
            'optimizer': optimizer.state_dict(),
            "epoch": epoch,
            "iteration": iteration,
            'lr_schedule': scheduler.state_dict()
        }
        torch.save(checkpoint, os.path.join(log_dir,f"iter{iteration}_epoch{epoch}.pth"))
        model.train()
# 




if __name__ == "__main__":
    main()
    nn.MultiheadAttention
    from mmdet.models.losses import GaussianFocalLoss, L1Loss, CrossEntropyLoss