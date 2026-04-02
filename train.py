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
def create_scheduler_with_warmup(optimizer, warmup_steps, total_steps, base_lr, eta_min=0):
    # 定义 warmup 阶段的学习率因子
    def lambda_lr(step):
        if step < warmup_steps:
            return step / warmup_steps  # 线性增长到 1
        else:
            # 超过 warmup 后，因子保持 1，让 CosineAnnealingLR 接手
            return 1.0

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda_lr)
    # 余弦退火调度器（假设从 base_lr 开始衰减）
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps - warmup_steps, eta_min=eta_min
    )
    # 将两个调度器串联
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps]
    )
    return scheduler

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
    seq_len = configs["seq_len"]
    batch_size = configs["batch_size"]
    batchidx  = batch_size-1
    cur_idx = 0
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
    


    total_steps = epochs * steps_per_epoch * (configs["current_frame_index"] - seq_len + 1)
    logging.info(f"epochs: {epochs}; steps_per_epoch: {steps_per_epoch}, total_steps:{total_steps}")


    warmup_steps =  int(0.05 * total_steps) 
    scheduler = create_scheduler_with_warmup(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        base_lr=lr,
        eta_min=0
    )

    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer,T_max=steps_per_epoch*epochs,eta_min=0)

    iteration = -1
    start_epoch = 0
    if  strict and configs["resume"]:      
        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        scheduler.load_state_dict(checkpoint['lr_schedule'])  # 加载优化器参数
        iteration = checkpoint['iteration'] + 1
        start_epoch = checkpoint['epoch']

    
    log_print_interval = configs.get("log_print_interval", 10)
    log_save_interval = configs.get("log_save_interval", 100)
    ckpt_save_interval = configs.get("ckpt_save_interval", 100)
    
    config_path = os.path.join(os.path.dirname(script_path),"config/config.py")
    os.system(f"cp {config_path} {log_dir}")
    writer = SummaryWriter(logdir=log_dir)
    
    # logging.info("Config:\n"+json.dumps(configs,indent=4))
    model.train()
    for epoch in range(start_epoch,epochs):
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
            
            input_names = configs["input_names"]  
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

            min_batch_num = inputs_tensor["x"].shape[2]
            bev_feats = []
            for mb in range(min_batch_num):
                x = inputs_tensor["x"][:,:,mb:mb+1]    # torch.Size([2, 1, 5, 8, 3, 128, 384])
                rots = inputs_tensor["rots"][:,:,mb:mb+1]
                trans = inputs_tensor["trans"][:,:,mb:mb+1] # torch.Size([2, 1, 5, 8, 1, 3])
                intrins = inputs_tensor["intrins"][:,:,mb:mb+1]
                distorts = inputs_tensor["distorts"][:,:,mb:mb+1]
                post_rots = inputs_tensor["post_rots"][:,:,mb:mb+1]
                post_trans = inputs_tensor["post_trans"][:,:,mb:mb+1]
                ego_pose = inputs_tensor["ego_pose"][:,:,mb+1-seq_len:mb+1]
                if mb < seq_len -1 : # 0 1 2 3
                    with torch.no_grad():
                        outputs = model(x, rots, trans, intrins, distorts, post_rots, post_trans, [],
                            metas=None,
                            decoder= False,
                            task_names=useful_type_names,
                            pre_features=bev_feats,
                            pre_stream= True,
                        )
                else:
                    iteration +=1
                    metas_mb = {}
                    for tyepe_name,type_gt in metas.items():
                        new_type_gt = {}
                        for key,gts in type_gt.items():
                            bi_gts = []
                            for gt in gts:
                                bi_gts.append([gt[mb - (seq_len-1)]])
                            new_type_gt[key] = bi_gts
                        metas_mb[tyepe_name] = new_type_gt

                    outputs = model(x, rots, trans, intrins, distorts, post_rots, post_trans, ego_pose,
                        metas=metas_mb,
                        decoder= True if iteration % log_save_interval == 0 else False,
                        task_names=useful_type_names,
                        pre_features=bev_feats,
                        pre_stream= False,
                    )
                bev_feats.append(outputs["bev_feat"].detach())
                bev_feats = bev_feats[-1*(seq_len -1):]

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

                # iteration += 1


                if iteration % log_print_interval == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    writer.add_scalar('epoch', epoch, iteration)
                    writer.add_scalar('lr', current_lr, iteration)
                    logging.info(f"Iteration [{iteration}] Epoch [{epoch+1}/{epochs}] Step [{batch_idx+1}/{steps_per_epoch}]; LR: {current_lr:.6f}; "
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
                    

                    from dataset.dataset import denormalize_img
                    import cv2
                    imgs = x[batchidx,:,cur_idx].cpu()
                    for j, type_name in enumerate(type_names):
                        for i, img in enumerate(imgs[j]) :
                            img_np =  denormalize_img(img)
                            img_cv = cv2.cvtColor(np.array(img_np), cv2.COLOR_RGB2BGR)
                            # cv2.imwrite(f"{j}_{i}.jpg", img_cv)
                            writer.add_image(f'{type_name}/input_img/{i}', img_cv, global_step=iteration, dataformats='HWC')



                    camera_names = configs["camera_names"]
                    if "dynamic" in metas_mb.keys():
                        meta = metas_mb["dynamic"]
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
                                                                    score_thresh=0.3  # 可调
                                                                ) #img_cv = cv2.cvtColor(np.array(img_np), cv2.COLOR_RGB2BGR)
                        for k,v in gt_dynamic_images.items():
                            writer.add_image(f'dynamic/gt/{k}', cv2.cvtColor(v, cv2.COLOR_RGB2BGR), global_step=iteration, dataformats='HWC')
                        for k,v in pt_dynamic_images.items():  
                            writer.add_image(f'dynamic/pt/{k}', cv2.cvtColor(v, cv2.COLOR_RGB2BGR), global_step=iteration, dataformats='HWC')



                    if "static" in metas_mb.keys():
                        meta = metas_mb["static"]
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
                            writer.add_image(f'static/gt/{k}', cv2.cvtColor(v, cv2.COLOR_RGB2BGR), global_step=iteration, dataformats='HWC')
                        for k,v in pt_static_images.items():
                            writer.add_image(f'static/pt/{k}', cv2.cvtColor(v, cv2.COLOR_RGB2BGR), global_step=iteration, dataformats='HWC')


                if iteration % ckpt_save_interval == 0:
                    model.eval()
                    torch.save(model.state_dict(), os.path.join(log_dir,f"iter{iteration}.pth"))
                    model.train()

        # model.eval()
        # checkpoint = {
        #     "state_dict": model.state_dict(),
        #     'optimizer': optimizer.state_dict(),
        #     "epoch": epoch,
        #     "iteration": iteration,
        #     'lr_schedule': scheduler.state_dict()
        # }
        # torch.save(checkpoint, os.path.join(log_dir,f"iter{iteration}_epoch{epoch}.pth"))
        # model.train()
# 




if __name__ == "__main__":
    main()