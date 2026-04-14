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
import random
from tqdm import tqdm
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


def set_seed(seed=42):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU
    # 确保卷积算法确定性（可能降低性能）
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # 可选：设置 Python 哈希种子
    os.environ['PYTHONHASHSEED'] = str(seed)

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

    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        local_rank = args.local_rank
    

    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        device = configs.get("device", "cuda")
        rank = 0
        world_size = 1


    set_seed(seed=42)


    model = Model(configs).to(device)
    if local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True
        )

    # BN层多卡数据共享均值方差
    num_gpus = torch.cuda.device_count()
    
    

    path_checkpoint = configs.get("pretrain", None)
    strict = False
    checkpoint = None
    if path_checkpoint is not None:
        checkpoint = torch.load(path_checkpoint, map_location=device)
        if "state_dict" not in checkpoint.keys():
            state_dict = checkpoint

            # model.load_state_dict(checkpoint, strict=False)
        else:
            state_dict = checkpoint["state_dict"]
        
        if local_rank != -1 and not any(k.startswith('module.') for k in state_dict):
                # 添加 'module.' 前缀
            state_dict = {'module.' + k: v for k, v in state_dict.items()}
        elif local_rank == -1 and any(k.startswith('module.') for k in state_dict):
            # 去除 'module.' 前缀
            state_dict = {k[7:]: v for k, v in state_dict.items()}

        try:
            model.load_state_dict(state_dict, strict=True)
            strict = True
        except Exception:
            model.load_state_dict(state_dict, strict=False)

    # 配置参数
    batch_size = configs["batch_size"]
    current_frame_index = configs["current_frame_index"]
    seq_len = configs["seq_len"]
    mini_batch = current_frame_index + 1
    batchidx  = batch_size -1
    cur_idx = seq_len-1
    steps_acc = mini_batch + 1 - seq_len

    epochs = configs["epoch"]
    load_types = configs["load_types"]
    types_names = sorted(list(load_types.keys()))
    # 优化器（可按需改成 AdamW 等）
    total_lr = configs.get("lr", 1e-3)          # config 中的 lr 视为总学习率
    if world_size > 1:
        lr = total_lr / world_size              # 单卡学习率 = 总学习率 / GPU 数量
    else:
        lr = total_lr
    weight_decay = configs.get("weight_decay", 1e-4)
    max_grad_norm = configs.get("max_grad_norm", 5)
    
    
    log_dir = os.path.join(os.path.dirname(script_path),configs["log_dir"])
    # os.makedirs(log_dir,exist_ok=True)
    # 日志文件名（带时间戳）
    # log_file = os.path.join(log_dir, f"train_{time.strftime('%Y%m%d_%H%M%S')}.log")
    if rank == 0:
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"train_{time.strftime('%Y%m%d_%H%M%S')}.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        writer = SummaryWriter(logdir=log_dir)
        config_path = os.path.join(os.path.dirname(script_path), "config/config.py")
        os.system(f"cp {config_path} {log_dir}")
    else:
        # 禁用非主进程的日志输出
        logging.basicConfig(level=logging.ERROR)


    active = []
    useful_type_names = []
    for i,type_name in enumerate(types_names):
        if load_types[type_name]:
            dataloader,sampler = build_dataloader(configs,mode=type_name)
            active.append((type_name, dataloader, sampler))
            useful_type_names.append(type_name)
        
    steps_per_epoch = max(len(dl) for _, dl, _ in active)
    logging.info(f"steps_per_epoch: {steps_per_epoch}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay,betas=(0.9,0.95))
    total_steps = epochs * steps_per_epoch
    warmup_steps = configs["warmup_steps"]# int(0.05 * total_steps) 
    scheduler = create_scheduler_with_warmup(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        base_lr=lr,
        eta_min=0
    )
    if rank == 0:
        logging.info(
            f"[train] epochs: {epochs}; steps_per_epoch: {steps_per_epoch}, "
            f"total_steps (optimizer steps): {total_steps}"
        )
        logging.info(f"Total LR (global): {total_lr:.2e}, Per-GPU LR: {lr:.2e}, world_size: {world_size}")
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
    
    # 
    # os.system(f"cp {config_path} {log_dir}")
    # writer = SummaryWriter(logdir=log_dir)
    
    # logging.info("Config:\n"+json.dumps(configs,indent=4))
    model.train()
    for epoch in range(start_epoch,epochs):
        np.random.seed()
        for _, _, sp in active:
            if sp is not None and hasattr(sp, "set_epoch"):
                sp.set_epoch(epoch)
        iters = [iter(dl) for _, dl, _ in active]
        
        for batch_idx in range(steps_per_epoch):
            # torch.cuda.empty_cache()

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
            
            input_names = configs["input_names"]  #["x","rots","trans","intrins","distorts","post_rots","post_trans","ego_poses"]
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
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            iteration += 1
            pre_bev = []
            for ii in tqdm(range(mini_batch), desc="Processing frames",leave=False):
                x = inputs_tensor["x"][:,:,ii:ii+1]
                rots = inputs_tensor["rots"][:,:,ii:ii+1]
                trans = inputs_tensor["trans"][:,:,ii:ii+1]
                intrins = inputs_tensor["intrins"][:,:,ii:ii+1]
                distorts = inputs_tensor["distorts"][:,:,ii:ii+1]
                post_rots = inputs_tensor["post_rots"][:,:,ii:ii+1]
                post_trans = inputs_tensor["post_trans"][:,:,ii:ii+1]
                ego_poses = inputs_tensor["ego_poses"][:,:,:ii+1][:,:,-1*seq_len:]

                metas_ii = {}
                for type,gt_infos in metas.items():
                    type_gts = {}
                    for key,gts in gt_infos.items(): # key,gts = gt_labels_det3D,gts
                        gts_batch = []
                        for gt in gts:
                            gts_batch.append([gt[ii]])
                        type_gts[key] = gts_batch
                    metas_ii[type] = type_gts

                
                if ii < cur_idx:
                    with torch.no_grad():
                        outputs = model(x, rots,trans, intrins,distorts,post_rots,post_trans,ego_poses,pre_bev=pre_bev,
                            metas=metas_ii,
                            decoder= False,
                            task_names=useful_type_names,
                            prepare_pre_bev = True
                        )
                else:
                    outputs = model(x, rots,trans, intrins,distorts,post_rots,post_trans,ego_poses,pre_bev=pre_bev,
                        metas=metas_ii,
                        decoder= True if iteration % log_save_interval == 0 else False,
                        task_names=useful_type_names,
                        prepare_pre_bev = False
                    )

                pre_bev.append(outputs["cur_bev"].detach())
                pre_bev = pre_bev[-1*(seq_len-1):]

                total_loss = outputs.get("total_loss", None)
                if total_loss is None:
                    continue


                scaled_loss = total_loss / steps_acc
                # if ii == mini_batch - 1:   # 最后一个 mb
                #     scaled_loss.backward()
                # else:
                #     with model.no_sync():
                #         scaled_loss.backward()

                if ii == mini_batch - 1:   # 最后一个 micro-batch
                    scaled_loss.backward()
                else:
                    if world_size > 1:
                        with model.no_sync():
                            scaled_loss.backward()
                    else:
                        scaled_loss.backward()
                    
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

                


            if rank == 0 and  iteration % log_print_interval == 0:
                current_lr = optimizer.param_groups[0]['lr']
                writer.add_scalar('epoch', epoch, iteration)
                writer.add_scalar('lr', current_lr, iteration)
                logging.info(f"Iteration [{iteration}] Epoch [{epoch+1}/{epochs}] Step [{batch_idx+1}/{steps_per_epoch}]; LR: {current_lr:.6f}; "
                    f"loss: {float(total_loss.detach().cpu()):.4f}")
                message = ""
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
                        message += f"{k}:{float(v.detach().cpu()):.4f}; \t"
                logging.info(message)


            if rank == 0 and iteration % log_save_interval == 0:
                # 可视化真值
                

                from dataset.dataset import denormalize_img
                import cv2
                imgs = inputs_tensor["x"][batchidx,:,cur_idx]
                for j, type_name in enumerate(type_names):
                    for i, img in enumerate(imgs[j]) :
                        img_np =  denormalize_img(img)
                        img_cv = cv2.cvtColor(np.array(img_np), cv2.COLOR_RGB2BGR)
                        # cv2.imwrite(f"{j}_{i}.jpg", img_cv)
                        writer.add_image(f'{type_name}/input_img/{i}', img_cv, global_step=iteration, dataformats='HWC')



                camera_names = configs["camera_names"]
                if "dynamic" in metas.keys():
                    meta = metas["dynamic"]
                    label_path = meta["label_path"][batchidx][cur_idx]# 最后一个batch的最后一帧(当前在)
                    gt_labels_det3D = meta["gt_labels_det3D"][batchidx][cur_idx].cpu().numpy()
                    gt_bboxes_det3D = meta["gt_bboxes_det3D"][batchidx][cur_idx].cpu().numpy()
                    gt_labels_det3D_mask = meta["gt_labels_det3D_mask"][batchidx][cur_idx].cpu().numpy()
                    gt_bboxes_det3D_mask = meta["gt_bboxes_det3D_mask"][batchidx][cur_idx].cpu().numpy()
                    if iteration == 10:
                        print(1)
                    
                    from utils.vis_gt import vis_dynamic_gt,vis_dynamic_pred
                    try:
                        gt_dynamic_canvas,gt_dynamic_images = vis_dynamic_gt(camera_names,label_path, gt_labels_det3D, gt_bboxes_det3D, gt_labels_det3D_mask, gt_bboxes_det3D_mask)
                    except:
                        print(1)
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
                    keys = gt_dynamic_images.keys()
                    for k in keys:
                        gt = cv2.cvtColor(gt_dynamic_images[k], cv2.COLOR_RGB2BGR)
                        pt = cv2.cvtColor(pt_dynamic_images[k], cv2.COLOR_RGB2BGR)
                        merge = np.concatenate([gt,pt],1)
                        writer.add_image(f'dynamic/gt_pt/{k}', merge, global_step=iteration, dataformats='HWC')
                    # for k,v in gt_dynamic_images.items():
                    #     writer.add_image(f'dynamic/gt/{k}', cv2.cvtColor(v, cv2.COLOR_RGB2BGR), global_step=iteration, dataformats='HWC')
                    # for k,v in pt_dynamic_images.items():  
                    #     writer.add_image(f'dynamic/pt/{k}', cv2.cvtColor(v, cv2.COLOR_RGB2BGR), global_step=iteration, dataformats='HWC')



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
                    keys = gt_static_images.keys()
                    for k in keys:
                        gt = cv2.cvtColor(gt_static_images[k], cv2.COLOR_RGB2BGR)
                        pt = cv2.cvtColor(pt_static_images[k], cv2.COLOR_RGB2BGR)
                        merge = np.concatenate([gt,pt],1)
                        writer.add_image(f'static/gt_pt/{k}', merge, global_step=iteration, dataformats='HWC')

                    # for k,v in gt_static_images.items():
                    #     writer.add_image(f'static/gt/{k}', cv2.cvtColor(v, cv2.COLOR_RGB2BGR), global_step=iteration, dataformats='HWC')
                    # for k,v in pt_static_images.items():
                    #     writer.add_image(f'static/pt/{k}', cv2.cvtColor(v, cv2.COLOR_RGB2BGR), global_step=iteration, dataformats='HWC')


            if rank == 0 and iteration % ckpt_save_interval == 0:
                model.eval()
                torch.save(model.state_dict(), os.path.join(log_dir,f"iter{iteration}.pth"))
                model.train()

            if world_size > 1:
                torch.distributed.barrier()

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
    # CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train_gpuxn.py