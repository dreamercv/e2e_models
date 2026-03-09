# -*- encoding: utf-8 -*-
'''
@File         :dataset.py
@Date         :2026/02/28 13:31:48
@Author       :Binge.Van
@E-mail       :afb5szh@bosch.com
@Version      :V1.0.0
@Description  :
    数据处理逻辑：
    一次forward需要91帧的数据：当前帧是第40帧，即历史是0-39，未来是41-90 # 历史40帧，未来50帧，再加当前帧
    任务包含：
    第一阶段2D图像表针学习 
    第二阶段BEV重建学习
    第三阶段BEV表征学习
        检测：             输入                36 37 38 39 40 
        地图:              输入                36 37 38 39 40
    第四阶段轨迹交互
        目标车轨迹：        输入     10 20 30               40   预测 50 60 70 80 90  --从动态拿instance
        端到端静态：        输入     10 20 30               40   预测 50 60 70 80 90  --从静态拿instance
        端到端动态：        输入     10 20 30               40   预测 50 60 70 80 90



由于数据不同源，所以训练逻辑如下：
    1、假设在dynamic数据上，可以先训练3d+目标轨迹，再训练端到端轨迹
    2、假设在static数据上，可以选训练map+静态轨迹，再训练端到端轨迹
    3、假设既要3d表征和map表征以及目标轨迹和静态轨迹，分阶段训练：
        3.1、先训练3d表征和目标轨迹
        3.2、再训练map表征和静态轨迹
        3.3、将上述输入到端到端，最后训练端到端轨迹
    4、假设想要端到端的训练(图像输入，轨迹输出)，那么就是同源数据，直接训练即可
'''


import os,sys
import json
import cv2
import numpy as np

import torchvision

import torch
import random

# configs = {
#     # 总的参数
#     "grid_conf" : {
#         'xbound': [-80.0, 120.0, 1],
#         'ybound': [-40.0, 40.0, 1],
#         'zbound': [-2.0, 4.0, 1.0]
#     },

#     "clip_paths":{
#         "dynamic":["/home/fb/project/models/Sparse4D-main/projects/e2e_models/dataset/clip_dataset/det.txt"],
#         "static":["/home/fb/project/models/Sparse4D-main/projects/e2e_models/dataset/clip_dataset/map.txt"],
#         "dynamic_static":["/home/fb/project/models/Sparse4D-main/projects/e2e_models/dataset/clip_dataset/dynamic_static.txt"],
#         "e2e":["/home/fb/project/models/Sparse4D-main/projects/e2e_models/dataset/clip_dataset/e2e.txt"],
#     },
#     "camera_infos":{
#         "FrontCam02":{
#             "resize_lim": [(0.1, 0.1),  (0.17,0.23), (0.35, 0.45)],
#             "bot_pct_lim":[(0.15, 0.25),(0.27, 0.37),(0.37,0.47)]
#         },     #前 大
#         "RearCam01":     {"resize_lim":[(0.2, 0.3)],"bot_pct_lim":[(0.2, 0.5)]},      #后方
#         "SideFrontCam01":{"resize_lim":[(0.2, 0.3)],"bot_pct_lim":[(0.2, 0.5)]}, #左前
#         "SideFrontCam02":{"resize_lim":[(0.2, 0.3)],"bot_pct_lim":[(0.2, 0.5)]}, #右前
#         "SideRearCam01": {"resize_lim":[(0.2, 0.3)],"bot_pct_lim":[(0.2, 0.5)]},  #左后
#         "SideRearCam02": {"resize_lim":[(0.2, 0.3)],"bot_pct_lim":[(0.2, 0.5)]}   #右后
#     },
#     'rot_lim': (-0.0, 0.0),
#     "flip":False,

#     'final_dim': (128, 384),

#     "mode":"dynamic",


#     "train_clips":2, # -1 所有，[]指定几个，xxx.txt写进txt中指定的，>0前N个
    
#     "total_len":20+1+50, # 一共71帧，当前帧是第21帧，往前20帧，往后50帧 ，最后一帧索引对应着70 ，未来50帧为了获取真值
#     "current_frame_index":20,# 0 1 2 3 4 5 ... [20] 21 22 23 ... 70
    


#     "task_flag":{
#         "det2D":False,
#         "det3D":True,

#         "map2D":False,
#         "map3D":False,

#         "obj_dynamic_traj":False,
#         "e2e_static_traj":False,
#         "e2e_dynamic_traj":False,
#     },
#     "task_class_names":{
#         "dynamic":["det3D","obj_dynamic_traj","det2D","e2e_dynamic_traj"], 
#         "static":["map3D","map2D","e2e_static_traj","e2e_dynamic_traj"],
#         "dynamic_static":[
#             "det3D","obj_dynamic_traj","det2D",
#             "map3D","map2D","e2e_static_traj",
#             "e2e_dynamic_traj"
#             ],
#         "e2e":[
#             "e2e_static_traj",
#             "e2e_dynamic_traj"
#             ],
#     },
#     "task_indexs":{ # 在每次getitem时，随机取值
#         # 在dyo的label上取值
#         "det2D":[0,20], # 起始帧和结束帧                #输入N帧输出N帧
#         "det3D":  [0,20], # 起始帧和结束帧              #输入N帧输出N帧
#         "obj_dynamic_traj": [0,20], # 起始帧和结束帧    #输入N帧，预测最后一帧的轨迹
#         # 在lane的label上取值
#         "map2D":[0,20], # 起始帧和结束帧                #输入N帧输出N帧
#         "map3D":  [0,20], # 起始帧和结束帧              #输入N帧输出N帧

#         # 在自车位姿上取值
#         "e2e_dynamic_traj": [0,20], # 起始帧和结束帧    #输入N帧，预测最后一帧的轨迹
#         "e2e_static_traj":  [0,20], # 起始帧和结束帧    #输入N帧，预测最后一帧的轨迹
#     },
#     "seq_len":5, 
#     # 预测任务的参数
#     "his_lens":{ # 历史长度 ,对于预测任务来说，需要输入历史真值才能预测未来真值，所以上述task_indexs需要减去历史长度
#         "obj_dynamic_traj":5, # seq_len 必须大于等于 his_lens 否则无法制作真值
#         "e2e_dynamic_traj":5,
#     },
#     "fur_lens":{ # 未来长度
#         "obj_dynamic_traj": 50,
#         "e2e_dynamic_traj": 50,
#     }, # his_lens , fur_lens 即输入历史真值预测未来轨迹
#     "frequency":{ # 频率 10Hz
#         "obj_dynamic_traj": 10, # 真值插值成10Hz,即5秒内真值为50个点
#         "e2e_dynamic_traj": 10, # 真值插值成10Hz,即5秒内真值为50个点
#     },





#     "camera_names":[
#         "FrontCam02",     #前 
#         "RearCam01",      #后方
#         "SideFrontCam01", #左前
#         "SideFrontCam02", #右前
#         "SideRearCam01",  #左后
#         "SideRearCam02"   #右后
#     ],
#     # 3D 检测类别：与 convert_det 输出的 object_infos[].sub_category 对应，用于映射到类别下标
#     "det_class_names": ["car", "truck", "bus", "pedestrian", "bicycle", "motorcycle", "traffic_cone", "barrier"],
# }



def quaternion_to_rotation_matrix( x: float, y: float, z: float,w: float) -> np.ndarray:
    R = np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]
    ])
    
    return R

def TransformationmatrixEgo(orientation,position):
    w,x,y,z = orientation
    rotation_matrix = quaternion_to_rotation_matrix(x,y,z,w)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = position
    return transform


def get_rot(h):
    return torch.Tensor([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])

normalize_img = torchvision.transforms.Compose((
                torchvision.transforms.ToTensor(),
                torchvision.transforms.GaussianBlur(kernel_size=(1,5), sigma=(0.1, 1)),  # 随机选择的高斯模糊模糊图像
                torchvision.transforms.ColorJitter(brightness=(0.8,2), contrast=(0.8,2), saturation=(0.2,2), hue=(-0.2,0.2)),
                torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ))

def img_transform(img,
                  resize, resize_dims, crop,
                  flip=False, rotate=0):
    post_rot = torch.eye(2)
    post_tran = torch.zeros(2)

    W,H = resize_dims  # 381 -- > (468, 374)
    img = cv2.resize(img, (W,H))
    img = img[crop[1]: crop[3], crop[0]: crop[2]]  #(112, 240, 87, 471)
    if flip:
        img = cv2.flip(img, 1)
    h, w, _ = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), rotate, 1)
    img = cv2.warpAffine(img, M, (w, h))
    post_rot *= resize
    post_tran -= torch.Tensor(crop[:2])

    if flip:
        A = torch.Tensor([[-1, 0], [0, 1]])
        b = torch.Tensor([crop[2] - crop[0], 0])
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b
    A = get_rot(rotate/180*np.pi)
    b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
    b = A.matmul(-b) + b
    post_rot = A.matmul(post_rot)
    post_tran = A.matmul(post_tran) + b
    post_rot3 = torch.eye(2)
    post_tran3 = torch.zeros(2)
    post_tran3[:2] = post_tran
    post_rot3[:2, :2] = post_rot
    return img, post_rot3, post_tran3

class Dataset(torch.utils.data.Dataset):
    def __init__(self,config,mode="dynamic"):
        self.config = config
        self.mode = mode if mode is not None else config.get("mode","dynamic") # 默认加载动态数据
        self.is_train = config.get("is_train",True) # 默认训练

        self.grid_conf = config["grid_conf"] # bev网格配置
        self.bevh = self.grid_conf["xbound"][1] - self.grid_conf["xbound"][0]
        self.bevw = self.grid_conf["ybound"][1] - self.grid_conf["ybound"][0]

        # 相机
        self.cns = config["camera_names"]#使用几个相机
        self.cnis = config["camera_infos"]#每个相机增强参数

        self.fH, self.fW  = config["final_dim"] # 网络输入的大小

        self.total_len =config["total_len"]
        self.current_frame_index = config["current_frame_index"]

        
        
        

        # 联合训练配置
        self.task_flags = config["task_flag"]
        self.task_indexs = config["task_indexs"]
        self.task_class_names = config["task_class_names"]
        self.task_index_random = config["task_index_random"]
        self.seq_len = config["seq_len"]
        self.his_lens = config["his_lens"]
        self.fur_lens = config["fur_lens"]
        self.frequency = config["frequency"]
        # self.useful_indexs,self.group_indexs = self.get_input_indexs()

        # 数据准备
        self.get_all_paths()
        self.prepro()
        self.sces_len = [(len(self.ixes[i]) - self.total_len ) for i in self.ixes.keys()]
        self.scenes = [i for i in self.ixes.keys()]

        # 检测配置
        self.det_class_names = config["det_class_names"]
        self.det_gt_names = config["det_gt_names"]

    def gen_aug_params(self,H,W,resize_lim,bot_pct_lim,rot_lim):
        fH, fW = self.fH, self.fW
        if self.is_train:
            resize = np.random.uniform(*resize_lim)
        else:
            resize = np.mean(resize_lim)
        resize_dims = (int(W * resize), int(H * resize))
        newW, newH = resize_dims
        # 保证 resize 后至少能裁出 (fW, fH)，否则用下限
        if newW < fW or newH < fH:
            scale = max(fW / max(newW, 1), fH / max(newH, 1))
            resize = min(1.0, resize * scale)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
        if self.is_train:
            crop_h = int((1 - np.random.uniform(*bot_pct_lim)) * newH) - fH
        else:
            crop_h = int(np.mean(bot_pct_lim) * newH) - fH
        crop_w = int(max(0, newW - fW) / 2)
        crop_h = int(np.clip(crop_h, 0, max(0, newH - fH)))
        crop_w = int(np.clip(crop_w, 0, max(0, newW - fW)))
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        if self.is_train:
            rotate = np.random.uniform(*rot_lim)
        else:
            rotate = 0.
        return resize,resize_dims,crop,rotate
        

    def get_image_data(self,recs):
        "输出增强后的图片(裁剪/旋转)，以及增强的参数，相关的内外惨"
        "输出：images, rots, trans, intrins, distorts, post_rots, post_trans"
        imgs, rots, trans, intrins, distorts, post_rots, post_trans = [],[],[],[],[],[],[]
        cnis = {}
        for i, label_path in enumerate(recs):
            imgs_i, rots_i, trans_i, intrins_i, distorts_i, post_rots_i, post_trans_i = [],[],[],[],[],[],[]
            json_data = json.load(open(label_path,"r"))
            image_paths = json_data["paths"]
            parameters = json_data["parameters"]
            for cn in self.cns:
                # 获取相关参数
                parameter = parameters[cn]
                lidar2camera = np.array(parameter["lidar2camera"]) # 外参
                distort = np.array(parameter["dist_coeffs"])#畸变
                intrin = np.array(parameter["camera_matrix"]) #内参
                rot = lidar2camera[:3,:3]
                tran =  lidar2camera[:3,3][None]
                # 获取图像信息
                image_name = os.path.basename(image_paths[cn])
                image_path = os.path.join(
                    os.path.dirname(label_path).replace("label",cn),
                    image_name
                )
                # image_path = label_path.replace("label",cn).replace(".json",".jpg")
                #如果图像不存在的话，给置成全0黑图，参数就全部初始化一下----这里默认所有视角的图片都存在
                # 数据增强的参数
                resize_lims,bot_pct_lims,rot_lim,flip = self.cnis[cn]["resize_lim"],self.cnis[cn]["bot_pct_lim"],self.config["rot_lim"],self.config["flip"]
                
                if i == 0: #同一getitem中的数据裁剪和旋转要相同
                    oriH,oriW = parameter["image_height"],parameter["image_width"]
                    for j in range(len(resize_lims)):
                        resize,resize_dims,crop,rotate = self.gen_aug_params(oriH,oriW,resize_lims[j],bot_pct_lims[j],rot_lim)
                        if cn in cnis.keys():
                            cnis[cn].append([resize,resize_dims,crop,rotate,flip])
                        else:
                            cnis[cn] = [[resize,resize_dims,crop,rotate,flip]]

                img = cv2.imread(image_path)
                for aug_param in cnis[cn]:
                    resize, resize_dims, crop, rotate,flip = aug_param
                    post_img, post_rot, post_tran = img_transform(img,resize, resize_dims, crop,flip=flip, rotate=rotate)
                    
                    imgs_i.append(normalize_img(post_img))
                    post_rots_i.append(post_rot)
                    post_trans_i.append(post_tran)
                    intrins_i.append(torch.Tensor(intrin))
                    distorts_i.append(torch.Tensor(distort)[None])
                    rots_i.append(torch.Tensor(rot))
                    trans_i.append(torch.Tensor(tran))
            intrins.append(torch.stack(intrins_i))
            imgs.append(torch.stack(imgs_i))
            distorts.append(torch.stack(distorts_i))
            rots.append(torch.stack(rots_i))
            trans.append(torch.stack(trans_i))
            post_rots.append(torch.stack(post_rots_i))
            post_trans.append(torch.stack(post_trans_i))
        return torch.stack(imgs), torch.stack(rots), torch.stack(trans), \
                torch.stack(intrins), torch.stack(distorts), torch.stack(post_rots), torch.stack(post_trans)

    def get_anno_det2D(self,recs):
        return {
            "gt_labels_det2D":None,
            "gt_bboxes_det2D":None
        }

    def get_anno_map2D(self,recs):
        return {
            "gt_labels_map2D":None,
            "gt_bboxes_map2D":None
        }

    def is_occupy_sample(self,in_cameras):
        is_no_occ = 0
        for cam,anno2d in in_cameras.items():
            if anno2d["occupy"].isdigit() and int(anno2d["occupy"]) < 2:
                is_no_occ = 1
            if is_no_occ==1:
                break
        return is_no_occ

        
    def get_anno_det3D(self,recs,dt=0.1):
        """
        从 rec 中每帧的 label json 读取 object_infos，转为 docs/DATA_FORMAT 约定的 3D 检测真值。
        与 convert_det.py 输出的标准格式一致：object_infos[track_id] = {
            "category", "sub_category", "position": {x,y,z}, "rotation": {yaw,...}, "dimension": {width, length, height}
        }
        Returns:
            gt_labels_per_frame: list of length T，每元素 (N_t,) long，类别下标
            gt_bboxes_per_frame: list of length T，每元素 (N_t, 10)，解码格式 [x,y,z,w,l,h,yaw,vx,vy,vz]
            yaw 为弧度；无速度填 0。
        """
        gt_labels_per_frame = []
        gt_bboxes_per_frame = []
        gt_labels_per_frame_mask = []
        gt_bboxes_per_frame_mask = []
        class_names = self.det_class_names #self.config.get("det_class_names", ["car", "truck", "bus", "pedestrian", "bicycle"])
        il = self.det_gt_names
        for label_path in recs:
            obj_traj_path = label_path.replace(".json","_object_trajectory.npy")
            if os.path.exists(obj_traj_path):
                object_trajectorys =  np.load(obj_traj_path, allow_pickle=True).item()
            else:
                object_trajectorys = {}
            
            with open(label_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            object_infos = data.get("object_infos", {})
            labels_list = []
            bboxes_list = []
            labels_maks_list,bboxes_mask_list = [],[]
            for track_id, obj in object_infos.items():
                label_mask,bboxes_mask = 1,[1 for i in range(10)]
                pos = obj.get("position")
                rot = obj.get("rotation")
                dim = obj.get("dimension")
                if pos is None or rot is None or dim is None:
                    continue
                sub = obj.get("sub_category","unkonw")
                if sub not in class_names:continue
                #label_mask 全样本逻辑：比如遮挡多少比例不能作为正样本
                label_mask = self.is_occupy_sample(obj.get("in_cameras",{}))

                cls_id = class_names.index(sub)
                x = self.gt_value("x",pos,bboxes_mask,0)# float(pos.get("x", 0))
                y = self.gt_value("y",pos,bboxes_mask,1)#float(pos.get("y", 0))
                z = self.gt_value("z",pos,bboxes_mask,2)#float(pos.get("z", 0))
                w = self.gt_value("width",dim,bboxes_mask,3)#float(dim.get("width", 0))
                length = self.gt_value("length",dim,bboxes_mask,4)#float(dim.get("length", 0))
                h = self.gt_value("height",dim,bboxes_mask,5)# float(dim.get("height", 0))
                yaw = self.gt_value("yaw",rot,bboxes_mask,6)#float(rot.get("yaw", 0.0))
                vx,vy,vz = self.diff(object_trajectorys,track_id,bboxes_mask,dt=dt,idx=7)
                bboxes_list.append([x, y, z, w, length, h, yaw, vx,vy,vz])
                labels_list.append(cls_id)
                labels_maks_list.append(label_mask)
                bboxes_mask_list.append(bboxes_mask)
            if len(labels_list) == 0:
                gt_labels_per_frame.append(torch.zeros(0, dtype=torch.long))
                gt_bboxes_per_frame.append(torch.zeros(0, 10, dtype=torch.float32))
                gt_labels_per_frame_mask.append(torch.zeros(0, dtype=torch.long))
                gt_bboxes_per_frame_mask.append(torch.zeros(0, 10, dtype=torch.float32))
            else:
                gt_labels_per_frame.append(torch.tensor(labels_list, dtype=torch.long))
                gt_bboxes_per_frame.append(torch.tensor(bboxes_list, dtype=torch.float32))
                gt_labels_per_frame_mask.append(torch.tensor(labels_maks_list, dtype=torch.long))
                gt_bboxes_per_frame_mask.append(torch.tensor(bboxes_mask_list, dtype=torch.float32))
        # return gt_labels_per_frame, gt_bboxes_per_frame
        return {
            "gt_labels_det3D":gt_labels_per_frame,
            "gt_bboxes_det3D":gt_bboxes_per_frame,
            "gt_labels_det3D_mask":gt_labels_per_frame_mask,
            "gt_bboxes_det3D_mask":gt_bboxes_per_frame_mask,
        }

    def get_anno_map3D(self, recs):
        """
        将 clip_dataset/map 中的静态地图标注转换为 MapTR 所需的 3D 地图真值。
        参考 demo_map.md / MAP_DATA_FORMAT.md：
        - 仅使用当前帧（默认取 recs[-1]）的地图标注；
        - 从 groups 中抽取折线/多边形，投影到 ego 坐标，并裁剪到 pc_range(grid_conf.xbound/ybound) 内；
        - 生成：
          * gt_labels_map3D / gt_bboxes_map3D（方便调试）；
          * gt_map_bboxes / gt_map_labels / gt_map_pts（MapHead.loss 直接使用）。
        """
        gt_labels_seq = []
        gt_bboxes_seq = []
        gt_pts_seq = []
        for label_path in recs:
        # 只取当前帧（最后一帧）的 map 标注
            label_path = recs[-1]
            with open(label_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            ego_pose = data["ego_pose"]
            T_ego2wld = TransformationmatrixEgo(ego_pose["orientation"], ego_pose["position"])
            T_wld2ego = np.linalg.inv(T_ego2wld)

            groups = data.get("groups", {})

            # pc_range（BEV 范围）
            x_min, x_max = self.grid_conf["xbound"][0], self.grid_conf["xbound"][1]
            y_min, y_max = self.grid_conf["ybound"][0], self.grid_conf["ybound"][1]

            # MapTR 中常用的 3 类：0=divider, 1=ped_crossing, 2=boundary
            polylines = []      # 每条线的 (num_pts, 2)
            labels = []         # 每条线的类别下标
            geom_types = []     # 每条线对应的几何类型：line_3d / polygon_3d

            # 统一的重采样点数 & 多顺序个数
            # 按论文思路：开曲线至少正/反 2 种，闭合折线可有 N*2 种（N=num_pts）
            # 这里对所有线统一使用 num_orders = 2 * num_pts：
            #   - 前 num_pts 个顺序：正向循环移位
            #   - 后 num_pts 个顺序：反向循环移位
            num_pts = self.config.get("map_num_pts", 20)
            num_orders = 2 * num_pts

            def resample_polyline(coords_xy, num_pts):
                """按均匀参数 t∈[0,1] 对折线做一维线性插值，得到固定点数。"""
                n = coords_xy.shape[0]
                if n <= 1 or n == num_pts:
                    return coords_xy.astype(np.float32)
                t_old = np.linspace(0.0, 1.0, n, dtype=np.float32)
                t_new = np.linspace(0.0, 1.0, num_pts, dtype=np.float32)
                x_new = np.interp(t_new, t_old, coords_xy[:, 0])
                y_new = np.interp(t_new, t_old, coords_xy[:, 1])
                return np.stack([x_new, y_new], axis=1).astype(np.float32)

            for _, grp in groups.items():
                gtype = grp.get("type", "")
                props = grp.get("properties", {})

                # 映射到 3 类：divider / ped_crossing / boundary
                cls_id = None
                if gtype in ["lane_line", "real_lane_line", "imaginary_lane_line", "lane_center_line"]:
                    cls_id = 0  # divider
                elif gtype == "road_marker_line":
                    # 仅把 category==3 (斑马线) 当作人行横道，其它暂不作为 map GT
                    cat = str(props.get("category", "-1"))
                    if cat == "3":
                        cls_id = 1  # ped_crossing
                elif gtype in ["road_edge", "non_drivable_area"]:
                    cls_id = 2  # boundary / non-drivable
                else:
                    continue

                if cls_id is None:
                    continue

                objects = grp.get("objects", [])
                for obj in objects:
                    # 仅使用 LidarFusion 的 3D 几何
                    if obj.get("sensor", "") != "LidarFusion":
                        continue
                    geom = obj.get("geometry", "")
                    if geom not in ["line_3d", "polygon_3d"]:
                        continue

                    pts3d = obj.get("points", [])
                    if len(pts3d) < 2:
                        continue

                    coords = []
                    for p in pts3d:
                        pw = np.array([p["x"], p["y"], p["z"], 1.0], dtype=np.float64)
                        pe = (T_wld2ego @ pw)[:3]  # ego 坐标
                        coords.append([pe[0], pe[1]])  # 只保留 x,y
                    coords = np.asarray(coords, dtype=np.float32)  # (N, 2)

                    # 与 pc_range 的粗裁剪：若整条线都在范围外，则忽略
                    xs, ys = coords[:, 0], coords[:, 1]
                    if xs.max() < x_min or xs.min() > x_max or ys.max() < y_min or ys.min() > y_max:
                        continue

                    # 将点裁剪到 pc_range 内，避免 bbox 超界
                    coords[:, 0] = np.clip(coords[:, 0], x_min, x_max)
                    coords[:, 1] = np.clip(coords[:, 1], y_min, y_max)

                    # 重采样为固定 num_pts
                    coords_rs = resample_polyline(coords, num_pts)  # (num_pts, 2)

                    polylines.append(coords_rs)
                    labels.append(cls_id)
                    geom_types.append(geom)

            if len(polylines) == 0:
                gt_labels = torch.zeros(0, dtype=torch.long)
                gt_bboxes = torch.zeros(0, 4, dtype=torch.float32)
                # 多顺序 GT：形状 (N, num_orders, num_pts, 2)
                gt_pts = torch.zeros(0, num_orders, num_pts, 2, dtype=torch.float32)
            else:
                polys = np.stack(polylines, axis=0)  # (N, num_pts, 2)
                labels_arr = np.asarray(labels, dtype=np.int64)  # (N,)

                x1 = polys[:, :, 0].min(axis=1)
                y1 = polys[:, :, 1].min(axis=1)
                x2 = polys[:, :, 0].max(axis=1)
                y2 = polys[:, :, 1].max(axis=1)
                bboxes_arr = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)  # (N, 4)

                gt_labels = torch.from_numpy(labels_arr).long()
                gt_bboxes = torch.from_numpy(bboxes_arr).float()

                # 多顺序 GT：分开/闭合两种情况
                # - 开曲线（line_3d）：只需要正序 + 反序 2 种表示，其余 order 用重复填充占位
                # - 闭合折线（polygon_3d）：正序 + 反序 + 循环移位，共 2*num_pts 种表示
                all_orders = []
                for poly, gtype in zip(polys, geom_types):
                    pts = torch.from_numpy(poly).float()  # (num_pts, 2)
                    orders = []
                    if gtype == "polygon_3d":
                        # 闭合折线：正向 & 反向的所有循环移位 → N*2 种
                        for k in range(num_pts):
                            orders.append(torch.roll(pts, shifts=-k, dims=0))
                        pts_rev = torch.flip(pts, dims=[0])
                        for k in range(num_pts):
                            orders.append(torch.roll(pts_rev, shifts=-k, dims=0))
                    else:
                        # 开曲线：只保留正序 + 反序两种表示，其它 order 用正序重复填充
                        orders.append(pts)                    # 正序
                        orders.append(torch.flip(pts, [0]))   # 反序
                        while len(orders) < num_orders:
                            orders.append(pts)
                    orders = torch.stack(orders, dim=0)  # (num_orders, num_pts, 2)
                    all_orders.append(orders)
                gt_pts = torch.stack(all_orders, dim=0)  # (N, num_orders, num_pts, 2)
            gt_labels_seq.append(gt_labels)
            gt_bboxes_seq.append(gt_bboxes)
            gt_pts_seq.append(gt_pts)
        # map3D 内部使用（可选）
        out = {
            "gt_labels_map3D": gt_labels_seq,
            "gt_bboxes_map3D": gt_bboxes_seq,
            "gt_pts_map3D": gt_pts_seq,
        }
        # MapHead.loss 直接使用的格式（单样本 → list 长度 1）
        # out.update({
        #     "gt_map_bboxes": [gt_bboxes],
        #     "gt_map_labels": [gt_labels],
        #     "gt_map_pts": [gt_pts],
        # })
        return out
    
    def get_anno_obj_dynamic_traj(self, recs, dt=0.1):
        """
        从 object_trajectory_in_ego 中获取目标在自车坐标系下的动态轨迹，并做插值补全：
        - mask=-1: 该帧目标消失
        - mask=0: 该帧为“当前帧”
        - mask=1: 该帧存在目标
        对于每个轨迹，在第一次出现 1 和最后一次出现 1 之间，将 mask=-1 的位置用线性插值
        补全 xyz，并将这些位置的 mask 置为 1。
        然后在当前帧附近截取一个时间窗（默认前 40 帧、后 50 帧），计算速度轨迹。
        """


        label_path = recs[-1]  # 只选择最后一帧作为真值
        obj_traj_path = label_path.replace(".json", "_object_trajectory.npy")
        if os.path.exists(obj_traj_path):
            object_trajectorys = np.load(obj_traj_path, allow_pickle=True).item()
        else:
            object_trajectorys = {}

        trackids = []
        obj_trajs = []
        masks = []

        for trackid, trajs in object_trajectorys.items():
            traj_mask = trajs["obj_trajectory_in_ego"]  # (T, 4), [:3] 为 xyz, 最后一维为 mask
            coords_full = traj_mask[:, :3].astype(np.float32)  # (T, 3)
            mask_full = traj_mask[:, -1].astype(np.float32)    # (T,)

            # 当前帧索引（mask==0），若不存在则跳过该轨迹
            cur_idx_arr = np.where(mask_full == 0)[0]
            if cur_idx_arr.size == 0:
                continue
            cur_idx = int(cur_idx_arr[0])

            T = coords_full.shape[0]
            start_idx = max(0, cur_idx - self.his_lens["obj_dynamic_traj"])
            end_idx = min(T - 1, cur_idx + self.fur_lens["obj_dynamic_traj"])

            coords = coords_full[start_idx:end_idx + 1].copy()  # (L, 3)
            mask = mask_full[start_idx:end_idx + 1].copy()      # (L,)
            mask[mask==0]=1
            # 在当前窗口内，找第一次和最后一次出现 1 的位置
            one_idx = np.where(mask == 1)[0]
            if one_idx.size >= 2:
                s = int(one_idx[0])
                e = int(one_idx[-1])
                x = np.arange(s, e + 1, dtype=np.float32)
                # 有效点：mask>=0（包括 0 和 1）
                valid = mask[s:e + 1] >= 0
                x_valid = x[valid]
                if x_valid.size >= 2:
                    for d in range(3):
                        y_seg = coords[s:e + 1, d]
                        y_valid = y_seg[valid]
                        # 用相邻有效点对 -1 位置做线性插值
                        y_interp = np.interp(x, x_valid, y_valid)
                        missing = mask[s:e + 1] < 0
                        coords[s:e + 1, d][missing] = y_interp[missing]
                    # 被插值的位置视作存在目标
                    mask[s:e + 1][mask[s:e + 1] < 0] = 0.
            #插值完后，被插值的地方标记为0，不需要插值的地方标记为1，剩余地方没法插值丢弃
            # 计算速度：一阶差分 / dt，长度为 L-1
            traj_v = (coords[1:] - coords[:-1]) / dt # 当前+历史=5帧，未来是50帧
            mask[s] = -1.
            mask = mask[1:] #和traj_v保持一致
            trackids.append(trackid)
            obj_trajs.append(traj_v)
            masks.append(mask.tolist())

            # 15 16 17 18 19 [20] 21 22 23 24 25
            # -1 -1  1  1  1   1   1  1  0  0  0

            #-1  1  1  1   1   1  1  0  0  0

            #16 17 18 19 [20] 21 22 23 24 25
            #-1  1  1  1   1   1  1  0  0  0
            
            #15 16 17 18 19 [20] 21 22 23 24
            #-1 -1  1  1  1   1   1  1  0  0


        return {
            "dynamic_trackids": trackids,
            "dynamic_trajs": obj_trajs,
            "dynamic_traj_masks": masks,
        }

    def get_Hz_by_filename_from_json(self, paths):
        filename1 = paths[0].strip(os.sep).split(os.sep)[-1].replace(".json","").replace("_LidarFusion.json", "").replace("_MainLidar01.json","")
        filename2 = paths[1].strip(os.sep).split(os.sep)[-1].replace(".json","").replace("_LidarFusion.json", "").replace("_MainLidar01.json","")
        dt = abs(int(filename1.split(".")[-1][:3]) - int(filename2.split(".")[-1][:3]) / 1000) # 秒为单位
        # Hz = 1 / dt # Hz为单位
        return dt

    def get_anno_e2e_static_traj(self, recs,  num_static_pts=20):
        """
        自车动静态轨迹：
        - 动态轨迹：取当前帧前 40 帧、后 50 帧的自车位置信息，做一阶差分 / dt 得到速度轨迹。
        - 静态轨迹：从当前帧到 clip 结束的自车位置中，找到“到达 BEV 范围边缘的点”，
          然后将当前点到该边缘点之间的路径按等步长插值成 num_static_pts 个点。
        """
        # dynamic_traj_vs = []
        # static_trajs = []
        # for label_path in recs:
        label_path = recs[-1]
        ego_traj_path = label_path.replace(".json", "_ego_trajectory.npy")
        traj_mask = np.load(ego_traj_path)  # (T, 4): xyz + mask
        coords_full = traj_mask[:, :3].astype(np.float32)  # (T, 3)
        mask_full = traj_mask[:, -1].astype(np.float32)    # (T,)

        # 当前帧索引：mask == 0
        cur_idx_arr = np.where(mask_full == 0)[0] # 做真值时要求必须存在当前帧
        cur_idx = int(cur_idx_arr[0])

        # -------- 静态轨迹：从当前到 BEV 范围边缘，等步长插值为 num_static_pts 点 --------
        x_min, x_max = self.grid_conf["xbound"][0], self.grid_conf["xbound"][1]
        y_min, y_max = self.grid_conf["ybound"][0], self.grid_conf["ybound"][1]
        xs = coords_full[:, 0]
        ys = coords_full[:, 1]
        inside = (xs >= x_min) & (xs <= x_max) & (ys >= y_min) & (ys <= y_max)

        # 从当前帧往后，找到最后一个仍在 BEV 范围内的点，视为“边缘点”
        inside_after = inside[cur_idx:]
        if inside_after.any():
            last_inside_rel = np.where(inside_after)[0][-1]
            edge_idx = cur_idx + int(last_inside_rel)
        else:
            # 当前之后都不在 BEV 范围内，则静态轨迹退化为当前点
            edge_idx = cur_idx

        p_start = coords_full[cur_idx].astype(np.float32)
        p_end = coords_full[edge_idx].astype(np.float32)

        if num_static_pts <= 1:
            static_traj = p_end[None, :]
        else:
            # 等步长插值 num_static_pts 个点（包含起点和终点）
            alphas = np.linspace(0.0, 1.0, num_static_pts, dtype=np.float32)[:, None]  # (N, 1)
            static_traj = p_start[None, :] + alphas * (p_end[None, :] - p_start[None, :])  # (N, 3)

        return {
            "gt_e2e_static_traj": static_traj,
        }

    

    def get_anno_e2e_dynamic_traj(self, recs, dt=0.1):
        """
        自车动静态轨迹：
        - 动态轨迹：取当前帧前 40 帧、后 50 帧的自车位置信息，做一阶差分 / dt 得到速度轨迹。
        - 静态轨迹：从当前帧到 clip 结束的自车位置中，找到“到达 BEV 范围边缘的点”，
          然后将当前点到该边缘点之间的路径按等步长插值成 num_static_pts 个点。
        """
        # dynamic_traj_vs = []
        # static_trajs = []
        # for label_path in recs:
        label_path = recs[-1]
        ego_traj_path = label_path.replace(".json", "_ego_trajectory.npy")
        traj_mask = np.load(ego_traj_path)  # (T, 4): xyz + mask
        coords_full = traj_mask[:, :3].astype(np.float32)  # (T, 3)
        mask_full = traj_mask[:, -1].astype(np.float32)    # (T,)

        # 当前帧索引：mask == 0
        cur_idx_arr = np.where(mask_full == 0)[0] # 做真值时要求必须存在当前帧
        cur_idx = int(cur_idx_arr[0])

        # -------- 动态轨迹：前 5 帧 + 后 50 帧，速度序列 --------
        T = coords_full.shape[0]
        start_idx = max(0, cur_idx - self.his_lens["e2e_dynamic_traj"])
        end_idx = min(T - 1, cur_idx + self.fur_lens["e2e_dynamic_traj"])
        dyn_coords = coords_full[start_idx:end_idx + 1].copy()  # (L, 3)
        if dyn_coords.shape[0] >= 2:
            dynamic_traj_v = (dyn_coords[1:] - dyn_coords[:-1]) / dt  # (L-1, 3)
        else:
            dynamic_traj_v = np.zeros((0, 3), dtype=np.float32)

        
        return {
            "gt_e2e_dynamic_traj": dynamic_traj_v
        }


    def get_algin_theta_mat(self,recs):
        theta_mats = []
        for i ,rec in enumerate(recs):
            label = json.load(open(rec,"r"))
            ego_pose = label["ego_pose"]
            T_ego2wld_cur = TransformationmatrixEgo(ego_pose["orientation"],ego_pose["position"])
            if i == 0:
                T_ego2wld_pre = T_ego2wld_cur
                theta_mat = np.array([
                    [1.,0.,0.],
                    [0.,1.,0.]
                ])
            else:
                sx = self.bevh / 2
                sy = self.bevw / 2
                pose_diff = np.linalg.inv(T_ego2wld_pre)@T_ego2wld_cur
                yaw = np.arctan2(pose_diff[1,0], pose_diff[0,0])
                dx = pose_diff[0, 3]
                dy = pose_diff[1, 3]
                cos = np.cos(yaw)
                sin = np.sin(yaw)
                eye = np.zeros((3,3),dtype=np.float32)
                rel_pose = eye.copy()
                rel_pose[2,2]=1
                rel_pose[0,0],rel_pose[0,1],rel_pose[0,2] = cos,-sin,dx
                rel_pose[1,0],rel_pose[1,1],rel_pose[1,2] = sin,cos,dy
                pre_mat = np.array([[0., 1./sy, 0.], # 车头往上
                                    [1./sx, 0., 1/5],
                                    [0., 0., 1.]])
                post_mat = np.array([[0., sx, -sx/5],
                                    [sy, 0., 0.],
                                    [0., 0., 1.]])
                theta_mat = (pre_mat@(rel_pose@post_mat))[:2,:]
            theta_mats.append(theta_mat)
        return np.array(theta_mats)

    def gt_value(self,m,src,mask,idx):
        v = src.get(m,False)
        if v:
            return float(v)
        else:
            mask[idx] = 0
            return 0.
    def diff(self,object_trajectorys,track_id,bboxes_mask,dt=0.1,idx=7):
        traj =  object_trajectorys.get(track_id,None)
        if traj  is None:
            bboxes_mask[idx:idx+3] = [0,0,0]
            return 0,0,0
        traj_obj = traj["obj_trajectory_in_obj"]
        masks = traj_obj[:,-1]
        ci = masks.reshape(-1).tolist().index(0)
        pi = masks[ci-1]
        if pi != 1:
            bboxes_mask[idx:idx+3] = [0,0,0]
            return 0,0,0
        traj_obj_sub =  traj_obj[ci-1:ci+1][:,:3]
        diff = (traj_obj_sub[-1] - traj_obj_sub[0] ) / dt
        return diff[0],diff[1],diff[2]

    def dynamic_static_data(self,clip_name):
        """"
        看该clip属于动态还是静态,还是端到端数据
        """
        # if self.mode is not None:
        return self.mode
        # for mode,clip_names in self.mode2clip.items():
        #     if clip_name in clip_names:
        #         return mode
    
    def gen_random_history_indexs(self,len=5,mode="dynamic"):
        indexs_dict = {}
        for task in self.task_class_names[mode]:
            if self.task_flags[task]:
                sub_indexs = sorted(random.sample(
                    [i for i in range(self.task_indexs[task][0]+len,self.task_indexs[task][1])], # 对预测任务来说
                    len)) # 历史长度
                indexs_dict[task] = sub_indexs
        return indexs_dict


    def get_input_indexs(self,mode="dynamic"):
        if self.task_index_random:
            #随机生成的轨迹合并，分成两大类：动态和静态，否则占资源
            his_len = self.seq_len - 1 # 历史长度
            task_indexs = self.gen_random_history_indexs(len=his_len,mode=mode)
            indexs = []
            for task in self.task_class_names[mode]:
                if task in task_indexs:
                    indexs += task_indexs[task]
            indexs = list(set(indexs))   
            indexs = sorted(random.sample(indexs,his_len)) + [self.current_frame_index]
        else:
            indexs = [self.current_frame_index - i for i in range(self.seq_len)][::-1]
        
        return indexs
        


    def __getitem__(self, index):
        sces_len = self.sces_len
        scenes = self.scenes
        sce_id = [i for i in range(len(sces_len)) if (sum(sces_len[:i]) <= index and sum(sces_len[:i + 1]) > index)][0]
        clip_name = scenes[sce_id]
        sce_id_ind = index - sum(sces_len[:sce_id])
        rec = self.ixes[clip_name][sce_id_ind:sce_id_ind + self.total_len]
        dt = self.get_Hz_by_filename_from_json(rec)
        mode = self.dynamic_static_data(clip_name) # 判断那种数据类型，是否指标了动态数据，还是只标了静态数据，还是两者都标注了
        indexs= self.get_input_indexs(mode)  # 16 17 18 19 20
        #数据增强部分
        recs = [rec[i] for i in indexs]
        imgs, rots, trans, intrins, distorts, post_rots, post_trans = self.get_image_data(recs)
        images_parma = {
            "x": imgs,                  # 14 8 3 128 384 # 最后在batch上拼接 # bs * T * 8 * 3 * 128 * 384
            "rots": rots,               # 14 8 3 3        
            "trans": trans,
            "intrins": intrins,         # 14 8 3 3 3
            "distorts": distorts,       # 14 8 3 1 8
            "post_rots": post_rots,     # 14 8 3 2 2
            "post_trans": post_trans
        }

        #时序对其部分
        theta_mats = self.get_algin_theta_mat(recs)
        algin_matrixs = {
            "theta_mats": torch.Tensor(theta_mats)
        }


        anno_infos = {}
        

        # 有哪些任务收集哪些真值
        if mode == "dynamic" or mode == "dynamic_static":
            if self.task_flags["det2D"]:
                anno_infos.update(self.get_anno_det2D(recs))
            if self.task_flags["det3D"]:
                anno_infos.update(self.get_anno_det3D(recs,dt=dt))
            if self.task_flags["obj_dynamic_traj"]:
                anno_infos.update(self.get_anno_obj_dynamic_traj(recs,dt=dt))

        if mode == "static" or mode == "dynamic_static":
            if self.task_flags["map2D"]:
                anno_infos.update(self.get_anno_map2D(recs))
            if self.task_flags["map3D"]:
                anno_infos.update(self.get_anno_map3D(recs))
            if self.task_flags["e2e_static_traj"]:
                anno_infos.update(self.get_anno_e2e_static_traj(recs,  num_static_pts=20))

        if self.task_flags["e2e_dynamic_traj"] or mode == "dynamic_static":
            anno_infos.update(self.get_anno_e2e_dynamic_traj(recs, dt=dt))
        
        # 避免 DataLoader default_collate 遇到 None 报错：未参与任务的键保持为 None 时改为可 collate 的占位
        for k in list(anno_infos.keys()):
            if anno_infos[k] is None:
                anno_infos[k] = []

        out = {}
        out.update(images_parma)
        out.update(algin_matrixs)
        out.update(anno_infos)

        return out


    def __len__(self):
        return sum([(len(self.ixes[i]) - self.total_len) for i in self.ixes.keys()])
    
    def prepro(self):
        all_clip_names = list(self.clip2path.keys())
        num = self.config["train_clips"]
        clip_names = []
        if isinstance(num, int):
            if num <=0 or num >= len(all_clip_names):
                clip_names = all_clip_names
            else:
                clip_names = all_clip_names[:num]
        elif isinstance(num, (list, str)):
            if isinstance(num, str) :
                if not os.path.exists(num):
                    clip_names = all_clip_names
                else:
                    clip_names = []
                    with open(num,"r") as f:
                        for line in f.readlines():
                            clip_name = line.strip(os.sep).split(os.sep)[-1]
                            if clip_name in all_clip_names:
                                clip_names.append(clip_name)
            else:
                clip_names = []
                for name in num:
                    if name in all_clip_names:
                        clip_names.append(name)
            if len(clip_names) == 0:
                clip_names = all_clip_names
        else:
            clip_names = all_clip_names
            
        sample_list = {}
        for clip_name in clip_names:
            path = self.clip2path[clip_name]
            files = sorted([os.path.join(path,"label",file) for file in os.listdir(os.path.join(path,"label")) if file.endswith(".json")])
            if len(files) < self.total_len:continue
            sample_list[clip_name] = files
        self.ixes = sample_list

        

    def get_all_paths(self):
        mode2clip = {}
        all_paths = []
        clip2path = {}
        for name,paths in self.config['clip_paths'].items():
            if name != self.mode:continue
            for path in paths:
                with open(path,"r") as f:
                    for line in f.readlines():
                        p = line.strip()
                        if not os.path.exists(p) or not os.path.isdir(p):continue
                        all_paths.append(p)
                        clip_name = p.strip(os.sep).split(os.sep)[-1]
                        clip2path[clip_name] = p
                        if name in mode2clip.keys():
                            mode2clip[name].append(clip_name)
                        else:
                            mode2clip[name] = [clip_name]
        self.mode2clip = mode2clip
        self.all_paths = all_paths
        self.clip2path = clip2path


from torch.utils.data.distributed import DistributedSampler 
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data import DataLoader
def custom_collate(batch):
    import torch
    import numpy as np

    elem = batch[0]
    out = {}

    for key in elem.keys():
        values = [d[key] for d in batch]

        # 处理已知的变长字段：保持为列表（不堆叠）
        if key in ['gt_labels_det3D', 'gt_bboxes_det3D',
                   'gt_labels_det3D_mask', 'gt_bboxes_det3D_mask',
                   'gt_labels_map3D', 'gt_bboxes_map3D', 'gt_pts_map3D',
                   'gt_obj_dynamic_traj']:
            out[key] = values  # 保持为 list of samples' original data
            continue

        # 处理需要堆叠的 Tensor 或 NumPy 数组
        if isinstance(values[0], torch.Tensor):
            # 检查形状是否一致，若不一致则需进一步处理（如填充）
            if all(v.shape == values[0].shape for v in values):
                out[key] = torch.stack(values, 0)
            else:
                # 如果形状不一致（如轨迹长度不同），可在此填充到统一长度
                # 这里简单抛出异常，提示需要预先填充或单独处理
                raise RuntimeError(f"Tensor shapes inconsistent for key {key}: {[v.shape for v in values]}")
        elif isinstance(values[0], np.ndarray):
            tensors = [torch.from_numpy(v) for v in values]
            if all(t.shape == tensors[0].shape for t in tensors):
                out[key] = torch.stack(tensors, 0)
            else:
                raise RuntimeError(f"NumPy shapes inconsistent for key {key}")
        elif isinstance(values[0], (int, float)):
            out[key] = torch.tensor(values)
        else:
            # 其他类型（如 list, dict）直接保留
            out[key] = values

    return out
    
def worker_rnd_init(x):
    np.random.seed(13 + x)

def build_dataloader(config, mode="dynamic"):
    dataset = Dataset(config, mode)
    if config["is_train"]:
        # 仅当已调用 torch.distributed.init_process_group() 时才用 DistributedSampler，否则单卡/本地调试会报错
        if torch.distributed.is_initialized():
            sampler = DistributedSampler(dataset)
            shuffle = False
        else:
            sampler = None
            shuffle = True
        pin_memory = False
    else:
        sampler = SequentialSampler(dataset)
        shuffle = False
        pin_memory = True
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        sampler=sampler,
        shuffle=shuffle,
        num_workers=config["num_workers"],
        drop_last=True,
        pin_memory=pin_memory,
        collate_fn=custom_collate,
        worker_init_fn=worker_rnd_init
    )

    return dataloader,sampler
 
