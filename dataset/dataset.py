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

'''


import os,sys
import json
import cv2
import numpy as np

import torchvision

import torch

configs = {
    # 总的参数
    "grid_conf" : {
        'xbound': [-80.0, 120.0, 1],
        'ybound': [-40.0, 40.0, 1],
        'zbound': [-2.0, 4.0, 1.0]
    },

    "clip_paths":{
        "dynamic":["/workspace/afb5szh-01/models/e2e_model/e2e_dataset_10Hz_dyo.txt"],
        "static":["/workspace/afb5szh-01/models/e2e_model/e2e_dataset_10Hz_lane.txt"],
    },
    "camera_infos":{
        "FrontCam02":{
            "resize_lim": [(0.1, 0.1),  (0.17,0.23), (0.35, 0.45)],
            "bot_pct_lim":[(0.15, 0.25),(0.27, 0.37),(0.37,0.47)]
        },     #前 大
        "RearCam01":     {"resize_lim":[(0.2, 0.3)],"bot_pct_lim":[(0.2, 0.5)]},      #后方
        "SideFrontCam01":{"resize_lim":[(0.2, 0.3)],"bot_pct_lim":[(0.2, 0.5)]}, #左前
        "SideFrontCam02":{"resize_lim":[(0.2, 0.3)],"bot_pct_lim":[(0.2, 0.5)]}, #右前
        "SideRearCam01": {"resize_lim":[(0.2, 0.3)],"bot_pct_lim":[(0.2, 0.5)]},  #左后
        "SideRearCam02": {"resize_lim":[(0.2, 0.3)],"bot_pct_lim":[(0.2, 0.5)]}   #右后
    },
    'rot_lim': (-0.0, 0.0),
    "flip":False,

    'final_dim': (128, 384),






    "train_clips":2, # -1 所有，[]指定几个，xxx.txt写进txt中指定的，>0前N个
    
    "total_len":91,
    "current_frame_index":40,#0 1 2 ... 40 ...90 当前帧往前用作网络输入，当前帧往后用作未来真值
    

    "task_flag":{
        "det2D":True,
        "det3D":True,

        "map2D":True,
        "map3D":True,

        "obj_dynamic_traj":True,
        "e2e_static_traj":True,
        "e2e_dynamic_traj":True,
    },
    "task_indexs":{
        "det2D":[36,37,38,39,40],
        "det3D":  [36,37,38,39,40],
        
        "map2D":[36,37,38,39,40],
        "map3D":  [36,37,38,39,40],

        "obj_dynamic_traj": [10,20,30,40],
        "e2e_dynamic_traj": [10,20,30,40],
        "e2e_static_traj":  [10,20,30,40],

    },




    "camera_names":[
        "FrontCam02",     #前 
        "RearCam01",      #后方
        "SideFrontCam01", #左前
        "SideFrontCam02", #右前
        "SideRearCam01",  #左后
        "SideRearCam02"   #右后
    ],
    # 3D 检测类别：与 convert_det 输出的 object_infos[].sub_category 对应，用于映射到类别下标
    "det_class_names": ["car", "truck", "bus", "pedestrian", "bicycle", "motorcycle", "traffic_cone", "barrier"],
}



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
    return img, post_rot, post_tran

class Dataset(torch.utils.data.Dataset):
    def __init__(self,config):
        self.config = config

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
        self.useful_indexs,self.group_indexs = self.get_input_indexs()

        # 数据准备
        self.get_all_paths()
        self.prepro()
        self.sces_len = [(len(self.ixes[i]) - self.total_len ) for i in self.ixes.keys()]
        self.scenes = [i for i in self.ixes.keys()]

    def gen_aug_params(self,H,W,resize_lim,bot_pct_lim,rot_lim):
        fH, fW = self.fH, self.fW
        resize = np.random.uniform(*resize_lim)
        resize_dims = (int(W * resize), int(H * resize))
        newW, newH = resize_dims
        # 保证 resize 后至少能裁出 (fW, fH)，否则用下限
        if newW < fW or newH < fH:
            scale = max(fW / max(newW, 1), fH / max(newH, 1))
            resize = min(1.0, resize * scale)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
        crop_h = int((1 - np.random.uniform(*bot_pct_lim)) * newH) - fH
        crop_w = int(max(0, newW - fW) / 2)
        crop_h = int(np.clip(crop_h, 0, max(0, newH - fH)))
        crop_w = int(np.clip(crop_w, 0, max(0, newW - fW)))
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        rotate = np.random.uniform(*rot_lim)
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
                tran =  lidar2camera[:3,3]
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
                    distorts_i.append(torch.Tensor(distort))
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

    def get_anno_det3D(self,recs):
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
        class_names = self.config.get("det_class_names", ["car", "truck", "bus", "pedestrian", "bicycle"])
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
                label_mask,bboxes_mask = 1,1
                pos = obj.get("position")
                rot = obj.get("rotation")
                dim = obj.get("dimension")
                if pos is None or rot is None or dim is None:
                    continue
                sub = obj.get("sub_category", "car")
                try:
                    cls_id = class_names.index(sub)
                except ValueError:
                    cls_id = 0
                    label_mask = 0
                x = self.gt_value("x",pos,bboxes_mask)# float(pos.get("x", 0))
                y = self.gt_value("y",pos,bboxes_mask)#float(pos.get("y", 0))
                z = self.gt_value("z",pos,bboxes_mask)#float(pos.get("z", 0))
                yaw = self.gt_value("yaw",rot,bboxes_mask)#float(rot.get("yaw", 0.0))
                w = self.gt_value("width",dim,bboxes_mask)#float(dim.get("width", 0))
                length = self.gt_value("length",dim,bboxes_mask)#float(dim.get("length", 0))
                h = self.gt_value("height",dim,bboxes_mask)# float(dim.get("height", 0))
                vx,vy,vz = self.diff(object_trajectorys,track_id,bboxes_mask,dt=0.1)
                bboxes_list.append([x, y, z, w, length, h, yaw, vx,vy,vz])
                labels_list.append(cls_id)
                labels_maks_list.append(label_mask)
                bboxes_mask_list.append(bboxes_mask)
            if len(labels_list) == 0:
                gt_labels_per_frame.append(torch.zeros(0, dtype=torch.long))
                gt_bboxes_per_frame.append(torch.zeros(0, 10, dtype=torch.float32))
            else:
                gt_labels_per_frame.append(torch.tensor(labels_list, dtype=torch.long))
                gt_bboxes_per_frame.append(torch.tensor(bboxes_list, dtype=torch.float32))
        # return gt_labels_per_frame, gt_bboxes_per_frame
        return {
            "gt_labels_det3D":gt_labels_per_frame,
            "gt_bboxes_det3D":gt_bboxes_per_frame
        }

    def get_anno_map3D(self,recs):
        return {
            "gt_labels_map3D":None,
            "gt_bboxes_map3D":None,
        }
    
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
            start_idx = max(0, cur_idx - 40)
            end_idx = min(T - 1, cur_idx + 50)

            coords = coords_full[start_idx:end_idx + 1].copy()  # (L, 3)
            mask = mask_full[start_idx:end_idx + 1].copy()      # (L,)

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
                    mask[s:e + 1][mask[s:e + 1] < 0] = 1.0

            # 计算速度：一阶差分 / dt，长度为 L-1
            traj_v = (coords[1:] - coords[:-1]) / dt

            trackids.append(trackid)
            obj_trajs.append(traj_v)
            masks.append(mask.tolist())

        return {
            "gt_obj_dynamic_traj": {
                "trackids": trackids,
                "dynamic_trajs": obj_trajs,
                "masks": masks,
            }
        }

    def get_anno_e2e_dynamic_traj(self,recs):
        return {
            "gt_e2e_dynamic_traj":None
        }

    def get_anno_e2e_static_traj(self,recs):
        return {
            "gt_e2e_static_traj":None
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
        return theta_mats

    def gt_value(self,m,src,mask):
        v = src.get(m,False)
        if v:
            return float(v)
        else:
            mask = 0
            return 0.
    def diff(self,object_trajectorys,track_id,bboxes_mask,dt=0.1):
        traj =  object_trajectorys.get(track_id,None)
        if traj  is None:
            bboxes_mask = 0
            return 0,0,0
        traj_obj = traj["obj_trajectory_in_obj"]
        masks = traj_obj[:,-1]
        ci = masks.reshape(-1).tolist().index(0)
        pi = masks[ci-1]
        if pi != 1:
            bboxes_mask = 0
            return 0,0,0
        traj_obj_sub =  traj_obj[ci-1:ci+1][:,:3]
        diff = (traj_obj_sub[-1] - traj_obj_sub[0] ) / dt
        return diff[0],diff[1],diff[2]




        

    def dynamic_static_data(self,clip_name):
        for mode,clip_names in self.mode2clip.items():
            if clip_name in clip_names:
                return mode
    
    def get_input_indexs(self):
        indexs = []
        group_indexs = []
        for task,useful in self.task_flags.items():
            if useful:
                sub_index = self.task_indexs[task]
                indexs += sub_index
                if sub_index not in group_indexs:
                    group_indexs.append(sub_index)
        return sorted(list(set(indexs))),group_indexs

    def __getitem__(self, index):
        sces_len = self.sces_len
        scenes = self.scenes
        sce_id = [i for i in range(len(sces_len)) if (sum(sces_len[:i]) <= index and sum(sces_len[:i + 1]) > index)][0]
        clip_name = scenes[sce_id]
        sce_id_ind = index - sum(sces_len[:sce_id])
        rec = self.ixes[clip_name][sce_id_ind:sce_id_ind + self.total_len]
        mode = self.dynamic_static_data(clip_name)
        
        #数据增强部分
        image_rec = [rec[i] for i in self.useful_indexs]
        imgs, rots, trans, intrins, distorts, post_rots, post_trans = self.get_image_data(image_rec)
        images_parma = {
            "x": imgs,
            "rots": rots,
            "trans": trans,
            "intrins": intrins,
            "distorts": distorts,
            "post_rots": post_rots,
            "post_trans": post_trans
        }

        #时序对其部分
        algin_matrixs = {}
        for indexs in self.group_indexs:
            str_index = "_".join(list(map(str,indexs)))
            algin_recs = [rec[i] for i in indexs]
            theta_mats = self.get_algin_theta_mat(algin_recs)
            algin_matrixs[str_index] = theta_mats

        anno_infos = {
            "gt_labels_det2D":None,
            "gt_bboxes_det2D":None,
            "gt_labels_map2D":None,
            "gt_bboxes_map2D":None,

            "gt_labels_det3D":None,
            "gt_bboxes_det3D":None,
            "gt_labels_map3D":None,
            "gt_bboxes_map3D":None,

            "gt_obj_dynamic_traj":None,
            "gt_e2e_dynamic_traj":None,
            "gt_e2e_static_traj":None
        }
        
        if mode == "dynamic" or mode == "static":
            if self.task_flags["e2e_dynamic_traj"]:
                gt_recs = [rec[i] for i in self.task_indexs["e2e_dynamic_traj"]]
                anno_infos.update(self.get_anno_e2e_dynamic_traj(gt_recs))
            
            if mode == "dynamic":
                if self.task_flags["det2D"]:
                    gt_recs = [rec[i] for i in self.task_indexs["det2D"]]
                    anno_infos.update(self.get_anno_det2D(gt_recs))
                if self.task_flags["det3D"]:
                    gt_recs = [rec[i] for i in self.task_indexs["det3D"]]
                    anno_infos.update(self.get_anno_det3D(gt_recs))
                if self.task_flags["obj_dynamic_traj"]:
                    gt_recs = [rec[i] for i in self.task_indexs["obj_dynamic_traj"]]
                    anno_infos.update(self.get_anno_obj_dynamic_traj(gt_recs))
            else:
                if self.task_flags["map2D"]:
                    gt_recs = [rec[i] for i in self.task_indexs["map2D"]]
                    anno_infos.update(self.get_anno_map2D(gt_recs))
                if self.task_flags["map3D"]:
                    gt_recs = [rec[i] for i in self.task_indexs["map3D"]]
                    anno_infos.update(self.get_anno_map3D(gt_recs))
                if self.task_flags["e2e_static_traj"]:
                    gt_recs = [rec[i] for i in self.task_indexs["e2e_static_traj"]]
                    anno_infos.update(self.get_anno_e2e_static_traj(gt_recs))
        else:
            print("没有动静态的数据，没办法训练")

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

        



if __name__ == '__main__':
    dataset = Dataset(configs)
    print(dataset.__len__())
    for i in range(dataset.__len__()):
        dataset.__getitem__(i)
    # root = "/home/fb/project/models/Sparse4D-main/projects/e2e_models/dataset/clip_dataset/map"
    # save_root = root.rstrip(os.sep) + ".txt"
    # with open(save_root,"w") as f:
    #     for file in os.listdir(root):
    #         f.write(os.path.join(root,file) + "\n")
