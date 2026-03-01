# -*- encoding: utf-8 -*-
'''
@File         :dataset.py
@Date         :2026/02/28 13:31:48
@Author       :Binge.Van
@E-mail       :afb5szh@bosch.com
@Version      :V1.0.0
@Description  :

'''


import os,sys
import json
import cv2
import numpy as np

import torchvision

import torch

configs = {
    # 总的参数
    "clip_paths":{
        "dyo":["/home/fb/project/models/Sparse4D-main/projects/e2e_models/dataset/clip_dataset/det.txt"],
        "lane":["/home/fb/project/models/Sparse4D-main/projects/e2e_models/dataset/clip_dataset/map.txt"],
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


    "camera_names":[
        "FrontCam02",     #前 
        "RearCam01",      #后方
        "SideFrontCam01", #左前
        "SideFrontCam02", #右前
        "SideRearCam01",  #左后
        "SideRearCam02"   #右后
    ],
    
}

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
        self.total_len =config["total_len"]
        self.current_frame_index = config["current_frame_index"]
        self.cns = config["camera_names"]
        self.fH, self.fW  = config["final_dim"]
        self.cnis = config["camera_infos"]


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
                # image_name = os.path.basename(image_paths[cn])
                # image_path = os.path.join(
                #     os.path.dirname(label_path).replace("label",cn),
                #     image_name
                # )
                image_path = label_path.replace("label",cn).replace(".json",".jpg")
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

    def get_annos_det2D(self,):
        pass

    def get_annos_det3D(self):
        pass

    def get_anno_map2D(self):
        pass

    def get_anno_map3D(self):
        pass
    
    def get_anno_object_trajectory(self):
        pass

    def get_anno_ego_trajectory(self):
        pass

    def get_algin_theta_mat(self):
        pass

    

    def __getitem__(self,index):
        sces_len = self.sces_len
        scenes = self.scenes
        sce_id = [i for i in range(len(sces_len)) if (sum(sces_len[:i]) <= index and sum(sces_len[:i + 1]) > index)][0]
        sce_id_ind = index - sum(sces_len[:sce_id])
        rec = self.ixes[scenes[sce_id]][sce_id_ind:sce_id_ind+self.total_len]
        imgs, rots, trans, intrins, distorts, post_rots, post_trans = self.get_image_data(rec)
        print(index,len(rec))

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
                        if name in clip2path.keys():
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