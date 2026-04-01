# -*- encoding: utf-8 -*-
'''
@File         :get_aug_param.py
@Date         :2026/04/01 14:27:47
@Author       :Binge.Van
@E-mail       :afb5szh@bosch.com
@Version      :V1.0.0
@Description  :

'''


import os,sys
import json
import os
import numpy as np
import cv2
def gen_aug_params(is_train,H,W,resize_lim,bot_pct_lim,rot_lim=(-0,0)):
    fH, fW = 128, 384
    if is_train:
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
    if is_train:
        crop_h = int((1 - np.random.uniform(*bot_pct_lim)) * newH) - fH
    else:
        crop_h = int(1 - np.mean(bot_pct_lim) * newH) - fH
    crop_w = int(max(0, newW - fW) / 2)
    crop_h = int(np.clip(crop_h, 0, max(0, newH - fH)))
    crop_w = int(np.clip(crop_w, 0, max(0, newW - fW)))
    crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
    if is_train:
        rotate = np.random.uniform(*rot_lim)
    else:
        rotate = 0.
    return resize,resize_dims,crop,rotate

aug_param = {
        "FrontCam02": {
            "resize_lim": [(0.1, 0.1), (0.12, 0.15), (0.25, 0.30)],
            "bot_pct_lim": [(0.15, 0.25), (0.20, 0.25), (0.30, 0.35)]
        },  # 前 大
        "RearCam01": {"resize_lim": [(0.05, 0.25)], "bot_pct_lim": [(0.05, 0.2)]},  # 后方
        "SideFrontCam01": {"resize_lim": [(0.1, 0.25)], "bot_pct_lim": [(0.05, 0.3)]},  # 左前
        "SideFrontCam02": {"resize_lim": [(0.1, 0.25)], "bot_pct_lim": [(0.05, 0.3)]},  # 右前
        "SideRearCam01": {"resize_lim": [(0.1, 0.25)], "bot_pct_lim": [(0.15, 0.25)]},  # 左后
        "SideRearCam02": {"resize_lim": [(0.1, 0.25)], "bot_pct_lim": [(0.15, 0.25)]}  # 右后
    }

camera_names = [
        "FrontCam02",  # 前
        "RearCam01",  # 后方
        "SideFrontCam01",  # 左前
        "SideFrontCam02",  # 右前
        "SideRearCam01",  # 左后
        "SideRearCam02"  # 右后
    ]

json_path = "/workspace/afb5szh-01/models/e2e_model/e2e_dataset_10Hz/20260101010101/label/13375C_20250508072723.000000_LidarFusion.json"
json_data = json.load(open(json_path,"r"))
for camera_name in camera_names:
    # camera_name = "SideRearCam02"
    # idx = 0
    is_train =True
    paths = json_data["paths"]
    params = json_data["parameters"]
    H,W = params[camera_name]["image_height"],params[camera_name]["image_width"]
    image_name = os.path.basename(paths[camera_name])
    image_path = os.path.join(json_path.split("label")[0],camera_name,image_name)
    image = cv2.imread(image_path)
    for idx in range(len(aug_param[camera_name]["resize_lim"])):
        for i in range(30):
            resize_lim,bot_pct_lim = aug_param[camera_name]["resize_lim"][idx],aug_param[camera_name]["bot_pct_lim"][idx]
            resize,resize_dims,crop,rotate = gen_aug_params(is_train,H,W,resize_lim,bot_pct_lim)
            # print()
            bbox = [int(box/resize) for box in crop]
            cv2.rectangle(image, bbox[:2],bbox[2:],color= (0,0,255), thickness=5)
            print(resize,bbox,crop[2]-crop[0],crop[3]-crop[1],image.shape)
        cv2.imwrite(f"/workspace/afb5szh-01/models/e2e_model/aug_show/{camera_name}_{idx}.jpg", image)