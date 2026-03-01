import os
import numpy as np
import cv2
camera_infos={
        "FrontCam02":{"resize_lim":(0.1, 0.1),"bot_pct_lim":(0.15, 0.25)},     #前 大
        "FrontCam02":{"resize_lim":(0.1, 0.1),"bot_pct_lim":(0.15, 0.25)},     #前 中
        "FrontCam02":{"resize_lim":(0.1, 0.1),"bot_pct_lim":(0.15, 0.25)},     #前 小 roi
        "RearCam01":{"resize_lim":(0.1, 0.1),"bot_pct_lim":(0.15, 0.25)},      #后方
        "SideFrontCam01":{"resize_lim":(0.1, 0.1),"bot_pct_lim":(0.15, 0.25)}, #左前
        "SideFrontCam02":{"resize_lim":(0.1, 0.1),"bot_pct_lim":(0.15, 0.25)}, #右前
        "SideRearCam01":{"resize_lim":(0.1, 0.1),"bot_pct_lim":(0.15, 0.25)},  #左后
        "SideRearCam02":{"resize_lim":(0.1, 0.1),"bot_pct_lim":(0.15, 0.25)}   #右后
    }

path = "/home/fb/project/models/Sparse4D-main/projects/e2e_models/dataset/clip_dataset/map/map_clip_000000/FrontCam01/20260226223018.684.jpg"
img = cv2.imread(path)
h,w = img.shape[:2]
cx,cy = w//2,h//2
cv2.circle(img,(cx,cy),10,(0,0,255),-1)
cv2.imwrite("img.jpg",img)
print("done!")


data_aug_conf = {
        'final_dim': (128, 352),
        'rot_lim': (-0.0, 0.0),
        'H_8M': 2160, 'W_8M': 3840,
        'H_3M': 1536, 'W_3M': 1920,
        'H_2M': 1080, 'W_2M': 3520,
        'H_fisheye': 1536, 'W_fisheye': 1920,
        # 'resize_lim_2M': [(0.14, 0.17), (0.2, 0.25),(0.35,0.7)],
        # 'bot_pct_lim_2M': [(0.04, 0.15), (0.20, 0.3),(0.3,0.43)],
        # 'resize_lim_fisheye': [(0.24, 0.32)], #(x-512) = 64+(x-512)/2;  x = 640 ---> scale = 640/3520=0.1818 
        ####
        'resize_lim_fisheye': [(0.2, 0.3)], #(x-512) = 64+(x-512)/2;  x = 640 ---> scale = 640/3520=0.1818 
       ########
        # 'bot_pct_lim_fisheye': [(0.3, 0.5)],
        'bot_pct_lim_fisheye': [(0.2, 0.5)],
        # 'resize_lim_3M': [(0.24, 0.32)], #(x-512) = 64+(x-512)/2;  x = 640 ---> scale = 640/3520=0.1818 
        'resize_lim_3M': [(0.2, 0.3)], #(x-512) = 64+(x-512)/2;  x = 640 ---> scale = 640/3520=0.1818 
        # 'bot_pct_lim_3M': [(0.3, 0.4)],
        'bot_pct_lim_3M': [(0.2, 0.5)],
        # 'resize_lim_8M': [(0.1, 0.1),(0.17,0.21), (0.4, 0.45)],
        # 'bot_pct_lim_8M': [(0.15, 0.25), (0.27, 0.37),(0.37,0.47)],
        'resize_lim_8M': [(0.1, 0.1),(0.17,0.23), (0.35, 0.45)],
        'bot_pct_lim_8M': [(0.15, 0.25), (0.27, 0.37),(0.37,0.47)],
        'cams': ['CAM_FRONT_roi0', 'CAM_FRONT_roi1'],
        'rand_flip': False,
        'Ncams': 2,
        }

def sample_augmentation( CAM, cam_idx, H, W):
    flip = False
    H_fisheye, W_fisheye = data_aug_conf['H_fisheye'], data_aug_conf['W_fisheye']
    H_3M, W_3M = data_aug_conf['H_3M'], data_aug_conf['W_3M']
    H_8M, W_8M = data_aug_conf['H_8M'], data_aug_conf['W_8M']
    H_2M, W_2M = data_aug_conf['H_2M'], data_aug_conf['W_2M']
    fH, fW = data_aug_conf['final_dim']  # 128,512

    if 1:
        if H == H_fisheye and W == W_fisheye:
            if CAM == 'back':
                resize = np.random.uniform(*data_aug_conf['resize_lim_3M'][cam_idx])
                resize_dims = (int(W * resize), int(H * resize))
                newW, newH = resize_dims
                crop_h = int((1 - np.random.uniform(*data_aug_conf['bot_pct_lim_3M'][cam_idx])) * newH) - fH
            else:
                resize = np.random.uniform(*data_aug_conf['resize_lim_fisheye'][cam_idx])  # 0.212
                resize_dims = (int(W * resize), int(H * resize))  # (408, 326)
                newW, newH = resize_dims
                crop_h = int(
                    (1 - np.random.uniform(*data_aug_conf['bot_pct_lim_fisheye'][cam_idx])) * newH) - fH

        elif H == H_8M and W == W_8M:
            resize = np.random.uniform(*data_aug_conf['resize_lim_8M'][cam_idx])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*data_aug_conf['bot_pct_lim_8M'][cam_idx])) * newH) - fH
        elif H == H_2M and W == W_2M:
            resize = np.random.uniform(*data_aug_conf['resize_lim_2M'][cam_idx])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*data_aug_conf['bot_pct_lim_2M'][cam_idx])) * newH) - fH
        if CAM == 'front':
            if cam_idx == 0:
                crop_w = int(np.random.uniform(0, newW - fW))
            else:
                crop_w = int(np.random.uniform(-fW / 8., fW / 8.) + (newW - fW) / 2.)
        if CAM == 'left' or CAM == 'right':
            crop_w = int(np.random.uniform(-fW / 32., fW / 32.) + (newW - fW) / 2.)
        if CAM == 'back':
            # crop_w = int(np.random.uniform(-fW/8., fW/8.)+(newW - fW)/2.)
            crop_w = int(np.random.uniform(0, newW - fW))
        crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        rotate = np.random.uniform(*data_aug_conf['rot_lim'])

    
    
    return resize, resize_dims, crop, flip, rotate


    
H, W = h,w
print(H, W)
result1=sample_augmentation('front', 0, H, W)
result2=sample_augmentation('front', 1, H, W)
result3=sample_augmentation('front', 2, H, W)
result4=sample_augmentation('back', 0, 1536, 1920)
result5=sample_augmentation('left', 0, 1536, 1920)
print(result1)
print(result2)
print(result3)
print(result4)
print(result5)