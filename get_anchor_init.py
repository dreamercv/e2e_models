# -*- encoding: utf-8 -*-
'''
@File         :get_anchor_init.py
@Date         :2026/03/30 13:47:58
@Author       :Binge.Van
@E-mail       :afb5szh@bosch.com
@Version      :V1.0.0
@Description  :

'''


import os,sys

import json
import numpy as np
from tqdm import tqdm
def main():
    pass



if __name__ == '__main__':
    path = "/workspace/afb5szh-01/models/e2e_model/e2e_dataset_10Hz_dyo_all.txt"
    anchors = []
    with open(path,"r") as f:
        for line in f.readlines():
            # print(line.strip())
            if "20260101010101" not in line:continue
            clip_path = line.strip()
            label_root = os.path.join(clip_path,"label")
            files = [file for file in os.listdir(label_root) if file.endswith(".json")]
            
            for file in tqdm(files):
                label_path  = os.path.join(label_root,file)
                json_data = json.load(open(label_path,"r"))
                # print(json_data)
                for id, obj in json_data["object_infos"].items():
                    name = obj["sub_category"]
                    if name not in ["car", "truck", "bus"]:continue
                    pos = obj.get("position")
                    rot = obj.get("rotation")
                    dim = obj.get("dimension")
                    x,y,z = pos["x"],pos["y"],pos["z"]
                    if x > 120 or x < -80 or y < -40 or y > 40:continue
                    w,l,h = dim["width"],dim["length"],dim["height"]
                    w, l, h = dim["width"], dim["length"], dim["height"]
                    log_w = np.log(w + 1e-6)  # 防止 log(0)
                    log_l = np.log(l + 1e-6)
                    log_h = np.log(h + 1e-6)

                    # 其他字段不变
                    yaw = rot["yaw"]
                    siny = np.sin(yaw)
                    cosy = np.cos(yaw)

                    
                    anchors.append([x, y, z, 1, 1, 1, 1, 0, 0.0, 0.0, 0.0])
    print()
    from sklearn.cluster import KMeans
    X = np.array(anchors, dtype=np.float32)
    kmeans = KMeans(n_clusters=50, random_state=42, n_init='auto')
    kmeans.fit(X)
    centers = kmeans.cluster_centers_.astype(np.float32)
    np.save("../anchor_init_20260101010101_50.npy", centers)