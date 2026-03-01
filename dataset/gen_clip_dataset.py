import os,sys
import json
import numpy as np
from time import time
from datetime import datetime,timedelta
import cv2
save_root = "/home/fb/project/models/Sparse4D-main/projects/e2e_models/dataset/clip_dataset"
det_path = "/home/fb/project/models/Sparse4D-main/projects/e2e_models/dataset/det.json"
map_path = "/home/fb/project/models/Sparse4D-main/projects/e2e_models/dataset/map.json"

map_json = json.load(open(map_path,"r"))
det_json = json.load(open(det_path,"r"))
jsons = [map_json,det_json]
clip_name = "clip_000003"
folder_name = [f"map/{clip_name}",f"det/{clip_name}"]
frame_num = 300
for json_data,folder_name in zip(jsons,folder_name):
    parameters = json_data["parameters"]
    
    now = datetime.now()- timedelta(hours=2)
    for frame_id in range(frame_num):
        now += timedelta(milliseconds=100)
        for sensor_name,parameter in parameters.items():
            image_height,image_width = parameter["image_height"],parameter["image_width"]
            
            image = np.zeros((image_height,image_width,3),dtype=np.uint8)
            #用毫秒级的时间戳，格式为202602262216000
            name = now.strftime("%Y%m%d%H%M%S") + f".{now.microsecond // 1000:03d}"
            image_path = os.path.join(save_root,folder_name,sensor_name,name + ".jpg")
            os.makedirs(os.path.dirname(image_path),exist_ok=True)  
            cv2.imwrite(image_path,image)
        json_path = os.path.join(save_root,folder_name,"labels",name + ".json")
        os.makedirs(os.path.dirname(json_path),exist_ok=True)
        with open(json_path,"w") as f:
            json.dump(json_data,f,indent=4)
            f.write("\n")