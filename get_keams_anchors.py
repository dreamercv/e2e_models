# -*- encoding: utf-8 -*-
'''
@File         :get_keams_anchors.py
@Date         :2026/03/20 15:45:41
@Author       :Binge.Van
@E-mail       :afb5szh@bosch.com
@Version      :V1.0.0
@Description  :

'''


import os,sys

import json

def main():
    pass



if __name__ == '__main__':
    path = "/workspace/afb5szh-01/xflow_models_task/lora/xflow/xtorch_usecases/src/xtorch_usecases/single_camera_mpc4/tasks/dyo/sparse4d/300clips_kmeans512_range_200.npy"
    import numpy as np
    anchors = np.load(path)
    save_path = "/workspace/afb5szh-01/models/e2e_models-master/keams_anchor.txt"
    messages = []
    for anchor in anchors:
        message = ",".join(list(map(str,anchor.tolist())))
        messages.append(f"{message}\n")
    with open(save_path,"w") as f:
        f.writelines(messages)