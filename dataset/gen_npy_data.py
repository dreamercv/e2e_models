import os
import json
import numpy as np

root = "/home/fb/project/models/Sparse4D-main/projects/e2e_models/dataset/clip_dataset/map"
clip_names = os.listdir(root)
for clip in clip_names:
    label_paths = [os.path.join(root,clip,"label",file) for file in os.listdir(os.path.join(root,clip,"label")) if file.endswith(".json")]
    clip_len = len(label_paths)
    for i, label_path in enumerate(label_paths):
        json_data = json.load(open(label_path,"r"))
        print(label_path)
        object_infos = json_data["object_infos"]
        trajs = {}
        for trackid in object_infos.keys():
            mask = np.zeros((clip_len,1)) + 1
            mask[i,0] = 0
            traj_obj = np.zeros((clip_len,3))
            traj_ego = np.zeros((clip_len,3))
            trajs[trackid] = {
                "obj_trajectory_in_obj":np.concatenate([traj_obj,mask],1),
                "obj_trajectory_in_ego":np.concatenate([traj_ego,mask],1)
            }
        save_path_obj = label_path.replace(".json", "_object_trajectory.npy")
        # np.save(save_path_obj,trajs)
        save_path_ego = label_path.replace(".json", "_ego_trajectory.npy")
        traj_ego = np.zeros((clip_len,3))
        mask_ego = np.zeros((clip_len,1)) + 1
        mask_ego[i,0] = 0
        np.save(save_path_ego,np.concatenate([traj_ego,mask_ego],1))
        print(f"save {save_path_ego}")
                    


            