# -*- encoding: utf-8 -*-
'''
@File         :dataset.py
@Date         :2026/01/27 18:01:00
@Author       :Binge.Van
@E-mail       :afb5szh@bosch.com
@Version      :V1.0.0
@Description  : 将eng的数据转成我想要的数据格式
{
    "paths":{"cam":path,....,"lidar":path,"label":path},
    "parameters":{每个相机的内外惨畸变},
    "object_infos":目标的3d标注以及2d
}

'''
# 目标
#   瞬时: position(xyz),demension(lwh),hea,
#   时序: traj(xy) -- 根据未来算出来的轨迹 --- 需要trackid匹配
#         speed(vx,vy) -- 根据历史算出来的速度----需要trackid匹配
# ego
#   时序: traj --- 根据未来
#         algin ---- 根据历史



import os,sys
import json
import shutil
from tqdm import tqdm
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

import multiprocessing

camera_names = ['FrontCam01', 'FrontCam02', 'SideFrontCam01', 'SideFrontCam02', 'SurCam01', 'SurCam02', 'SurCam03', 'SurCam04', 'SideRearCam01', 'SideRearCam02', 'RearCam01']

def parse_frame_info(frame_info_path):
    data = open(frame_info_path,"r",encoding = "utf-8")
    json_data = json.load(data)
    camera_calibration = json_data["camera_calibration"]
    cameras = camera_calibration["cameras"]
    lidar_imu = camera_calibration["lidar_imu"]
    ego_pose = json_data["pose"]
    infos = {}
    for camera in cameras:
        # print(camera["name"])
        infos[camera["name"]] = camera
    infos["lidar_imu"] = lidar_imu
    infos["ego_pose"] = ego_pose
    # print(json.dumps(infos,indent=4))
    return infos

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

def TransformationmatrixObject(position,rotation):
    roll, pitch, yaw = rotation["roll"],rotation["pitch"],rotation["yaw"]
    xyz = [position["x"],position["y"],position["z"]]
    rot_matrix = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rot_matrix
    transform[:3, 3] = xyz
    return transform

def draw_bev(bev=None,bev_h=200*8,bev_w=80*8,trajs=None,world_x=200,world_y=80,color=(0,0,255)):
    # trajs = trajs.astype(np.uint16)
    ego = (bev_w//2,120*8)
    if bev is None:
        bev = np.zeros((bev_h,bev_w,3),dtype=np.uint8) + 255
    traj_x,traj_y,masks = trajs[:,0],trajs[:,1],trajs[:,-1]
    piex_y = ego[1] - traj_x * (bev_h /world_x )
    piex_x = ego[0] - traj_y * (bev_w /world_y )
    # print(piex_x,piex_y)
    for i in range(len(piex_x)):
    # 用不同颜色表示轨迹点
        px, py = piex_x[i],piex_y[i]
        mask = masks[i]
        if mask==-1:continue
        if mask==0: # 当前帧是黑色
            color_c = (0,0,0)
        else:
            color_c = color
        cv2.circle(bev, (int(px),int(py)), 2, color_c, 3)  # 注意：OpenCV使用(x,y)即(列,行)
    return bev

def standard_format_parameters(frame_info):
    parameters = {}
    for name,parameter_info in frame_info.items():
        if name not in camera_names:continue
        image_height,image_width = parameter_info["image_height"],parameter_info["image_width"]
        lidar2camera = parameter_info["T_c_b"]
        lidar2camera_static = parameter_info["T_c_b_static"]
        camera_matrix = parameter_info["camera_matrix"]
        dist_coeffs = parameter_info["dist_coeffs"]
        parameters[name] = {
            "image_height":image_height,
            "image_width":image_width,
            "lidar2camera":lidar2camera,
            "lidar2camera_static":lidar2camera_static,
            "dist_coeffs":dist_coeffs,
            "camera_matrix":camera_matrix
        }
        
    ego_pose = frame_info["ego_pose"]
    # parameters["ego_pose"]={
    #     "orientation":ego_pose["orientation"],
    #     "position":ego_pose["position"]
    # }
    
    ego_pose_new = {
        "orientation":ego_pose["orientation"],
        "position":ego_pose["position"]
    }
    return parameters,ego_pose_new

def find_sensor_path(json_data):
    collected_frames = json_data["collected_frames"]
    sensor_paths = {}
    frame_info_path = None
    for collected_frame in collected_frames:
        frame_info_path = collected_frame["frame_info_uri"]
        resources = collected_frame["resources"]
        for resource in resources:
            sensor = resource["sensor"]
            uri = resource["uri"]
            sensor_paths[sensor] = uri
    return sensor_paths,frame_info_path
    
def convert_engjson2standardjson(root,files,clip_folder):
    
    # print(files)
    for i, file in enumerate(tqdm(files,desc=f"deal,{clip_folder.strip(os.sep).split(os.sep)[-1]}")) :
        # if i >= 1:break
        json_path = os.path.join(root,file)
        annotation = {}
        data = open(json_path,"r")
        json_data = json.load(data)
        sensor_paths,frame_info_path = find_sensor_path(json_data)
        sensor_paths["label"] = json_path
        frame_info = parse_frame_info(frame_info_path)
        parameters,ego_pose = standard_format_parameters(frame_info) #------------ 统一参数格式

        object_infos = {}
        frames = json_data["labeled_data"]["frames"]
        no_track_id = 0
        for frame in frames:
            groups = frame["groups"]
            for group in groups:
                try:
                    track_id = group["properties"].get("track_id",None)
                    if track_id is None:continue
                    if track_id == "-1":
                    #     track_id = "-"+str(track_id * -1 + no_track_id)
                        no_track_id += 1
                        track_id = str(no_track_id*-1)
                        # print(track_id)
                    category = group["properties"]["category"]
                    sub_category = group["properties"]["sub_category"]
                    objects = group["objects"]
                    in_cameras = {}
                    position,rotation,dimension = None,None,None
                    for object in objects:
                        if object["sensor"] == "LidarFusion": 
                            position = object["cube"]["position"]
                            rotation = object["cube"]["rotation"]
                            dimension = object["cube"]["dimension"]
                        elif object["sensor"] in camera_names:
                            sensor_name = object["sensor"]
                            try:
                                xyxy = [
                                    object["points"][0]["x"],
                                    object["points"][0]["y"],
                                    object["points"][1]["x"],
                                    object["points"][1]["y"]
                                ]
                            except:
                                xyxy = [-1,-1,-1,-1]
                            try:
                                occupy = object["properties"]['occupy']
                            except:
                                occupy = "-1"
                            in_cameras[sensor_name] = {
                                "xyxy":xyxy,
                                "occupy":occupy
                            }
                    object_info = {
                        "category":category,
                        "sub_category":sub_category,
                        "in_cameras":in_cameras,
                        "position":position,
                        "rotation":rotation,
                        "dimension":dimension
                    }
                    if position is None:continue
                    object_infos[track_id] = object_info
                except:
                    pass

        # print(json.dumps(parameters,indent=4))
        # print(json.dumps(sensor_paths,indent=4))
        # print(json.dumps(object_infos,indent=4))

        new_json = {
            "paths":sensor_paths,
            "ego_pose":ego_pose,
            "parameters":parameters,
            "object_infos":object_infos
        }

        # 保存标注的信息
        paths = new_json["paths"]
        for sensor_name,path in paths.items():
            if "label" == sensor_name:continue
            sensor_folder = os.path.join(clip_folder,sensor_name)
            os.makedirs(sensor_folder,exist_ok=True)
            image_name = os.path.basename(path)
            shutil.copy(path,os.path.join(sensor_folder,image_name))
        
        json_folder = os.path.join(clip_folder,"label")
        os.makedirs(json_folder,exist_ok=True)
        save_json = os.path.join(json_folder,file)
        with open(save_json,"w") as f:
            json.dump(new_json, f,indent=4)
        # print("done!")

def gen_ego_trajectory(clip_folder):
    label_folder = os.path.join(clip_folder,"label")
    files = sorted([file for file in os.listdir(label_folder) if file.endswith(".json")])
    for i, cur_file in  enumerate(tqdm(files,desc=f"generate ego trajectory,{clip_folder.strip(os.sep).split(os.sep)[-1]}")):
        label_path = os.path.join(label_folder,cur_file)
        json_data = json.load(open(label_path,"r"))
        ego_pose = json_data["ego_pose"]
        cur_ego_pose = TransformationmatrixEgo(ego_pose["orientation"],ego_pose['position'])
        traj_mask = np.zeros(len(files)) - 1 # 初始为-1，当前在为0，其余帧为1
        trajs = np.zeros((len(files),3))
        for  j, file in enumerate(files):
            if i==j:
                traj_mask[j] = 0
                traj = np.zeros(3)
            else:
                traj_mask[j] = 1
                label_path_j = os.path.join(label_folder,file)
                json_data_j = json.load(open(label_path_j,"r"))
                ego_pose_j = json_data_j["ego_pose"]
                his_fur_ego_pose = TransformationmatrixEgo(ego_pose_j["orientation"],ego_pose_j['position'])
                traj = (np.linalg.inv(cur_ego_pose)@his_fur_ego_pose)[:3,3]
            trajs[j] = traj
            
        save_path = os.path.join(label_folder,cur_file.replace(".json", "_ego_trajectory.npy"))
        np.save(save_path,np.concatenate([trajs,traj_mask[:,None]],1))


def gen_object_trajectory(clip_folder):
    label_folder = os.path.join(clip_folder,"label")
    files = sorted([file for file in os.listdir(label_folder) if file.endswith(".json")])
    for i, file_i in  enumerate(tqdm(files,desc=f"generate object trajectory,{clip_folder.strip(os.sep).split(os.sep)[-1]}")):
        label_path_i = os.path.join(label_folder,file_i)
        json_data_i = json.load(open(label_path_i,"r"))
        ego_pose_i = json_data_i["ego_pose"]
        T_i_ego2world = TransformationmatrixEgo(ego_pose_i["orientation"],ego_pose_i['position'])
        object_infos_i = json_data_i["object_infos"]
        trackids = list(object_infos_i.keys())

        object_trajectory_dict = {}

        for z , trackid in enumerate(trackids):
            if int(trackid) <0:continue
            position_obj_i = object_infos_i[trackid]["position"]
            rotation_obj_i = object_infos_i[trackid]["rotation"]
            T_i_obj2ego = TransformationmatrixObject(position_obj_i,rotation_obj_i)


            one_obj_trajectory_mask_in_iobj = np.zeros((len(files),4))-1 # 初始化为-1 #在当前目标下目标的轨迹路线
            one_obj_trajectory_mask_in_iego = np.zeros((len(files),4))-1 # 初始化为-1 #在当前自车下目标的轨迹路线

            for j,file_j in enumerate(tqdm(files,desc=f"traversal... {z}:{len(trackids)}:trackid:{trackid}-->{clip_folder.strip(os.sep).split(os.sep)[-1]}:{i}:{len(files)}")):
                if i == j:
                    traj_in_iobj_mask = [0,0,0,0] #当前帧标记位为0，没有跟踪为-1，跟踪为1
                    traj_in_iego_mask = [0,0,0,0] #当前帧标记位为0，没有跟踪为-1，跟踪为1
                else:
                    label_path_j = os.path.join(label_folder,file_j)
                    json_data_j = json.load(open(label_path_j,"r"))
                    ego_pose_j = json_data_j["ego_pose"]
                    T_j_ego2world = TransformationmatrixEgo(ego_pose_j["orientation"],ego_pose_j['position'])
                    object_infos_j = json_data_j["object_infos"]
                    if object_infos_j.get(trackid,None) is None:
                        traj_in_iobj_mask = [0,0,0,-1] # 无跟踪目标
                        traj_in_iego_mask = [0,0,0,-1] # 无跟踪目标
                    else:
                        position_obj_j = object_infos_j[trackid]["position"]
                        rotation_obj_j = object_infos_j[trackid]["rotation"]
                        T_j_obj2ego = TransformationmatrixObject(position_obj_j,rotation_obj_j)
                        
                        
                        traj_in_iego = (np.linalg.inv(T_i_ego2world) @  T_j_ego2world) @ T_j_obj2ego
                        traj_in_iobj = (np.linalg.inv(T_i_ego2world @ T_i_obj2ego) @ T_j_ego2world) @ T_j_obj2ego

                        traj_in_iego_mask = np.zeros(4)
                        traj_in_iobj_mask = np.zeros(4)
                        traj_in_iego_mask[:3] = traj_in_iego[:3,3]
                        traj_in_iobj_mask[:3] = traj_in_iobj[:3,3]
                        traj_in_iego_mask[-1] = 1
                        traj_in_iobj_mask[-1] = 1
                        
                one_obj_trajectory_mask_in_iego[j] = traj_in_iego_mask
                one_obj_trajectory_mask_in_iobj[j] = traj_in_iobj_mask

            object_trajectory_dict[trackid] = {
                "obj_trajectory_in_obj":one_obj_trajectory_mask_in_iobj,
                "obj_trajectory_in_ego":one_obj_trajectory_mask_in_iego,
            }

        save_path = os.path.join(label_folder,os.path.basename(file_i).replace(".json", "_object_trajectory.npy"))
        np.save(save_path,object_trajectory_dict)

        # object_trajectory_dict =  np.load(save_path, allow_pickle=True).item()
        # bev_h=200*8
        # bev_w=80*8
        # bev = np.zeros((bev_h,bev_w,3),dtype=np.uint8) + 255
        # for trackid,trajs in object_trajectory_dict.items():
        #     traj_in_ego =trajs["obj_trajectory_in_ego"]
        #     bev = draw_bev(bev=bev,trajs=traj_in_ego,color=(255,0,0))

        # ego_traj_path = save_path.replace("_object_trajectory.npy","_ego_trajectory.npy")
        # ego_traj = np.load(ego_traj_path)
        # bev = draw_bev(bev=bev,trajs=ego_traj,color=(0,0,255))
        # cv2.imwrite("bev.jpg",bev)
        # print()

def convert_labels(task_info):
    root,clip_name,save_root=task_info
    clip_root = os.path.join(root,clip_name)
    # clip_name = root.strip(os.sep).split(os.sep)[-1]
    clip_folder = os.path.join(save_root,clip_name)
    os.makedirs(clip_folder,exist_ok=True)
    files = sorted([file for file in os.listdir(clip_root) if file.endswith(".json")])
    # 数据格式转换
    convert_engjson2standardjson(clip_root,files,clip_folder)
    # 生成轨迹：以当前在的自车和目标为远点，计算当前的token的轨迹
    gen_ego_trajectory(clip_folder)
    gen_object_trajectory(clip_folder)

if __name__ == '__main__':
    save_root = "/workspace/mpc4-occ/e2e_dataset_10Hz"
    root = "/datasets/xlabel/delivered_data/eng/eng_ad_object_detection_3d_t_filter/1.0.0/13375C/2025/05/08/"
    clip_names = sorted([file for file in os.listdir(root)])
    batch_num = len(clip_names)

    task_args = []
    for clip_name in clip_names:
        task_args.append((root, clip_name, save_root))
    with multiprocessing.Pool(processes=batch_num) as pool:
        # map 方法会将 task_args 中的每个元素作为参数传递给 jbf_info_convert_worker
        pool.map(convert_labels, task_args)

    # for clip_name in clip_names:
    #     clip_root = os.path.join(root,clip_name)
    #     # clip_name = root.strip(os.sep).split(os.sep)[-1]
    #     clip_folder = os.path.join(save_root,clip_name)
    #     os.makedirs(clip_folder,exist_ok=True)
    #     files = sorted([file for file in os.listdir(clip_root) if file.endswith(".json")])
    #     # 数据格式转换
    #     convert_engjson2standardjson(clip_root,files,clip_folder)
    #     # 生成轨迹：以当前在的自车和目标为远点，计算当前的token的轨迹
    #     gen_ego_trajectory(clip_folder)
    #     gen_object_trajectory(clip_folder)

    # # 生成目标车轨迹：
    # clip_folder = "/workspace/afb5szh-01/models/e2e_model/e2e_dataset_10Hz/20250508074556"
    # label_folder = os.path.join(clip_folder,"label")
    # files = sorted([file for file in os.listdir(label_folder) if file.endswith(".json")])
    # for i, file_i in  enumerate(tqdm(files,desc=f"generate object trajectory,{clip_folder.strip(os.sep).split(os.sep)[-1]}")):
    #     label_path_i = os.path.join(label_folder,file_i)
    #     json_data_i = json.load(open(label_path_i,"r"))
    #     ego_pose_i = json_data_i["ego_pose"]
    #     T_i_ego2world = TransformationmatrixEgo(ego_pose_i["orientation"],ego_pose_i['position'])
    #     object_infos_i = json_data_i["object_infos"]
    #     trackids = list(object_infos_i.keys())

    #     object_trajectory_dict = {}

        
        
    #     for z , trackid in enumerate(trackids):
    #         if int(trackid) <0:continue
    #         position_obj_i = object_infos_i[trackid]["position"]
    #         rotation_obj_i = object_infos_i[trackid]["rotation"]
    #         T_i_obj2ego = TransformationmatrixObject(position_obj_i,rotation_obj_i)


    #         one_obj_trajectory_mask_in_iobj = np.zeros((len(files),4))-1 # 初始化为-1 #在当前目标下目标的轨迹路线
    #         one_obj_trajectory_mask_in_iego = np.zeros((len(files),4))-1 # 初始化为-1 #在当前自车下目标的轨迹路线

    #         for j,file_j in enumerate(tqdm(files,desc=f"traversal... {z}:{len(trackids)}:trackid:{trackid}-->{clip_folder.strip(os.sep).split(os.sep)[-1]}:{i}:{len(files)}")):
    #             if i == j:
    #                 traj_in_iobj_mask = [0,0,0,0] #当前帧标记位为0，没有跟踪为-1，跟踪为1
    #                 traj_in_iego_mask = [0,0,0,0] #当前帧标记位为0，没有跟踪为-1，跟踪为1
    #             else:
    #                 label_path_j = os.path.join(label_folder,file_j)
    #                 json_data_j = json.load(open(label_path_j,"r"))
    #                 ego_pose_j = json_data_j["ego_pose"]
    #                 T_j_ego2world = TransformationmatrixEgo(ego_pose_j["orientation"],ego_pose_j['position'])
    #                 object_infos_j = json_data_j["object_infos"]
    #                 if object_infos_j.get(trackid,None) is None:
    #                     traj_in_iobj_mask = [0,0,0,-1] # 无跟踪目标
    #                     traj_in_iego_mask = [0,0,0,-1] # 无跟踪目标
    #                 else:
    #                     position_obj_j = object_infos_j[trackid]["position"]
    #                     rotation_obj_j = object_infos_j[trackid]["rotation"]
    #                     T_j_obj2ego = TransformationmatrixObject(position_obj_j,rotation_obj_j)
                        
                        
    #                     traj_in_iego = (np.linalg.inv(T_i_ego2world) @  T_j_ego2world) @ T_j_obj2ego
    #                     traj_in_iobj = (np.linalg.inv(T_i_ego2world @ T_i_obj2ego) @ T_j_ego2world) @ T_j_obj2ego

    #                     traj_in_iego_mask = np.zeros(4)
    #                     traj_in_iobj_mask = np.zeros(4)
    #                     traj_in_iego_mask[:3] = traj_in_iego[:3,3]
    #                     traj_in_iobj_mask[:3] = traj_in_iobj[:3,3]
    #                     traj_in_iego_mask[-1] = 1
    #                     traj_in_iobj_mask[-1] = 1
                        
    #             one_obj_trajectory_mask_in_iego[j] = traj_in_iego_mask
    #             one_obj_trajectory_mask_in_iobj[j] = traj_in_iobj_mask

    #         object_trajectory_dict[trackid] = {
    #             "obj_trajectory_in_obj":one_obj_trajectory_mask_in_iobj,
    #             "obj_trajectory_in_ego":one_obj_trajectory_mask_in_iego,
    #         }

    #     save_path = os.path.join(label_folder,os.path.basename(file_i).replace(".json", "_object_trajectory.npy"))
    #     np.save(save_path,object_trajectory_dict)

    #     object_trajectory_dict =  np.load(save_path, allow_pickle=True).item()
    #     bev_h=200*8
    #     bev_w=80*8
    #     bev = np.zeros((bev_h,bev_w,3),dtype=np.uint8) + 255
    #     for trackid,trajs in object_trajectory_dict.items():
    #         traj_in_ego =trajs["obj_trajectory_in_ego"]
    #         bev = draw_bev(bev=bev,trajs=traj_in_ego,color=(255,0,0))

    #     ego_traj_path = save_path.replace("_object_trajectory.npy","_ego_trajectory.npy")
    #     ego_traj = np.load(ego_traj_path)
    #     bev = draw_bev(bev=bev,trajs=ego_traj,color=(0,0,255))
    #     cv2.imwrite("bev.jpg",bev)
    #     print()


