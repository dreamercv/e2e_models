import os

configs = {
    "epoch":100,
    
    "log_dir":"/home/fb/project/models/Sparse4D-main/projects/e2e_models/logs",
    "log_save_interval":100,
    "log_print_interval":10,



    "load_types":{ # 具体加载哪几种数据，加载不同的数据，会对应着不同的任务
        "dynamic":True,
        "static":True,
        "dynamic_static":False,
        "e2e":False,
    },
    # 总的参数
    "grid_conf" : {
        'xbound': [-80.0, 120.0, 1],
        'ybound': [-40.0, 40.0, 1],
        'zbound': [-2.0, 4.0, 1.0]
    },

    "clip_paths":{
        "dynamic":["/home/fb/project/models/Sparse4D-main/projects/e2e_models/dataset/clip_dataset/det.txt"],
        "static":["/home/fb/project/models/Sparse4D-main/projects/e2e_models/dataset/clip_dataset/map.txt"],
        "dynamic_static":["/home/fb/project/models/Sparse4D-main/projects/e2e_models/dataset/clip_dataset/dynamic_static.txt"],
        "e2e":["/home/fb/project/models/Sparse4D-main/projects/e2e_models/dataset/clip_dataset/e2e.txt"],
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

    "mode":"dynamic",
    "is_train":True,
    "batch_size":1,
    "num_workers":1,

    "train_clips":2, # -1 所有，[]指定几个，xxx.txt写进txt中指定的，>0前N个
    
    "total_len":20+1+50, # 一共71帧，当前帧是第21帧，往前20帧，往后50帧 ，最后一帧索引对应着70 ，未来50帧为了获取真值
    "current_frame_index":20,# 0 1 2 3 4 5 ... [20] 21 22 23 ... 70
    


    "task_flag":{
        "det2D":False,
        "det3D":True,

        "map2D":False,
        "map3D":True,

        "obj_dynamic_traj":False,
        "e2e_static_traj":False,
        "e2e_dynamic_traj":False,
    },
    "task_class_names":{
        "dynamic":["det3D","obj_dynamic_traj","det2D","e2e_dynamic_traj"], 
        "static":["map3D","map2D","e2e_static_traj","e2e_dynamic_traj"],
        "dynamic_static":[
            "det3D","obj_dynamic_traj","det2D",
            "map3D","map2D","e2e_static_traj",
            "e2e_dynamic_traj"
            ],
        "e2e":[
            "e2e_static_traj",
            "e2e_dynamic_traj"
            ],
    },
    "task_indexs":{ # 在每次getitem时，随机取值
        # 在dyo的label上取值
        "det2D":[0,20], # 起始帧和结束帧                #输入N帧输出N帧
        "det3D":  [0,20], # 起始帧和结束帧              #输入N帧输出N帧
        "obj_dynamic_traj": [0,20], # 起始帧和结束帧    #输入N帧，预测最后一帧的轨迹
        # 在lane的label上取值
        "map2D":[0,20], # 起始帧和结束帧                #输入N帧输出N帧
        "map3D":  [0,20], # 起始帧和结束帧              #输入N帧输出N帧

        # 在自车位姿上取值
        "e2e_dynamic_traj": [0,20], # 起始帧和结束帧    #输入N帧，预测最后一帧的轨迹
        "e2e_static_traj":  [0,20], # 起始帧和结束帧    #输入N帧，预测最后一帧的轨迹
    },
    "seq_len":5, 
    "seq_len_random":True,

    # 预测任务的参数
    "his_lens":{ # 历史长度 ,对于预测任务来说，需要输入历史真值才能预测未来真值，所以上述task_indexs需要减去历史长度
        "obj_dynamic_traj":5, # seq_len 必须大于等于 his_lens 否则无法制作真值
        "e2e_dynamic_traj":5,
    },
    "fur_lens":{ # 未来长度
        "obj_dynamic_traj": 50,
        "e2e_dynamic_traj": 50,
    }, # his_lens , fur_lens 即输入历史真值预测未来轨迹
    "frequency":{ # 频率 10Hz
        "obj_dynamic_traj": 10, # 真值插值成10Hz,即5秒内真值为50个点
        "e2e_dynamic_traj": 10, # 真值插值成10Hz,即5秒内真值为50个点
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

