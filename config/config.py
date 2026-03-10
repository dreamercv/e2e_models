import os

configs = {

    "device":"cuda",# cuda cpu


    "is_train": True,
    "batch_size": 1,
    "num_workers": 1,
    "epoch": 100,
    "seq_len": 2,


    "log_dir": "/home/fb/project/models/Sparse4D-main/projects/e2e_models/logs",
    "log_save_interval": 100,
    "log_print_interval": 10,

    "load_types": {  # 具体加载哪几种数据，加载不同的数据，会对应着不同的任务
        "dynamic": True,
        "static": True,
        "dynamic_static": False,
        "e2e": False,
    },
    # 总的参数
    "grid_conf": {
        'xbound': [-80.0, 120.0, 1],
        'ybound': [-40.0, 40.0, 1],
        'zbound': [-2.0, 4.0, 1.0]
    },

    "clip_paths": {
        "dynamic": ["/home/fb/project/models/Sparse4D-main/projects/e2e_models/dataset/clip_dataset/det.txt"],
        "static": ["/home/fb/project/models/Sparse4D-main/projects/e2e_models/dataset/clip_dataset/map.txt"],
        "dynamic_static": [],
        "e2e": [],
    },
    "camera_infos": {
        "FrontCam02": {
            "resize_lim": [(0.1, 0.1), (0.17, 0.23), (0.35, 0.45)],
            "bot_pct_lim": [(0.15, 0.25), (0.27, 0.37), (0.37, 0.47)]
        },  # 前 大
        "RearCam01": {"resize_lim": [(0.2, 0.3)], "bot_pct_lim": [(0.2, 0.5)]},  # 后方
        "SideFrontCam01": {"resize_lim": [(0.2, 0.3)], "bot_pct_lim": [(0.2, 0.5)]},  # 左前
        "SideFrontCam02": {"resize_lim": [(0.2, 0.3)], "bot_pct_lim": [(0.2, 0.5)]},  # 右前
        "SideRearCam01": {"resize_lim": [(0.2, 0.3)], "bot_pct_lim": [(0.2, 0.5)]},  # 左后
        "SideRearCam02": {"resize_lim": [(0.2, 0.3)], "bot_pct_lim": [(0.2, 0.5)]}  # 右后
    },
    'rot_lim': (-0.0, 0.0),
    "flip": False,

    'final_dim': (128, 384),

    "mode": "static",  # 默认加载动态数据
    

    "train_clips": 2,  # -1 所有，[]指定几个，xxx.txt写进txt中指定的，>0前N个

    "total_len": 20 + 1 + 50,  # 一共71帧，当前帧是第21帧，往前20帧，往后50帧 ，最后一帧索引对应着70 ，未来50帧为了获取真值
    "current_frame_index": 20,  # 0 1 2 3 4 5 ... [20] 21 22 23 ... 70

    "task_flag": {
        "det2D": False,
        "det3D": True,
        "obj_dynamic_traj": True,

        "map2D": False,
        "map3D": True,
        "e2e_static_traj": True,

        "e2e_dynamic_traj": True,
    },
    "task_class_names": {
        "dynamic": ["det2D", "det3D", "obj_dynamic_traj"],
        "static": ["map3D", "map2D", "e2e_static_traj"],
        "e2e": [
            "e2e_dynamic_traj"
        ],
        "dynamic_static": [
            "det2D", "det3D", "obj_dynamic_traj",
            "map3D", "map2D", "e2e_static_traj",
            "e2e_dynamic_traj"
        ],
    },
    "task_indexs": {  # 在每次getitem时，随机取值 ,其实这里不需要随随机是为了增加间隔，不同的车速就相当于增加了间隔,
        # 在dyo的label上取值
        "det2D": [0, 20],  # 起始帧和结束帧                #输入N帧输出N帧
        "det3D": [0, 20],  # 起始帧和结束帧              #输入N帧输出N帧
        "obj_dynamic_traj": [0, 20],  # 起始帧和结束帧    #输入N帧，预测最后一帧的轨迹
        # 在lane的label上取值
        "map2D": [0, 20],  # 起始帧和结束帧                #输入N帧输出N帧
        "map3D": [0, 20],  # 起始帧和结束帧              #输入N帧输出N帧
        "e2e_dynamic_traj": [0, 20],  # 起始帧和结束帧    #输入N帧，预测最后一帧的轨迹

        # 在自车位姿上取值
        "e2e_static_traj": [0, 20],  # 起始帧和结束帧    #输入N帧，预测最后一帧的轨迹
    },
    "task_index_random": False,  # 是否使用随机的index，即在历史20帧内随机选择seq_len帧
    

    # 预测任务的参数
    "his_lens": {  # 历史长度 ,对于预测任务来说，需要输入历史真值才能预测未来真值，所以上述task_indexs需要减去历史长度
        "obj_dynamic_traj": 5,  # seq_len 必须大于等于 his_lens 否则无法制作真值
        "e2e_dynamic_traj": 5,
    },
    "fur_lens": {  # 未来长度
        "obj_dynamic_traj": 50,
        "e2e_dynamic_traj": 50,
    },  # his_lens , fur_lens 即输入历史真值预测未来轨迹
    "frequency": {  # 频率 10Hz
        "obj_dynamic_traj": 10,  # 真值插值成10Hz,即5秒内真值为50个点
        "e2e_dynamic_traj": 10,  # 真值插值成10Hz,即5秒内真值为50个点
    },
    "num_cams":8, # 环视相机个数
    "camera_names": [
        "FrontCam02",  # 前
        "RearCam01",  # 后方
        "SideFrontCam01",  # 左前
        "SideFrontCam02",  # 右前
        "SideRearCam01",  # 左后
        "SideRearCam02"  # 右后
    ],
    # 3D 检测类别：与 convert_det 输出的 object_infos[].sub_category 对应，用于映射到类别下标
    "det_class_names": [
        "pedestrian",  # 人
        "car", "truck", "bus",  # 车
        "bicycle", "motorcycle"  # 骑行者
    ],
    "det_gt_names": ["x", "y", "z", "w", "l", "h", "yaw", "vx", "vy", "vz"],

    "gt_names": {  # 真值对应的名称，必须与dataset曾一一对应
        "dynamic": [
            "gt_labels_det2D", "gt_bboxes_det2D",
            "gt_labels_det3D", "gt_bboxes_det3D", "gt_labels_det3D_mask", "gt_bboxes_det3D_mask",
            "dynamic_trackids", "dynamic_trajs", "dynamic_traj_masks"
        ],
        "static": [
            "gt_labels_map2D", "gt_bboxes_map2D",
            "gt_labels_map3D", "gt_bboxes_map3D", "gt_pts_map3D",
            "gt_e2e_static_traj"
        ],
        "e2e": ["gt_e2e_dynamic_traj"]
    },
    "input_names": ["x", "rots", "trans", "intrins", "distorts", "post_rots", "post_trans", "theta_mats"],

    # 模型相关
    "img_outchannels": 32,
    # 2d
    "det_2d_num": 10,
    "map_2d_num": 3,
    # det3D
    "det_3d_head": {
        "num_anchor": 20,
        "embed_dims": 256,
        "num_decoder": 6,
        "num_single_frame_decoder": 5,
        "num_classes": 10,
        "bev_bounds": None,
        "anchor_init": None,
        "decouple_attn": True,
        "num_heads": 8,
        "dropout": 0.1,
        "feedforward_dims": 1024,
        "use_dn": True,
        "num_dn_groups": 10,
        "dn_noise_scale": 0.5,
        "max_dn_gt": 32,
        "add_neg_dn": True,
        "reg_weights": None,
        "use_decoder": True,
        "decoder_num_output": 300,
        "decoder_score_threshold": None,
    },
    # tracking
    "track_head": {
        "feat_dim": 256,
        "anchor_dim": 11,
        "num_heads": 8,
        "dropout": 0.5,
        "embed_dim": 256
    },
    "map_3d_head": {

        "bev_feat_dim": 256,
        "embed_dims": 256,
        "num_vec":  50,
        "num_pts_per_vec": 20,
        "num_pts_per_gt_vec":  20,
        "num_classes":  3,
        "num_decoder_layers":  2,
        "num_heads": 4,
        "dropout": 0.1,
        "feedforward_dims":  512,
        "grid_conf":None,
        "bev_bounds":None,

        "with_box_refine":True,
        "use_instance_pts" : True,
        "dir_interval": 1,
        "cls_weight":1.0,
        "reg_weight": 1.0,
        "pts_weight":1.0,
        "loss_pts_src_weight": 1.0,
        "loss_pts_dst_weight": 1.0,
        "loss_dir_weight": 0.005,
        "aux_loss_weight": 0.5,

        "use_maptr_decoder": True,


        "maptr_im2col_step": 192,
        "maptr_num_levels":1,
        "maptr_num_points":4,
        "bbox_coder_max_num" :50
    }

}