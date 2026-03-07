import torch

import torch.nn as nn


from model.build_model import *



def model(
    config: dict,
    num_anchor: int = 32,
    num_det_classes: int = 10,
    num_map_classes: int = 3,
    num_temporal_frames: int = 3,
    use_det_decoder: bool = False,
    use_dn: bool = False,
) -> nn.Module:
    """构建带检测 + 跟踪 + MapTR 的完整模型，用于自测。use_dn=True 时检测 head 会计算 loss。"""

    image_backbone = build_image_backbone(out_channels=256)

    det2d_head = build_det2d_model(out_channels=256,det_class_num=num_det_classes)

    seg2d_head = build_seg2d_model(out_channels=256,seg_class_num=num_map_classes,pos_weight=20.)

    # BEV 网格与范围（与 model.py 中一致）
    grid_conf = config["grid_conf"]
    # grid_conf = {
    #     "xbound": [-80.0, 120.0, 1],
    #     "ybound": [-40.0, 40.0, 1],
    #     "zbound": [-2.0, 4.0, 1.0],
    # }
    bev_h = int((grid_conf["xbound"][1] - grid_conf["xbound"][0]) / grid_conf["xbound"][2])
    bev_w = int((grid_conf["ybound"][1] - grid_conf["ybound"][0]) / grid_conf["ybound"][2])
    bev_backbone = build_bev_backbone(channels=256,grid_conf=grid_conf,input_size=(128,384),num_temporal=num_temporal_frames)

    # Sparse4D 检测 head（减少 anchor 数，方便测试）
    det_head = build_sparse4d_bev_head(
        num_anchor=num_anchor,
        embed_dims=256,
        num_decoder=2,
        num_single_frame_decoder=2,
        num_classes=num_det_classes,
        bev_bounds=(
            [grid_conf["xbound"][0], grid_conf["xbound"][1], 1.0],
            [grid_conf["ybound"][0], grid_conf["ybound"][1], 1.0],
        ),
        use_dn=use_dn,
        use_decoder=use_det_decoder,
        decoder_num_output=min(100, num_anchor),
        decoder_score_threshold=0.2,
    )

    # 跟踪 head
    track_head = build_track_head(
        feat_dim=256,
        anchor_dim=11,
        num_heads=8,
        dropout=0.1,
        embed_dim=256,
    )

    # MapTR head（使用 maptr 风格 config 构建）
    # point_cloud_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
    point_cloud_range = [
        grid_conf["xbound"][0],
        grid_conf["ybound"][0],
        grid_conf["zbound"][0],
        grid_conf["xbound"][1],
        grid_conf["ybound"][1],
        grid_conf["zbound"][1],
    ]
    xmin, xmax = grid_conf["xbound"][0], grid_conf["xbound"][1]
    ymin, ymax = grid_conf["ybound"][0], grid_conf["ybound"][1]
    bbox_coder_post_center_range = [
        xmin, ymin, xmin, ymin,
        xmax, ymax, xmax, ymax,
    ]
    # bev_h, bev_w = 80, 40
    map_head = build_map_head(
        bev_feat_dim=256,
        embed_dims=256,
        num_vec=50,
        num_pts_per_vec=20,
        num_pts_per_gt_vec=20,
        num_classes=num_map_classes,
        pc_range=point_cloud_range,
        bev_h=bev_h,
        bev_w=bev_w,
        decoder_num_layers=2,
        decoder_num_heads=4,
        decoder_im2col_step=192,
        decoder_feedforward_channels=512,
        bbox_coder_post_center_range=bbox_coder_post_center_range,#[-20, -35, -20, -35, 20, 35, 20, 35],
        bbox_coder_max_num=50,
    )

    model = build_sparse4d_bev_model(
        image_backbone=image_backbone,
        det2d_head=det2d_head,
        seg2d_head=seg2d_head,
        bev_backbone=bev_backbone,
        det_head=det_head,
        track_head=track_head,
        map_head=map_head,
        num_temporal_frames=num_temporal_frames,
    )
    return model

class Model(nn.Module):
    def __init__(self,
                 config,
                 num_temporal_frames=7,
                 modules={}
                 ):
        super().__init__()
        self.config = config



    def forward(self, x):
        return self.model(x)