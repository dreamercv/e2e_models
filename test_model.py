import numpy as np
import torch
import torch.nn as nn

from backbone.image_backbone import ImageBackBone
from model_2d.det2d_model import ModelDet2D
from model_2d.seg2d_model import ModelSeg2D
from backbone.bev_backbone import BEVBackbone

from sparse4d.instance_bank import InstanceBank
from sparse4d.detection3d_blocks import (
    SparseBox3DEncoder,
    SparseBox3DRefinementModule,
    SparseBox3DKeyPointsGenerator,
)
from sparse4d.bev_aggregation import BEVFeatureAggregation
from sparse4d.head import Sparse4DHead, MHAWrapper, SimpleFFN
from sparse4d.decoder import SparseBox3DDecoder
from sparse4d.dn_sampler import DenoisingSampler
from sparse4d.losses import FocalLoss, SparseBox3DLoss
from sparse4d.track_head import TrackHead, track_affinity_loss, decode_track

from maptr import build_map_head


def build_sparse4d_bev_head(
    num_anchor: int = 900,
    embed_dims: int = 256,
    num_decoder: int = 6,
    num_single_frame_decoder: int = 5,
    num_classes: int = 10,
    bev_bounds=([-80.0, 120.0, 1.0], [-40.0, 40.0, 1.0]),
    anchor_init: np.ndarray = None,
    decouple_attn: bool = True,
    num_heads: int = 8,
    dropout: float = 0.1,
    feedforward_dims: int = 1024,
    use_dn: bool = True,
    num_dn_groups: int = 10,
    dn_noise_scale: float = 0.5,
    max_dn_gt: int = 32,
    add_neg_dn: bool = True,
    reg_weights: list = None,
    use_decoder: bool = False,
    decoder_num_output: int = 300,
    decoder_score_threshold: float = None,
):
    """从 model.py 拷贝的 Sparse4D BEV head 构建函数，用于测试。"""
    if anchor_init is None:
        anchor_init = np.zeros((num_anchor, 11), dtype=np.float32)
    if reg_weights is None:
        reg_weights = [2.0] * 3 + [0.5] * 3 + [0.0] * 5

    instance_bank = InstanceBank(
        num_anchor=num_anchor,
        embed_dims=embed_dims,
        anchor=anchor_init,
        anchor_handler=None,
        num_temp_instances=0,
        default_time_interval=0.5,
        confidence_decay=0.6,
        feat_grad=False,
    )
    anchor_encoder = SparseBox3DEncoder(
        embed_dims=[128, 32, 32, 64] if decouple_attn else 256,
        vel_dims=3,
        mode="cat" if decouple_attn else "add",
        output_fc=not decouple_attn,
        in_loops=1,
        out_loops=4 if decouple_attn else 2,
    )
    gnn_dim = embed_dims * 2 if decouple_attn else embed_dims
    graph_model = MHAWrapper(gnn_dim, num_heads, dropout=dropout, batch_first=True)
    norm_layer = nn.LayerNorm(embed_dims)
    ffn = SimpleFFN(embed_dims, feedforward_dims, dropout=dropout)
    kps = SparseBox3DKeyPointsGenerator(embed_dims=embed_dims, num_learnable_pts=0, fix_scale=((0.0, 0.0, 0.0),))
    bev_agg = BEVFeatureAggregation(
        embed_dims=embed_dims,
        bev_bounds=bev_bounds,
        kps_generator=kps,
        proj_drop=dropout,
        residual_mode="add",
    )
    refine_layer = SparseBox3DRefinementModule(
        embed_dims=embed_dims,
        output_dim=11,
        num_cls=num_classes,
        refine_yaw=True,
        with_cls_branch=True,
        with_quality_estimation=True,
    )

    operation_order = (
        ["gnn", "norm", "deformable", "ffn", "norm", "refine"] * num_single_frame_decoder
        + ["gnn", "norm", "deformable", "ffn", "norm", "refine"] * (num_decoder - num_single_frame_decoder)
    )
    operation_order = operation_order[2:]

    sampler = None
    loss_cls = None
    loss_reg = None
    if use_dn:
        sampler = DenoisingSampler(
            num_dn_groups=num_dn_groups,
            dn_noise_scale=dn_noise_scale,
            max_dn_gt=max_dn_gt,
            add_neg_dn=add_neg_dn,
            reg_weights=reg_weights,
        )
    loss_cls = FocalLoss(alpha=0.25, gamma=2.0, ignore_index=num_classes)
    loss_reg = SparseBox3DLoss(reg_weights=reg_weights, loss_centerness=True, loss_yawness=True)

    decoder = None
    if use_decoder:
        decoder = SparseBox3DDecoder(
            num_output=decoder_num_output,
            score_threshold=decoder_score_threshold,
            sorted=True,
        )

    head = Sparse4DHead(
        instance_bank=instance_bank,
        anchor_encoder=anchor_encoder,
        graph_model=graph_model,
        norm_layer=norm_layer,
        ffn=ffn,
        deformable_model=bev_agg,
        refine_layer=refine_layer,
        operation_order=operation_order,
        num_single_frame_decoder=num_single_frame_decoder,
        decouple_attn=decouple_attn,
        sampler=sampler,
        decoder=decoder,
        loss_cls=loss_cls,
        loss_reg=loss_reg,
        reg_weights=reg_weights,
        gt_cls_key="gt_labels_3d",
        gt_reg_key="gt_bboxes_3d",
        cls_threshold_to_reg=0.05,
    )
    head.init_weights()
    return head


def build_track_head(
    feat_dim: int = 256,
    anchor_dim: int = 11,
    num_heads: int = 8,
    dropout: float = 0.1,
    embed_dim: int = 256,
):
    """从 model.py 拷贝的跟踪 head 构建函数。"""
    return TrackHead(
        feat_dim=feat_dim,
        anchor_dim=anchor_dim,
        num_heads=num_heads,
        dropout=dropout,
        embed_dim=embed_dim,
    )


def build_sparse4d_bev_model(
    image_backbone: nn.Module,
    det2d_head:nn.Module,
    seg2d_head:nn.Module,
    bev_backbone: nn.Module,
    det_head: nn.Module,
    track_head: nn.Module = None,
    map_head: nn.Module = None,
    num_temporal_frames: int = 7,
):
    """从 model.py 拷贝的完整模型封装。"""

    class Sparse4DBEVModel(nn.Module):
        def __init__(self, image_backbone,bev_backbone,
                     det2d_head=None,seg2d_head=None,
                     det_head=None, track_head=None, map_head=None, num_temporal_frames=7):
            super().__init__()
            self.image_backbone = image_backbone
            self.bev_backbone = bev_backbone
            self.det2d_head = det2d_head
            self.seg2d_head = seg2d_head
            self.det_head = det_head
            self.track_head = track_head
            self.map_head = map_head
            self.num_temporal_frames = num_temporal_frames

        def forward(
            self,
            x,
            rots,
            trans,
            intrins,
            distorts,
            post_rot3,
            post_tran3,
            theta_mats,
            T_ego_his2curs,
            metas=None,
        ):
            det2d_out,seg2d_out = None,None
            det_out, track_out = None, None
            map_out = None

            x = self.image_backbone(x)
            if self.det_head is not None:
                det2d_out = self.det_head(x)
            if self.seg2d_head is not None:
                seg2d_out = self.seg2d_head(x)

            feature_maps = self.bev_backbone(x, rots, trans, intrins, distorts, post_rot3, post_tran3, theta_mats)

            if self.det_head is not None:
                det_out, seq_features, seq_anchors = self.det_head(feature_maps, metas)
                if self.track_head is not None and T_ego_his2curs is not None:
                    track_affinity = self.track_head(seq_features, seq_anchors, T_ego_his2curs)
                    track_out = {"track_affinity": track_affinity}
                    if self.training and metas is not None and metas.get("gt_track_match") is not None:
                        track_out["loss_track"] = track_affinity_loss(
                            track_affinity, metas["gt_track_match"], ignore_index=-1
                        )
                    if not self.training:
                        track_ids, positions = decode_track(track_affinity, seq_anchors, use_hungarian=True)
                        track_out["track_ids"] = track_ids
                        track_out["track_positions"] = positions


            if self.map_head is not None:
                Bt, C, H, W = feature_maps.shape
                T = self.num_temporal_frames
                B = Bt // T if (T > 0 and Bt % T == 0) else Bt
                if T > 0 and Bt % T == 0:
                    bev_for_map = feature_maps.view(B, T, C, H, W)[:, -1]
                else:
                    bev_for_map = feature_maps
                map_pred = self.map_head(bev_for_map, metas)
                map_out = {
                    "map_cls_scores": map_pred["map_cls_scores"],
                    "map_bbox_preds": map_pred["map_bbox_preds"],
                    "map_pts_preds": map_pred["map_pts_preds"],
                }
                if self.training and metas is not None:
                    gt_map_bboxes = metas.get("gt_map_bboxes")
                    gt_map_labels = metas.get("gt_map_labels")
                    gt_map_pts = metas.get("gt_map_pts")
                    if gt_map_bboxes is not None and gt_map_labels is not None and gt_map_pts is not None:
                        map_losses = self.map_head.loss(map_pred, gt_map_bboxes, gt_map_labels, gt_map_pts)
                        map_out["loss_map"] = map_losses
                if not self.training:
                    map_out["map_polylines"] = self.map_head.decode(map_pred, score_threshold=0.5)

            return det2d_out,seg2d_out,det_out, track_out, map_out

    return Sparse4DBEVModel(image_backbone,bev_backbone, det2d_head,seg2d_head,det_head, track_head, map_head, num_temporal_frames)

def build_image_backbone(out_channels):
    return ImageBackBone(out_channels=out_channels)

def build_det2d_model(out_channels=256,det_class_num=3):
    return ModelDet2D(out_channels=out_channels,det_class_num=det_class_num)

def build_seg2d_model(out_channels=256,seg_class_num=3,pos_weight=20.):
    return ModelSeg2D(out_channels=out_channels,seg_class_num=seg_class_num)

def build_bev_backbone(channels,grid_conf,input_size=(128, 384),num_temporal=7):
    model = BEVBackbone(channels=channels,grid_conf=grid_conf, input_size=input_size,num_temporal=num_temporal)
    return model

def build_full_model(
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
    grid_conf = {
        "xbound": [-80.0, 120.0, 1],
        "ybound": [-40.0, 40.0, 1],
        "zbound": [-2.0, 4.0, 1.0],
    }
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
    bev_h, bev_w = 80, 40
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


def build_dummy_inputs(
    batch_size: int = 2,
    num_temporal_frames: int = 3,
    num_cams: int = 1,
) -> tuple:
    """构造一组假的相机输入和几何变换，用于快速前向测试。"""
    B, T, Cams = batch_size, num_temporal_frames, num_cams
    H_img, W_img = 128, 384

    x = torch.randn(B, T, Cams, 3, H_img, W_img)

    # 几何相关全部用单位阵 / 零向量占位
    rots = torch.eye(3).view(1, 1, 1, 3, 3).expand(B, T, Cams, 3, 3).clone()
    trans = torch.zeros(B, T, Cams, 1, 3)
    intrins = torch.eye(3).view(1, 1, 1, 3, 3).expand(B, T, Cams, 3, 3).clone()
    distorts = torch.zeros(B, T, Cams, 1, 8)
    post_rot3 = torch.eye(3).view(1, 1, 1, 3, 3).expand(B, T, Cams, 3, 3).clone()
    post_tran3 = torch.zeros(B, T, Cams, 1, 3)
    theta_mats = torch.zeros(B, T, 2, 3)
    T_ego_his2curs = torch.eye(4).view(1, 1, 4, 4).expand(B, T, 4, 4).clone()

    return (
        x,
        rots,
        trans,
        intrins,
        distorts,
        post_rot3,
        post_tran3,
        theta_mats,
        T_ego_his2curs,
    )


def build_dummy_metas(
    batch_size: int,
    num_temporal_frames: int,
    num_anchor: int,
    num_det_classes: int,
    num_map_classes: int,
    num_pts_per_gt_vec: int = 20,
) -> dict:
    """构造假的 GT，测试检测 + 跟踪 + MapTR 的 loss 和匈牙利匹配是否能正常跑。"""
    B, T, N = batch_size, num_temporal_frames, num_anchor

    # 检测 GT（3D boxes）
    gt_bboxes_3d = []
    gt_labels_3d = []
    for b in range(B):
        n_det = 5
        gt_bboxes_3d.append(torch.randn(n_det, 11))  # 11-dim box 编码
        gt_labels_3d.append(
            torch.randint(low=0, high=num_det_classes, size=(n_det,), dtype=torch.long)
        )

    # MapTR GT（2D 线 + 多点）
    gt_map_bboxes = []
    gt_map_labels = []
    gt_map_pts = []
    for b in range(B):
        n_map = 4
        # xyxy in pc_range [-15,15] x [-30,30]
        x1y1 = torch.rand(n_map, 2)
        x2y2 = x1y1 + 0.5 * torch.rand(n_map, 2)
        bboxes = torch.cat([x1y1, x2y2], dim=-1)
        gt_map_bboxes.append(bboxes)
        gt_map_labels.append(
            torch.randint(low=0, high=num_map_classes, size=(n_map,), dtype=torch.long)
        )
        pts = torch.rand(n_map, num_pts_per_gt_vec, 2)
        gt_map_pts.append(pts)

    # 跟踪 GT：gt_track_match (B, T, N)
    gt_track_match = torch.full((B, T, N), -1, dtype=torch.long)
    for b in range(B):
        # 第一帧全部 ignore
        # 后续帧：简单地 1:1 匹配到上一帧同 index
        for t in range(1, T):
            gt_track_match[b, t] = torch.arange(N, dtype=torch.long)

    metas = {
        "gt_bboxes_3d": gt_bboxes_3d,
        "gt_labels_3d": gt_labels_3d,
        "gt_map_bboxes": gt_map_bboxes,
        "gt_map_labels": gt_map_labels,
        "gt_map_pts": gt_map_pts,
        "gt_track_match": gt_track_match,
    }
    return metas


def _fmt_shape(x):
    if isinstance(x, torch.Tensor):
        return str(tuple(x.shape))
    if isinstance(x, (list, tuple)):
        return f"list(len={len(x)})"
    return str(type(x).__name__)


def run_full_pipeline_test():
    """一次 forward 跑通 + 各模块 loss 形状/值 + 各模块 decode 形状，确认所有模块跑通。"""
    torch.manual_seed(0)
    batch_size = 2
    num_temporal_frames = 7
    num_anchor = 32
    num_det_classes = 10
    num_map_classes = 3

    model = build_full_model(
        num_anchor=num_anchor,
        num_det_classes=num_det_classes,
        num_map_classes=num_map_classes,
        num_temporal_frames=num_temporal_frames,
        use_det_decoder=True,
        use_dn=True,  # 已修复 head 中 DN 与 B*T 维度：将 dn_anchor 等按 T 扩展后拼接
    )
    inputs = build_dummy_inputs(
        batch_size=batch_size,
        num_temporal_frames=num_temporal_frames,
        num_cams=1,
    )
    metas = build_dummy_metas(
        batch_size=batch_size,
        num_temporal_frames=num_temporal_frames,
        num_anchor=num_anchor,
        num_det_classes=num_det_classes,
        num_map_classes=num_map_classes,
        num_pts_per_gt_vec=20,
    )
    (
        x,
        rots,
        trans,
        intrins,
        distorts,
        post_rot3,
        post_tran3,
        theta_mats,
        T_ego_his2curs,
    ) = inputs
    forward_args = (
        x, rots, trans, intrins, distorts,
        post_rot3, post_tran3, theta_mats, T_ego_his2curs,
    )

    # ---------- 1) Forward 一次是否跑通 ----------
    print("=" * 60)
    print("1) Forward 一次流程")
    print("=" * 60)
    model.train()
    det2d_out,seg2d_out,det_out, track_out, map_out = model(
        *forward_args, metas=metas,
    )
    print("  [OK] Forward 完成。")
    print("  det_out keys:", list(det_out.keys()) if isinstance(det_out, dict) else type(det_out))
    print("  track_out:", "None" if track_out is None else list(track_out.keys()))
    print("  map_out:", "None" if map_out is None else list(map_out.keys()))

    # ---------- 2) 各模块 loss 形状与值 ----------
    print()
    print("=" * 60)
    print("2) 各模块 Loss 形状与值")
    print("=" * 60)

    # 检测 loss（需显式调用 head.loss；use_dn=False 时 sampler 为 None 返回 {}）
    det_loss = model.det_head.loss(det_out, metas)
    if det_loss:
        print("  [Detection]")
        for k, v in sorted(det_loss.items()):
            val = float(v) if isinstance(v, torch.Tensor) and v.numel() == 1 else v
            print(f"    {k}: shape={_fmt_shape(v)}, value={val}")
    else:
        print("  [Detection] 未计算（use_dn=False 时无 sampler）")
        for k in ("classification", "prediction", "quality"):
            if k in det_out:
                v = det_out[k]
                if isinstance(v, (list, tuple)):
                    print(f"    det_out[{k}]: len={len(v)}, elem_shape={v[0].shape}")
                else:
                    print(f"    det_out[{k}] shape: {v.shape}")

    # 跟踪 loss
    if track_out is not None and "loss_track" in track_out:
        lt = track_out["loss_track"]
        print("  [Track]")
        print(f"    loss_track: shape={_fmt_shape(lt)}, value={float(lt)}")
    else:
        print("  [Track] loss_track 未在输出中")

    # MapTR loss
    if map_out is not None and "loss_map" in map_out:
        print("  [MapTR]")
        for k, v in sorted(map_out["loss_map"].items()):
            shape = v.shape if isinstance(v, torch.Tensor) else "scalar"
            val = float(v) if isinstance(v, torch.Tensor) and v.numel() == 1 else v
            print(f"    {k}: shape={_fmt_shape(v)}, value={val}")
    else:
        print("  [MapTR] loss_map 未在输出中")

    # ---------- 3) 各模块 Decode 形状 ----------
    print()
    print("=" * 60)
    print("3) 各模块 Decode 输出形状")
    print("=" * 60)
    model.eval()
    with torch.no_grad():
        det2d_out_eval, seg2d_out_eval, det_out_eval, track_out_eval, map_out_eval = model(
            *forward_args, metas={},
        )

    # 检测 decoder
    head = model.det_head
    if getattr(head, "decoder", None) is not None:
        det_results = head.decoder.decode(
            det_out_eval["classification"],
            det_out_eval["prediction"],
            quality=det_out_eval.get("quality"),
            instance_id=det_out_eval.get("instance_id"),
        )
        print("  [Detection Decoder]")
        print(f"    num_samples (B*T): {len(det_results)}")
        if det_results:
            r0 = det_results[0]
            for k, v in r0.items():
                if isinstance(v, torch.Tensor):
                    print(f"    {k}: {v.shape}")
    else:
        print("  [Detection Decoder] 未构建")

    # MapTR decode
    if map_out_eval is not None and "map_polylines" in map_out_eval:
        polylines = map_out_eval["map_polylines"]
        print("  [MapTR Decode]")
        print(f"    num_samples (B): {len(polylines)}")
        if polylines:
            for k, v in polylines[0].items():
                if isinstance(v, torch.Tensor):
                    print(f"    {k}: {v.shape}")
    else:
        print("  [MapTR Decode] map_polylines 未在输出中")

    # 跟踪 decode
    if track_out_eval is not None:
        print("  [Track Decode]")
        if "track_ids" in track_out_eval:
            print(f"    track_ids: {track_out_eval['track_ids'].shape}")
        if "track_positions" in track_out_eval:
            print(f"    track_positions: {track_out_eval['track_positions'].shape}")
    else:
        print("  [Track Decode] track_out 为 None")

    print()
    print("=" * 60)
    print("所有模块已跑通。")
    print("=" * 60)


if __name__ == "__main__":
    run_full_pipeline_test()

