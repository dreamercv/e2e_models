# 训练：Clip 内按时间逐帧流式前向（降低显存峰值）

## 1. 背景与目标

默认训练中，一次 `forward` 会把整个 clip 的时序 `T`、任务维 `M`、相机数 `N` 一起摊平，送入 2D backbone 与 BEV，显存占用近似为「batch × M × T × N」量级，容易 OOM。

本功能在 **不改变 DataLoader 与 epoch/step 外层逻辑** 的前提下，在 `Model` 内部增加 **clip 内按帧顺序的第三层循环**：每一帧单独做 2D + 单步 BEV，再把 **上一帧融合后的 BEV** 与当前帧做与原版一致的 **theta 对齐 + 融合**，最终堆叠成与原先相同的 `(B*M*T, C, H, W)` 供 det3d / map3d 计算损失。

## 2. 如何开启

在 `config/config.py` 中设置：

```python
"train_stream_clip_frames": True,
```

`train.py` 调用模型时会传入：

```python
stream_clip_frames=configs.get("train_stream_clip_frames", False)
```

默认为 `False`，与改前行为完全一致。

## 3. 数据流对比

### 3.1 关闭流式（默认）

- 输入 `x`：`(b, m, t, n, c, h, w)`
- 一次 `rearrange` 为 `(b*m*t*n, c, h, w)`，整段 clip 同时过 `image_backbone`、`det2d`、`map2d`、`BEVBackbone.forward`
- BEV 内部一次性完成 `T` 帧的 pre-align，再在模块内做帧间 `theta` 融合

### 3.2 开启流式

- 外层仍是一个 batch 一条完整 clip（形状不变）
- 对 `ti = 0 .. t-1` **顺序**循环：
  1. 取 `x[:, :, ti, ...]` → `(b*m*n, c, h, w)` → `image_backbone` → `det2d` / `map2d`
  2. `BEVBackbone.encode_single_frame(...)`：仅当前帧的 pre-align BEV，形状 `(b*m, C, H, W)`
  3. `ti == 0`：当前帧融合结果 = 该 pre-align 输出（与原版第 0 帧一致）
  4. `ti > 0`：`fuse_with_prev(prev_fused, cur_pre, theta_bm[:, ti])`，与原版 `forward` 中 `i>0` 分支一致（先 warp 上一帧融合特征，再与当前帧 concat 过 `algin_fusion`）
  5. 将每帧结果 `stack` 为 `(b*m, t, C, H, W)`，再 `reshape` 为 `(b, m, t, C, H, W)`，后续 det3d / map3d 与原先相同

**注意**：clip 内帧顺序固定为时间下标递增，不在此循环内打乱。

## 4. 涉及代码位置

| 模块 | 文件 | 说明 |
|------|------|------|
| BEV | `backbone/bev_backbone.py` | `_forward_pre_align_features`：从原 `forward` 抽出的 splat→多视角→高度融合；`encode_single_frame`：单时间步 pre-align；`fuse_with_prev`：与原版帧间融合一致；`forward` 改为复用上述逻辑 |
| 模型 | `model/models.py` | `_forward_stream_clip_frames`：按帧循环 + 汇总 2D 输出 + 与原逻辑相同的 3D 分支与 loss；`forward(..., stream_clip_frames=False)` |
| 配置 | `config/config.py` | `train_stream_clip_frames` |
| 训练 | `train.py` | 传入 `stream_clip_frames` |

## 5. 与原版的一致性

- **BEV 时序融合数学**：与 `BEVBackbone.forward` 中 `theta_mats[:, i]` + `warp_feature` + `algin_fusion` 一致，仅由「一次算完 T 帧再融合」改为「逐帧融合并携带状态」。
- **det3d / map3d 输入**：最终仍为 `(b*m*t, C, H, W)` 的展平形式，损失定义与聚合方式不变（在未改 loss 代码的前提下）。
- **2D 分支输出**：各帧结果在 batch 维拼接，与一次性 forward 得到的 `(b*m*t*n)` 级别 batch 拼接方式一致。

## 6. 显存与迭代语义

- **显存**：2D backbone 与 BEV 的单次前向规模从「整段 T」降为「单帧」，峰值通常明显下降；det3d/map3d 仍接收完整 `B*M*T` 的 BEV 张量，其显存与原先同量级。
- **优化步数**：每个 DataLoader batch 仍对应 **一次** `optimizer.step()`（与改前相同）；若需要按「帧」计 iteration / 学习率，需另行改 `train.py` 与 scheduler。

## 7. 限制与后续可做

- **DataLoader 未改**：每个 step 仍加载完整 clip；若需「每个 step 只加载一帧」以进一步省内存或 I/O，需改 `dataset`/`collate`。
- **推理脚本**：若单独调用 `Model.forward`，需自行传入 `stream_clip_frames=True` 才会走流式路径（训练默认由 config 控制）。

## 8. 相关文档

- BEV 与对齐矩阵：`doc/BEV_BACKBONE_AND_DET3D_LOSS_NOTES.md`、`doc/BEV_BACKBONE_AND_DYNAMIC_TRAINING_NOTES.md`
