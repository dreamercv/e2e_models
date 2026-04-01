# 训练：Clip 内按时间逐帧流式前向（降低显存峰值）

## 1. 背景与目标

默认训练中，一次 `forward` 会把整个 clip 的时序 `T`、任务维 `M`、相机数 `N` 一起摊平，送入 2D backbone 与 BEV，显存占用近似为「batch × M × T × N」量级，容易 OOM。

本功能在 **不改变 DataLoader 与 epoch/step 外层逻辑** 的前提下，在 `Model` 内部增加 **clip 内按帧顺序的第三层循环**：每一帧单独做 2D + 单步 BEV，再把 **上一帧融合后的 BEV** 与当前帧做与原版一致的 **theta 对齐 + 融合**，最终堆叠成与原先相同的 `(B*M*T, C, H, W)` 供 det3d / map3d 计算损失。

## 2. 如何开启

### 2.1 仅对「已采样的 seq_len 帧」做模型内流式（显存优化）

在 `config/config.py` 中设置：

```python
"train_stream_clip_frames": True,
```

`train.py` 调用模型时会传入：

```python
stream_clip_frames=configs.get("train_stream_clip_frames", False)
```

默认为 `False`，与改前行为完全一致。

此时 Dataset **仍只取 `seq_len` 帧**（例如 5），与改前相同；只是模型里按这 5 帧顺序逐帧算 2D+BEV。

### 2.2 对「滑动窗口 rec 内全部帧」流式（不等长 clip：300 / 600 等）

目标：不是只采 `seq_len` 帧，而是对 **当前 `rec` 里实际取到的所有帧**（时间维长度 = `len(rec)`）做模型内流式。`rec` 由 **滑动窗口** 切出：`rec = clip[sce_id : min(sce_id + total_len, len(clip))]`，因此 **同一 `total_len` 可适配不同总长度的 clip**（300、600 等），只要 clip 内帧数 ≥ 窗口策略要求。

1. 在 `config/config.py` 中同时设置：

```python
"total_len": 128,   # 建议：单窗最大帧数（chunk），不必等于整段 clip；长 clip 靠多次滑动覆盖
"train_full_window_temporal": True,
"train_stream_clip_frames": True,
# 若仍嫌单次样本 T 太大、DataLoader/内存阻塞，可限制单样本最大帧数：
"stream_max_frames": 64,   # 或 128；None 表示不截断，用满当前 rec
```

2. **索引**：`train_full_window_temporal=True` 时由 `build_streaming_temporal_indexs(rec_len)` 生成 `0 … min(rec_len, stream_max_frames)-1`，**不再写死 `range(total_len)`**，避免 `len(rec) < total_len` 时越界。

3. `Model._forward_stream_clip_frames` 对 `t = x.shape[2]` 循环（即当前 batch 的实际 T），帧间 BEV 仍用 `fuse_with_prev`。

4. **长 clip（如 600 帧）**：不要把 `total_len` 拉到 600 再一次性训练；应使用 **较小的 `total_len`（或 `stream_max_frames`）** + **滑动窗口**（`__len__` 随 `len(clip)` 与 `total_len` 变化），多次迭代覆盖全段，避免「seq_len/total_len 过大导致阻塞」。

5. **batch_size**：建议为 **1**，并关注 **CPU 内存**（一次 collate 帧数 = `T × 相机数`）。

6. **map3D 等**：若标注逻辑仍假设短序列，全窗口下需自行核对；det3D 按帧 list 一般可随 `T` 变长。

7. **`L == total_len` 的 clip**：对 `L == total_len` 且 `L - total_len == 0` 时仍保留 **1 条窗口** 的补丁，避免无样本。

### 2.3 历史段内滑动 seq_len（避免整窗超长流式）

若希望 **`total_len=71`（前 20 + 当前 + 后 50）** 的语义不变，但网络输入仍为 **`seq_len=5`** 的常规 BEV 时序融合（不要整段 `T=71` 的流式或全窗 collate）：

```python
"train_history_sliding_chunks": True,
"history_sliding_num_frames": 20,  # 仅在窗口的前 H 帧上滑动
"seq_len": 5,
"train_full_window_temporal": False,
"train_stream_clip_frames": False,
```

Dataset 在每个 `total_len` 滑动窗内取 `rec[0:H]`，对 `k = 0 .. H-seq_len` 生成 `indexs = [k, k+1, ..., k+seq_len-1]`，**`__len__` 乘以 `K = H - seq_len + 1`**（如 16）。自车 e2e 真值仍锚在窗口内 **`current_frame_index`**（如 20）对应的那一帧，与 5 帧图像子窗分离。

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

## 6. 数据增强：如何保证同一 clip 内策略一致

### 6.1 当前工程里的做法（与是否流式无关）

流式前向 **只改变 `Model` 里张量怎么送网络**，**不在模型里做随机增强**。几何/外观增强发生在 **`dataset/dataset.py` 的 `get_image_data`**：对 `recs` 里每一帧循环时，仅在 **时间下标 `i == 0`**（该 clip 的第一帧）对每个相机调用 `gen_aug_params` 得到 `resize / crop / rotate / flip`，写入字典 `cnis[cn]`；**后续帧 `i > 0` 直接复用同一组 `cnis[cn]`**，再对各自图像做 `img_transform`。注释写明：「同一 getitem 中的数据裁剪和旋转要相同」。

因此：**同一 `__getitem__` 加载的一条 clip 内，各帧、各相机共用同一套增强参数**；`train_stream_clip_frames=True` 时只是按时间维从已增强好的 `x` 里切片，**不会**破坏 clip 内增强一致性。

### 6.2 若你以后改成「按帧单独取数」或「在模型里做增强」

需要自行保证 clip 级一致，常见做法：

1. **clip 级采样一次**：在 `__getitem__` 或 `collate` 里为每个 clip 生成一份 `aug_params`，对该 clip 所有帧复用（与现逻辑同思路）。
2. **确定性随机**：用 `seed = hash(clip_id, epoch)` 或 `torch.Generator` 在 clip 边界手动 `manual_seed`，再采样增强参数。
3. **避免**：对每一帧独立调用 `np.random` / `ColorJitter` 且不共享参数，会导致同 clip 内几何或颜色不一致，与时序/BEV 对齐假设冲突。

## 7. 显存与迭代语义

- **显存**：2D backbone 与 BEV 的单次前向规模从「整段 T」降为「单帧」，峰值通常明显下降；det3d/map3d 仍接收完整 `B*M*T` 的 BEV 张量，其显存与原先同量级。
- **优化步数**：每个 DataLoader batch 仍对应 **一次** `optimizer.step()`（与改前相同）；若需要按「帧」计 iteration / 学习率，需另行改 `train.py` 与 scheduler。

## 8. 限制与后续可做

- **DataLoader 未改**：每个 step 仍加载完整 clip；若需「每个 step 只加载一帧」以进一步省内存或 I/O，需改 `dataset`/`collate`。
- **推理脚本**：若单独调用 `Model.forward`，需自行传入 `stream_clip_frames=True` 才会走流式路径（训练默认由 config 控制）。

## 9. 相关文档

- BEV 与对齐矩阵：`doc/BEV_BACKBONE_AND_DET3D_LOSS_NOTES.md`、`doc/BEV_BACKBONE_AND_DYNAMIC_TRAINING_NOTES.md`
