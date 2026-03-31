# BEV Backbone / `algin_features` NaN 排查手册

本文档汇总 `backbone/bev_backbone.py` 及相关数据路径上**可能导致 NaN / Inf** 的位置，便于按顺序逐项排除。建议配合临时断言：`torch.isfinite(t).all()`，或 `torch.isnan(t).any()`。

---

## 1. 先区分现象

| 现象 | 更可能的方向 |
|------|----------------|
| **整幅特征 `(B,C,H,W)` 几乎全是 NaN** | 上游 `x` 已坏、`theta_mats`/`warp_feature` 全图网格坏、`cam_pos_embed` 分支整图坏、**模块权重已含 NaN** |
| **稀疏 NaN 或局部异常、其余为 0/正常** | BEV→图像投影深度/无效点、grid 大量越界（`padding_mode="zeros"` 多为 0，不一定 NaN）、鱼眼/开方局部数值问题 |

---

## 2. 推荐排查顺序（由前到后）

按下面序号在训练或单次 `forward` 中检查；**某一步已非有限，则优先修该步及更上游**。

### 步骤 A：模型参数是否已污染

- **查什么**：`BEVBackbone` 及上游 2D backbone 的 `weight`/`bias` 是否含 NaN/Inf。
- **怎么查**：训练出现异常后执行  
  `any(not torch.isfinite(p).all() for p in model.parameters())`
- **若 True**：属于优化侧问题（学习率、AMP、loss 爆炸、梯度裁剪等），需结合 `autograd` anomaly 或关 AMP 复现。

---

### 步骤 B：`forward` 入口图像特征 `x`

- **文件/位置**：`BEVBackbone.forward` 中第一次使用 `x` 之前。
- **原因**：`x` 若已为 NaN，后续所有 `grid_sample`、`Conv`、`BN` 均可整幅传播。
- **怎么查**：`torch.isfinite(x).all()`。
- **若 False**：查 2D backbone、前置归一化、数据加载（图像是否异常）、AMP。

---

### 步骤 C：相机位置编码（仅 `use_cam_pos_embed=True`）

- **文件/位置**：`get_cam_pos_embedding`（`bev_backbone.py`）；`forward` 内 `torch.cat([x, cam_emb], dim=1)` 之后对拼接结果检查。
- **原因**：`torch.inverse(post_R)`、`torch.inverse(intrins)` 在矩阵奇异或病态时产生 Inf，再经 `matmul`、除法、`normalize` 可能变为整幅 NaN，并通过 concat **污染整条 `x`**。
- **怎么查**：
  1. `torch.isfinite(cam_emb).all()`（在 concat 前）；
  2. 临时将配置中 `use_cam_pos_embed=False`，若 NaN 消失则重点查标定与 `inverse`。
- **相关辅助函数**：`_pixels_to_cam_dirs_fisheye` 内含除法与迭代，一般较少单独导致「整幅」NaN，除非输入已为 Inf。

---

### 步骤 D：BEV→相机投影与 `grid`（`Splat.bev2eachroi`）

- **文件/位置**：`bev_backbone.py` 中 `Splat.bev2eachroi`、`projectPoints_fisheye`、`normalize_coords`。
- **子项 D1 — 深度 / 透视除法**  
  - **原因**：`z ≤ 0` 或 `z → 0` 时 `x/z、y/z` 为 Inf；虽已对 `z <= eps` 等做屏外处理，若实现不一致或未覆盖所有分支仍可能出问题。  
  - **怎么查**：透视除法后对 `x_norm, y_norm` 或等价张量 `torch.isfinite(...).all()`；抽查 `z_cam` 分布。
- **子项 D2 — 标定张量**  
  - **原因**：`intrins`、`distorts`、`post_rot3`、`post_tran3`、`rots`、`trans` 任一带 NaN，经 `matmul` 可扩散。  
  - **怎么查**：在 `bev2eachroi` 入口对上述张量 `isfinite`。
- **子项 D3 — 鱼眼投影**  
  - **原因**：`sqrt`、`radial` 中 `r + 1e-6` 等通常较稳；若 `proj_points` 已为 Inf，会整段异常。  
  - **怎么查**：`projectPoints_fisheye` 返回前对输出 `isfinite`。
- **子项 D4 — `normalize_coords`**  
  - **原因**：若特征高宽退化为 1，`shape - 1` 为 0 会导致除零（正常 32×96 不应出现）。  
  - **怎么查**：确认 `feat_h, feat_w` 与真实 `x.shape[-2:]` 一致；对 `norm_coords` 做 `isfinite`。

---

### 步骤 E：第一次 `grid_sample`（多高度 BEV→图像）

- **文件/位置**：`forward` 中 `F.grid_sample(input=x, grid=grid[index, ...], ...)`。
- **原因**：`grid` 含 NaN/Inf 或 `x` 已坏，输出非有限。
- **怎么查**：`grid` 与每个 `outs` 元素在 `cat` 前 `isfinite`。

---

### 步骤 F：`fusion_height` → `multiview_fusion` 后的 `out_put`

- **文件/位置**：`out_put = self.multiview_fusion(...)` 之后、`view(-1, S, ...)` 之后。
- **原因**：若此处已全 NaN，问题在 E 及以前；若此处仍正常，问题更可能在时序对齐。
- **怎么查**：`torch.isfinite(out_put).all()`；可再对 `out_put[:, 0]` 单独查（第一帧是否已坏）。

---

### 步骤 G：时序 `warp_feature` 与 `theta_mats`

- **文件/位置**：`warp_feature`、`algin_fusion` 循环；`theta_mats` 由 dataloader / `dataset.get_algin_theta_mat` 等提供。
- **原因**：`theta` 含 NaN/Inf 时，`affine_grid` 整张采样网格异常，`grid_sample` 常导致**整幅** warped 特征为 NaN；从 `i >= 1` 起 `algin_features` 会连锁变坏。
- **怎么查**：
  1. `torch.isfinite(theta_mats).all()`；
  2. 在 `warp_feature` 内对 `grids`、`cropped_features` 做 `isfinite`；
  3. 若仅 `i>=1` 开始出现 NaN，优先怀疑本步与 `theta` 维度和 batch 对齐（`rearrange` 是否与 `out_put` 的 B、T 一致）。

---

### 步骤 H：`algin_fusion`（2×C → C）

- **文件/位置**：`self.algin_fusion(torch.cat([warped_prev, out_put[:, i]], 1))`。
- **原因**：输入已 NaN 则输出必 NaN；BN 在参数已坏时同样传播。
- **怎么查**：对 `cat` 后的张量及 `algin_fusion` 输出 `isfinite`。

---

## 3. 训练侧辅助手段（可选）

- **`torch.autograd.set_detect_anomaly(True)`**：定位首次产生 NaN 的反向算子（较慢）。
- **关闭 AMP（`autocast`）**：区分混合精度溢出与纯前向几何/数据问题。
- **减小 batch / 单卡单样本**：缩小变量，便于对照步骤 A～H。

---

## 4. 关键文件索引

| 内容 | 路径 |
|------|------|
| BEV 前向、采样、对齐 | `backbone/bev_backbone.py` |
| `theta_mats` 生成（示例） | `dataset/dataset.py` → `get_algin_theta_mat` |
| 模型里 `theta` 维度的 `rearrange` | `model/models.py`（以仓库当前实现为准） |

---

## 5. 文档维护说明

若你修改了 `bev2eachroi` 的深度阈值、`normalize_coords` 的 `shape` 定义、或 `warp_feature` 的语义，请同步更新本文档对应小节，避免排查顺序与代码脱节。
