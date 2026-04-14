# MapTR 实现全过程与官方差异详解（面向自定义数据集复现）

本文档面向当前仓库 `projects/e2e_models`，系统说明：

1. 本仓库里 MapTR 分支是如何从输入走到损失与解码的；
2. 与官方 MapTR（`hustvl/MapTR`）实现的关键差异；
3. 在自定义数据集上复现时，应优先对齐哪些点。

---

## 1. 先说结论（便于快速建立全局认知）

- 当前仓库的 MapTR 不是“官方整套端到端 MapTR 原样搬运”，而是 **“已有 BEV 主干 + MapTR 风格 map head”** 的混合架构。
- 也就是说，**图像到 BEV** 这段主要由本仓库 `BEVBackbone` 完成；`maptr/` 目录主要负责 **query 解码、匹配、loss、decode**。
- 和官方最核心的差异不是 head，而是 **BEV 生成与时序组织方式**（官方使用 MapTRPerceptionTransformer/BEVFormer 编码器，本仓库使用自定义 `BEVBackbone` + `pre_bev` 对齐融合）。

---

## 2. 代码结构与职责映射

### 2.1 本仓库中与 MapTR 直接相关的核心文件

- `maptr/map_head.py`：MapHead 主体（query、decoder、分类/回归、loss、decode）。
- `maptr/decoder.py`：MapTR Decoder 层（self-attn + deformable cross-attn + ffn）。
- `maptr/deformable_attn.py`：纯 PyTorch 的 `CustomMSDeformableAttention`。
- `maptr/assigner_loss.py`：归一化、匈牙利匹配、direction loss 等。
- `maptr/bbox_coder.py`：`MapTRNMSFreeCoder` 解码与后处理（topk、阈值、范围过滤）。
- `maptr/build_model.py`：从 `grid_conf` 构建 `pc_range/post_center_range/bev_h/bev_w`，并实例化 `MapHead`。

### 2.2 与 MapTR 串联的“上游/外围”文件

- `model/models.py`：总模型，把 `BEVBackbone` 输出送进 `map3d_head`，并汇总 `loss_map_*`。
- `backbone/bev_backbone.py`：多相机特征投影到 BEV，历史 `pre_bev` 按 ego pose 对齐融合。
- `train.py`：时序训练循环，维护 `pre_bev`，逐帧调用 `model(...)`。
- `dataset/dataset.py`：构建 `gt_labels_map3D / gt_bboxes_map3D / gt_pts_map3D`。
- `config/config.py`：`map_3d_head` 超参数与任务开关。

---

## 3. 本仓库 MapTR 全流程（训练视角）

下述流程等价于你训练时真正发生的执行链路。

### 3.1 数据准备与采样（`dataset/dataset.py`）

1. `Dataset.__getitem__` 先取一个 clip 的滑窗片段；
2. `get_image_data` 输出图像与相机参数：
   - `x, rots, trans, intrins, distorts, post_rots, post_trans`
3. `get_ego_pose` 输出 `ego_poses`（时序对齐使用）；
4. `get_anno_map3D` 输出地图 GT：
   - `gt_labels_map3D`
   - `gt_bboxes_map3D`
   - `gt_pts_map3D`
5. `custom_collate` 把变长地图 GT 保持为 list，供后续 Hungarian 匹配。

### 3.2 训练循环与时序状态维护（`train.py`）

1. 外层按 mini-batch；
2. 内层按时间步 `ii` 逐帧 forward；
3. 每帧把 `outputs["cur_bev"]` append 到 `pre_bev`；
4. `pre_bev` 只保留最近 `seq_len-1` 帧。

这就是“在线建图”在本仓库的时序来源：不是 map head 自带状态，而是 **BEV 历史特征缓存**。

### 3.3 模型前向与 MapTR 分支接入（`model/models.py`）

1. 图像 backbone 提取特征；
2. 进入 `BEVBackbone` 得到：
   - `cur_bev`：当前帧 BEV
   - `fusion_bev`：与历史对齐融合后的 BEV
3. static 任务调用 `forward_static_branch`：
   - `map_pred = self.map3d_head(bev_feat, metas)`
   - `map_losses = self.map3d_head.loss(...)`
   - 可选 `map_polylines = self.map3d_head.decode(...)`

### 3.4 MapHead 细节（`maptr/map_head.py`）

#### 3.4.1 输入输出定义

- 输入：`bev_feature`，形状 `(B, C, H, W)`（在当前工程里经常是 `B*T` 展平后）。
- 输出字典（多层）：
  - `map_cls_scores`：每层 `(B, num_vec, num_classes)`
  - `map_bbox_preds`：每层 `(B, num_vec, 4)`（归一化 `cx,cy,w,h`）
  - `map_pts_preds`：每层 `(B, num_vec, num_pts, 2)`（归一化）

#### 3.4.2 Query 构造

- 默认 `use_instance_pts=True`：
  - `instance_embedding(num_vec, 2*D)` + `pts_embedding(num_pts, 2*D)`；
  - 相加后展平为 `num_vec * num_pts` 个 query；
  - split 为 `query` 与 `query_pos`。

#### 3.4.3 BEV memory 构造

- `bev_proj`（1x1 conv）统一通道；
- `LearnedPositionalEncoding2D` 加到 BEV；
- flatten 成 `memory`（`B, H*W, D`）。

#### 3.4.4 Decoder 两种模式

- `use_maptr_decoder=True` 时：
  - 走 `MapTRDecoder`（`maptr/decoder.py`）
  - 每层结构是 `self_attn -> norm -> deformable_cross_attn -> norm -> ffn -> norm`
- 否则回退到 `nn.TransformerDecoder`（标准注意力版本）。

#### 3.4.5 回归与分类

- 每层 decoder 输出经：
  - `reg_branches` 预测参考点增量（或直接预测）
  - `transform_box` 将点集合转为 bbox（min/max）
  - `cls_branches` 对每条 vector 的点特征均值后做分类

### 3.5 匹配与损失（`maptr/assigner_loss.py` + `maptr/map_head.py`）

#### 3.5.1 Hungarian assigner

- 代价项：
  - `cls_cost`（FocalLossCost 形式）
  - `reg_cost`（bbox L1，归一化后）
  - `pts_cost`（Ordered Pts L1，支持多顺序 GT）
- `linear_sum_assignment` 做一对一匹配。

#### 3.5.2 多顺序 GT

- `gt_pts` 支持 `(N_gt, num_orders, num_pts, 2)`；
- assign 时在 `num_orders` 上取最小 cost，得到 `order_index`；
- 训练时按 `order_index` 选对应序列监督。

#### 3.5.3 loss 组成

- `loss_map_cls`：sigmoid focal（实现中乘以 2.0）；
- `loss_map_bbox`：L1（归一化 bbox）；
- `loss_map_pts`：L1（归一化点，默认放大 5.0）；
- `loss_map_dir`：方向余弦损失（在反归一化坐标上计算）。

并且每个 decoder 层有 aux loss（最后层权重 1.0，其他层 `aux_loss_weight`）。

### 3.6 推理解码（`maptr/map_head.py` + `maptr/bbox_coder.py`）

1. 取最后一层 `cls/bbox/pts`；
2. `MapTRNMSFreeCoder.decode`：
   - `sigmoid` 分类分数
   - top-k 选 query
   - 反归一化到物理坐标
   - `score_threshold` + `post_center_range` 过滤
3. 返回每样本 `bboxes/scores/labels/pts`。

---

## 4. 与官方 MapTR 的逐项差异（重点）

对比基准：

- 官方配置：`projects/configs/maptr/maptr_nano_r18_110e.py`
- 官方 head：`projects/mmdet3d_plugin/maptr/dense_heads/maptr_head.py`
- 官方 assigner：`projects/mmdet3d_plugin/maptr/assigners/maptr_assigner.py`

### 4.1 架构级差异（最关键）

1. **BEV 生成路径不同**
   - 官方：`MapTRPerceptionTransformer` 内含 BEVFormer encoder（可用 prev_bev、can_bus、shift/rotate）。
   - 本仓库：`BEVBackbone` 自行完成多相机投影 + ego pose 对齐 + `algin_fusion`。

2. **时序机制位置不同**
   - 官方：时序更多嵌在 transformer/prev_bev 机制中。
   - 本仓库：时序显式在 `train.py` 的 `pre_bev` 缓存与 `BEVBackbone.egopose2thetamat+warp_feature`。

### 4.2 Head 与损失层差异

1. **`num_vec` 默认值**
   - 官方 nano 配置常用 `num_vec=100`。
   - 本仓库当前配置默认 `num_vec=50`（`config/config.py`）。

2. **bbox/iou 权重设定倾向不同**
   - 官方 nano：`loss_bbox=0.0`, `loss_iou=0.0`，重点在 pts 和 cls。
   - 本仓库：实现了 bbox L1，且实际训练会参与（受配置权重控制）。

3. **pts loss 形式**
   - 官方 nano：`PtsL1Loss`（有序点 L1）。
   - 本仓库当前：`loss_map_pts` 使用点级 L1（与 OrderedPtsL1 目标对齐），并在 assigner 侧用 ordered L1 cost。
   - 早期文档中提到的 Chamfer 方案，在当前代码中不是主路径。

4. **分类损失**
   - 官方：FocalLoss（`gamma=2`, `alpha=0.25`, `weight=2.0`）。
   - 本仓库：实现同类 focal 形式，末尾乘 2.0 与官方权重保持一致趋势。

5. **Decoder 实现来源**
   - 官方：mmcv/mmdet 体系下模块化实现。
   - 本仓库：纯 PyTorch重写（`MapTRDecoder` + `CustomMSDeformableAttention`），接口对齐但工程依赖更轻。

### 4.3 工程化差异

1. **训练框架**
   - 官方：MMDetection3D Runner/Pipeline 体系。
   - 本仓库：原生 PyTorch 训练循环（`train.py`），手工组装 `metas`。

2. **数据集接口**
   - 官方：NuScenes LocalMap 数据集类，`gt_bboxes + gt_labels + gt_shifts_pts`。
   - 本仓库：`dataset.py` 直接输出 `gt_*_map3D` 列表结构，命名与 `Model.forward_static_branch` 对齐。

3. **多任务耦合**
   - 官方 MapTR 项目主要关注 map 分支（另有检测配置）。
   - 本仓库：map 与 det3d、traj 等任务共用同一 BEV 特征、同一次前向。

---

## 5. 你在自定义数据集复现 MapTR 时，必须对齐的 10 个要点

### 5.1 标签定义与坐标

1. 明确 map 类别集合（例如 divider/ped_crossing/boundary）；
2. 保证所有 GT 点在统一车体坐标系（ego x-y）；
3. 定义固定 `pc_range`（训练/解码都依赖它）；
4. 每条线固定采样到 `num_pts_per_gt_vec` 点；
5. 为闭合/有方向歧义线段构造多顺序 `gt_pts`（`num_orders`）。

### 5.2 数据结构与加载

6. 输出结构建议直接对齐本仓库：
   - `gt_labels_map3D`: list[Tensor(N_gt)]
   - `gt_bboxes_map3D`: list[Tensor(N_gt,4)]
   - `gt_pts_map3D`: list[Tensor(N_gt,num_orders,num_pts,2)]
7. `collate_fn` 不要把变长 map GT 强行 stack，保持 list。

### 5.3 模型与训练超参

8. 初始复现优先对齐：
   - `num_vec=100`
   - `num_pts_per_vec=20`
   - `loss_cls(focal)=2.0`
   - `loss_pts=5.0`
   - `loss_dir=0.005`
9. 如果你先在本仓库复现，不必先纠结官方 encoder；先保证 head 与标签可收敛，再迭代 BEV 主干。

### 5.4 验证与诊断

10. 至少做三类 sanity check：
   - 可视化 GT 折线与 `pc_range` 对齐；
   - 检查 assigner 正样本数（不是长期 0）；
   - 验证 decode 后 `pts` 是否在合理物理范围内。

---

## 6. 推荐复现策略（实操顺序）

1. **第一阶段（最小闭环）**
   - 固定单任务 map3d；
   - 只跑短序列/小数据；
   - 检查 loss 下降 + 可视化输出。

2. **第二阶段（对齐官方参数）**
   - 把 `num_vec/num_pts/loss 权重` 拉到官方配置；
   - 比较 mAP/chamfer 曲线形态。

3. **第三阶段（增强时序与主干）**
   - 再考虑更换/增强 BEV encoder；
   - 对比 `pre_bev` 融合和官方 prev_bev 机制差异收益。

---

## 7. 一句话总结

你现在这套代码可以看作：

**“官方 MapTR 的 map head 思路 + 本仓库自定义 BEV 时序主干”**。

如果目标是“在你自己的数据集上稳定复现 MapTR 结果”，优先把 **标签格式、assigner 输入、loss 权重、坐标归一化** 四件事对齐，再优化 backbone 与时序策略，成功率最高。

