# 动态目标轨迹 VQ-VAE 与 Index Transformer 预测管线

本文档说明在 PoseGPT 参考代码目录下为 **Sparse4D 多目标未来轨迹** 新增的离线 VQ-VAE、索引预测 Transformer，以及二者组合的 **TrajectoryVQPredictionPipeline**。量化层复用 `models/quantize.py` 中的 `VectorQuantizer`（与 PoseGPT / taming-transformers 一致）。

---

## 1. 目标与总体方案

**阶段 A：离线训练 VQ-VAE**

- 将每个槽位上的 **未来轨迹**（例如 50 个时间步 × xy → 100 维）编码为连续瓶颈 `z`，经向量量化得到离散 **indices**，再经解码器重建轨迹。
- 使用 **instance_mask** 区分真实目标与「填到固定槽位数」的 padding；使用 **traj_mask**（与轨迹同形状）标记每一步/维是否参与重构损失。
- **默认槽位数 N = 100**，与 Sparse4D 单帧检测目标数一致（若在其它实验中改为 128，需保证全链路 `N` 一致）。

**阶段 B：训练轨迹索引预测器（不接 GPT）**

- 输入：**instance_feature** `(B, N, 256)`、**anchor** `(B, N, 11)`（Sparse4D 风格）。
- 使用 **三层（可配）Transformer Encoder** 在 N 个槽位上做自注意力，`instance_mask` 作为 `src_key_padding_mask`。
- 对每个槽位输出与 VQ-VAE **子码本** 对齐的 **logits**，用 **交叉熵** 监督；教师标签来自 **冻结的 VQ-VAE** 对 GT 轨迹前向得到的 **indices**。
- 推理：**argmax** 得到 `pred_indices` → `vqvae.decode_from_indices` → **未来轨迹** `(B, N, 100)`。

---

## 2. 涉及文件与改动摘要

| 路径 | 作用 |
|------|------|
| `models/quantize.py` | 既有 PoseGPT 向量量化（未改逻辑，本文档仅引用）。 |
| `models/trajectory_vqvae.py` | **TrajectoryVQVAE**：Encoder（1D Conv 时序）+ `VectorQuantizer` + Decoder；`decode_from_indices`。**默认 `num_slots=100`**。 |
| `models/trajectory_index_transformer.py` | **TrajectoryIndexTransformer** + **trajectory_vq_index_ce_loss**。**默认 `max_slots=100`**。 |
| `models/trajectory_vq_prediction_pipeline.py` | **TrajectoryVQPredictionPipeline**：一次 forward 中 CE loss + `pred_indices` + decoder 解压；**build_trajectory_vq_pipeline_100**；`__main__` demo。 |

---

## 3. 张量约定

| 名称 | 形状 | 含义 |
|------|------|------|
| `x` / `gt_traj` | `(B, N, 100)` | N 个槽位；100 = 50 步 × 2(xy)。 |
| `traj_mask` | `(B, N, 100)` | 轨迹内有效位置为 1，无效为 0。 |
| `instance_mask` | `(B, N)` | 真实目标为 1，为凑满 N 的 padding 槽为 0。 |
| VQ 瓶颈 `z` / `z_q` | `(B, N, e_dim)` | 与 `VectorQuantizer` 的 `e_dim` 一致。 |
| `indices` / `pred_indices` | `(B, N, nbooks)` | 每本书一个整数索引；padding 槽在训练标签中可为 **-1**（不参与 CE）。 |
| `logits` | `(B, N, nbooks, n_e_i)` | `n_e_i = n_e // nbooks`，与子码本大小一致。 |

**Sparse4D 对齐**：`N=100` 时，`instance_feature` 与 `anchor` 的第二维均为 100，与已训练 VQ-VAE 的 `num_slots` 必须相同。

---

## 4. TrajectoryVQVAE（离线）

- **编码**：输入掩码后的轨迹，`Conv1d` 在时间长度 50 上提取特征，GAP 后线性到 `e_dim`。
- **量化**：`VectorQuantizer(n_e, e_dim, beta, nbooks)`，可多码本；**straight-through** 与 VQ 损失在模块内部完成。
- **掩码**：
  - `instance_mask=0` 的槽位：`z` 置零后再量化；VQ 与 indices 对该类槽位按实现置 `-1` / 乘掩码。
  - 重构损失：`instance_mask` 与 `traj_mask` 联合加权 MSE。
- **解码**：`decode_from_indices(indices, instance_mask)`：`get_codebook_entry` → `z_q` → MLP Decoder → `(B, N, 100)`；**indices 中 -1 会先 clamp 再查表，随后用 instance_mask 将 padding 的 `z_q` 置零**（避免 `Embedding(-1)` 报错）。

训练时总损失可写为：`recon_loss + vq_loss`（可对 `vq_loss` 再乘系数）。

---

## 5. TrajectoryIndexTransformer

- 将 `[instance_feature; anchor]` 线性投影到 `d_model`，可选 **槽位嵌入**（`nn.Embedding(max_slots, d_model)`，需 `N <= max_slots`）。
- **自定义 Encoder**：堆叠多层 `TrajectoryEncoderBlock`（Pre-LN），自注意力为 `blocks/attention.py` 中的 **`CausalSelfAttention`** 且 **`causal=False`**（全连接注意力）；`instance_mask` 以 **`valid_mask`** 传入，padding 槽不作为 key 被 attend（与原先 `nn.TransformerEncoder` 的 key padding 语义一致）。
- **nbooks 个独立 `Linear(d_model, n_e_i)`** 输出每本码本的分类 logits。
- **trajectory_vq_index_ce_loss**：对每个子码本在有效位置（`instance_mask` 且 `target >= 0`）上求 CE，再对子码本取平均。

---

## 6. TrajectoryVQPredictionPipeline（一次 forward）

**输入**

- `instance_feature`, `anchor`, `instance_mask`（必填）。
- 训练时任选其一：
  - 提供 **`gt_traj` + `traj_mask`**：在 `inference_mode` 下用冻结 VQ-VAE 前向得到 **teacher_indices**；
  - 或直接提供 **`teacher_indices`**。

**输出字典**

- `loss`：CE（无监督信号时为 `None`）。
- `logits`：`(B, N, nbooks, n_e_i)`。
- `pred_indices`：argmax 结果，padding 槽为 -1。
- `teacher_indices`：若本步计算或传入教师，则出现在此。
- `decoded_traj`：`decode_from_indices(pred_indices, instance_mask)`，**VQ-VAE 冻结时 decode 在 `no_grad` 中执行**（argmax 不可微，主监督为 CE）。

**工厂函数**

- `build_trajectory_vq_pipeline_100(...)`：构造默认 **100 槽** 的 VQ-VAE 与 Index Transformer；实际使用时应用 **`load_state_dict`** 加载已训练 VQ-VAE 权重，并保证 **`n_e`、`nbooks`、`e_dim`、`traj_dim`** 与检查点一致。

---

## 7. 训练与推理流程（ checklist）

1. 用 GT 轨迹训练 **TrajectoryVQVAE** 至收敛，保存权重。
2. 构建 **TrajectoryIndexTransformer**，`n_e`、`nbooks` 与 VQ-VAE **完全一致**；`max_slots >= N`（默认 100）。
3. 加载 VQ-VAE，设为 **eval** 且 **requires_grad=False**，封装进 **TrajectoryVQPredictionPipeline**。
4. 每个 batch：`pipeline(..., gt_traj, traj_mask, instance_mask)`，对 **`out['loss']`** 反传，仅更新 index transformer。
5. 推理：`pipeline(..., decode_trajectory=True)`，取 **`out['decoded_traj']`** 作为预测未来轨迹；需要指标时可仅在 **instance_mask==1** 的槽位上算误差。

---

## 8. 运行自带 Demo

在 `reference_code/PoseGPT` 下执行（需 **CUDA**：当前 `quantize.py` 内 **EmaCodebookMeter** 使用 `.cuda()`，CPU 环境需自行改为按 tensor device 创建）：

```bash
python -m models.trajectory_vq_prediction_pipeline
```

预期：打印 CE loss、`decoded_traj` 形状 `(B, 100, 100)`、`pred_indices` 形状 `(B, 100, nbooks)`。

---

## 9. 与后续模块的衔接

若之后接入 **自回归 GPT** 修正 indices：在离散空间建模即可；解压阶段仍调用同一 **`decode_from_indices`**，保持码本与 `nbooks` 定义不变。

---

## 10. 版本说明

- 文档与实现针对 **`projects/e2e_models/reference_code/PoseGPT`** 下的相对路径；主工程引用时请把 `PoseGPT` 加入 `PYTHONPATH` 或按包名安装。
- 若将槽位从 100 改为其它 **N**，请同时修改 **VQ-VAE `num_slots`、Index Transformer `max_slots`、数据与检测头输出维度**。
