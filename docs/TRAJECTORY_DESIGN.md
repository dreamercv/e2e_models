## 端到端轨迹整体设计思路（VQ-VAE + GPT + Diffusion）

本文档整理当前工程里的整体设计与未来规划，方便后续阅读与实现。

---

### 1. 整体框架与分阶段训练

数据与模型大致流程（参考 `dataset.py` 9–36 行）：

1. **多相机时序输入**
   - 输入：`2 * bs * 5 * 8 * 3 * 128 * 384`
     - 5：时序 5 帧
     - 8：环视 8 个相机
     - 2：两种数据源（动态 / 静态）

2. **2D image_backbone**
   - 自定义纯 2D CNN，下采样 4 倍，输出通道数 256。
   - 输出尺寸：`2 * bs * 5 * 8 * 256 * 32 * 96`。

3. **BEVBackbone**
   - 利用相机内外参，将像素特征投影到 BEV 空间：
     - 自车坐标系下初始化 \(200m \times 80m \times 6m\) 的 3D 网格；
     - 通过内外参映射到像素坐标，再回填到 BEV。
   - 输出尺寸：`2 * bs * 5 * 256 * 200 * 80`。

4. **时序对齐模块 algin_module**
   - 用自车相对位姿，对 5 帧 BEV 特征逐帧 warp / 融合：
     - 第一帧：F1
     - 第二帧：align(F1) 与 F2 拼接，卷积得到 aligned F2
     - ...
     - 第五帧：最终得到 `algin_F5`
   - 将 F1、algin_F2、…、algin_F5 拼成 `algin_F1_5`：
     - 输出尺寸：`2 * bs * 5 * 256 * 200 * 80`
     - 表示“与当前帧对齐的历史 5 帧 BEV 特征”
   - 之后通过 split_module 分成：
     - `dynamic_input = bs * 5 * 256 * 200 * 80`
     - `static_input = bs * 5 * 256 * 200 * 80`

5. **世界模型分支**
   - 使用自车速度 / 位姿 + 历史 BEV 特征重建当前 BEV：
     - 输入：历史 BEV + ego motion
     - 输出：预测当前 BEV
   - 与真实当前 BEV 做重建损失约束，使 BEV feature 具备“感知 + 预测 / 重建”的能力。

6. **动态分支（Sparse4D + 目标轨迹）**
   - 将 `dynamic_input` 输入 Sparse4DHead：
     - 输出 instance：`(bs, 5 * 900, 256)`（最多 900 个目标）
     - 输出 anchor：`(bs * 5 * 900, 11)`，11 维包含 `[x,y,z,w,l,h,vx,vy,vz,yaw,id]`
   - 在 Sparse4D 输出的基础上，加 **目标轨迹预测头 obj_traj_head**：
     - 输入：
       - 时序 instance 特征（`seq_features`）
       - 时序 anchor（`seq_anchors`）
       - 历史目标轨迹（`bs * 900 * 10`）等
     - 输出：
       - 当前帧周围目标未来 5 秒轨迹：`bs * 900 * 100`（100 = 50 × 2，未来 50 帧的 xy）
       - 对应目标级 instance 特征：`bs * 900 * 256`

7. **静态分支（MapTR + 拓扑轨迹）**
   - 将 `static_input` 输入 MapTR：
     - 输出 map instance：`bs * 5 * 200 * 256`
     - 输出 map anchor：`bs * 5 * 200 * 3`（最多 200 条线/多边形，3 维为 x, y, 类别）
   - 在 MapTR instance 基础上，加 **自车拓扑轨迹头 ego_static_head**：
     - 输入：map instance + 最后一帧 BEV 特征
     - 输出：
       - 自车为起点到 BEV 边缘的拓扑轨迹：`bs * 1 * 20 * 2`（20 个等步长点）
       - 对应 instance 特征：`bs * 1 * 256`

8. **端到端分支 e2ehead**
   - 在上述分支训练收敛后开启：
   - 将：
     - BEV 特征（`bs * 256 * 200 * 80`）
     - det instance（`bs * 900 * 256`）
     - 动态目标轨迹 instance（`bs * 900 * 256`）
     - map instance（`bs * 200 * 256`）
     - 自车拓扑轨迹 instance（`bs * 1 * 256`，最后一帧）
   - 一起输入到 e2ehead：
     - 输出：自车未来 5 秒端到端轨迹（`bs * 100`，通常 reshape 为 50 × 2 或 50 × 3）
     - 这个轨迹要融合：
       - BEV 的场景描述（静态+动态）
       - 检测/轨迹 instance（动态场景理解）
       - 地图 + 拓扑轨迹（静态结构理解）

整体训练策略：**分阶段训练 + 最后阶段端到端微调**，从 2D backbone → 世界模型 → Sparse4D det → MapTR → 各种轨迹头 → 最终 e2ehead，全链路保持可导。

---

### 2. 三种轨迹头的建模思路

#### 2.1 目标轨迹（obj_traj_head）——计划使用 VQ-VAE

目标轨迹的特点：

- 多目标、多模态，多种驾驶风格（加速、减速、变道、跟车等），形状多样。
- 对单个目标的轨迹精度容忍度相对较高（只要合理、不太离谱即可）。

**设计：**

- 使用 **VQ-VAE** 对目标轨迹进行离散化建模：
  - Encoder：将一条连续轨迹（例如 50 帧、2D 或 3D）映射到 latent 表示；
  - Quantizer：在离散 codebook 中寻找最近的 code，得到 code id；
  - Decoder：从 code（及条件上下文）重构轨迹。
- 好处：
  - 轨迹模式被压缩为一组 **离散“轨迹原子”/code**，更易学习和组合；
  - 支持多模态：同一环境下可采样不同 code 组合，得到不同轨迹；
  - 与主干 Sparse4D 兼容：可以把 VQ 的 code embedding 作为“目标未来意图”的特征输入给后续模块（例如 e2ehead 或 world model）。

**建议实现路径：**

1. 先在目标轨迹真值上 offline 训练一个 VQ-VAE：  
   - 只用轨迹本身训练 codebook + encoder/decoder。  
2. 在主网络中：
   - 用 VQ encoder 将 GT 或预测轨迹编码为 code id 或 embedding；
   - 将这些 embedding 作为 **“目标未来行为提示”**，供后续模块（如 e2e diffusion）使用。

---

#### 2.2 拓扑轨迹（ego_static_head）——计划使用自回归（GPT 风格）

拓扑轨迹的特点：

- 路径受道路拓扑强约束：必须在车道内、与 lane center 对齐、不能乱飞。
- 轨迹长度短（例如 10 或 20 个等步长点）。

**设计：**

- 使用 **自回归（GPT 风格）** 的小型序列模型：
  - 输入：上一时刻的拓扑轨迹点 + map instance + BEV 特征；
  - 逐步预测下一步的 (x, y)，一共预测 10/20 步；
  - 可以看成一个小型 autoregressive decoder。

**两种实现思路：**

1. **纯学习自回归**：
   - 完全由 GPT/GRU 从零生成 10 个点。
   - 每一步使用 teacher forcing 训练，推理时用自身预测递推。
   - 简单直接，但可能出现轻微漂移（离开 lane）。

2. **“图 + GPT refine”的混合方案（推荐）**：
   - 先通过 MapTR 得到向量化 map（lane centerline 等），建立 lane graph；
   - 使用简单图搜索（A* / Dijkstra）从 ego pose 出发找到一条初始 path；
   - 使用 GPT/GRU **在这条 path 上做细化/偏移预测**（而不是从零生成）：
     - 学习调节偏离量、速度变化等；
   - 优点：
     - 轨迹天然在拓扑图里，安全性更好；
     - 学习任务难度更小（网络主要做 refine）。

---

#### 2.3 自车端到端轨迹（e2ehead）——计划使用 Diffusion

端到端轨迹的目标：

- 在 **动态目标轨迹（VQ）** + **静态拓扑轨迹（GPT）** + **BEV 场景特征** 的条件下：
  - 生成自车未来 50 帧的 **多模态**、**安全（不碰撞）**、**遵守拓扑** 的轨迹；
  - 从图像到底层控制，都保持可导。

**设计：使用 Diffusion 在轨迹空间建模**

- 把自车未来轨迹视为高维向量（例如 50 × 2/3），在该空间上做扩散模型：
  - 前向：逐步加噪到近似高斯；
  - 反向：条件在 BEV + det/map instance + 目标 VQ code + topo GPT 输出，逐步去噪生成轨迹。
- Diffusion 的优势：
  - 能建模 \(p(\text{ego\_traj} | \text{scene}, \text{targets\_traj}, \text{topo\_traj})\) 的完整分布；
  - 能自然表达多模态（直行 vs 变道）和不确定性；
  - 可通过训练数据中“collision-free + 合规”轨迹，间接学到避障与顺拓扑行为。

**需要注意：**

- 训练与推理成本高：
  - 每次采样要若干步（例如 10–50 步）反向扩散；
  - 叠加在现有 BEV+Sparse4D+MapTR backbone 上，显存与时延都需要仔细评估。
- Loss 设计要清晰：
  - 明确是在做 score matching / denoising loss，而不是简单 L1 回归；
  - 确保训练数据中“好轨迹”的定义足够清楚（不碰撞、平滑、遵守拓扑）。
- 建议作为 **在已有回归式 e2e 头上的升级版本**：
  - 先实现单模态回归 e2ehead 作为 baseline；
  - 再在此基础上替换为 diffusion，以 baseline 为对照，逐步调参。

---

### 3. 三个轨迹头选择方案的整体可行性与工程推进顺序

从建模角度看：

- **目标轨迹（多模态 + 容错高） → VQ-VAE**：  
  - 非常契合“轨迹模式 / 原子动作”的离散表示需求；
  - codebook 为后续模块（如 e2e diffusion）提供“目标未来行为提示”。

- **拓扑轨迹（强拓扑约束 + 短序列） → GPT 自回归（或图+GPT refine）**：  
  - 符合轨迹本质上的序列结构；
  - 结合 MapTR 的 vectorized lane，可以在几何/拓扑上强约束轨迹合法性。

- **自车端到端轨迹（多模态 + 安全+顺拓扑） → Diffusion**：  
  - 在融合目标轨迹 code + 拓扑轨迹 + BEV 场景特征的条件下，建模完整的未来轨迹分布；
  - 长期看是表达能力最强的一种方式。

从工程推进角度看，建议的顺序是：

1. **第一阶段**：  
   - 用简单回归式头（MLP/GRU/Transformer）实现：
     - obj_traj_head（目标轨迹）
     - ego_static_head（拓扑轨迹）
     - e2ehead（端到端轨迹）  
   - 目标：先让 det3D / map3D / 三种轨迹的基本管线全部 **跑通 & 收敛**。

2. **第二阶段**：  
   - 在 **目标轨迹 head** 上引入 VQ-VAE（局部替换）：
     - 不动 backbone 与其他分支，实现难度最低。

3. **第三阶段**：  
   - 在 **拓扑轨迹 head** 上，引入 GPT 自回归或「图+GPT refine」结构：
     - 先利用 MapTR 的 lane graph，保证几何合法；
     - 再由 GPT 学习更细致的轨迹形状与速度控制。

4. **第四阶段**：  
   - 在 **e2ehead** 上，引入 diffusion 作为对 baseline 的增强版本：
     - 条件 = BEV + det/map instance + 目标 VQ code + topo GPT 输出；
     - 确保有清晰的指标与可视化手段评估其收益。

整体目标是：  
- 保持从图像到端到端轨迹的**可导链路**；  
- 同时利用 VQ / GPT / Diffusion 在各自最适合的问题子块上发挥优势；  
- 通过分阶段实现与替换，控制工程复杂度与调试风险。  

