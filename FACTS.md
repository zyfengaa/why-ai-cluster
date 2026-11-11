# 全书事实表（Single Source of Truth）

> 本文件锁定全书跨章节引用的数字与口径。各章正文只引用这里的值，不再各自推算。
> 最后核对：2026-10-17。硬件与集群数据来自厂商公开规格、官方博客与论文；标注「公开拆解」「厂商口径」的数字不作精确承诺。

---

## 0. 四条口径约定

| 口径 | 约定 |
|------|------|
| **稠密 / 稀疏** | 算力默认写**稠密（dense）**；引用稀疏值时必须在同一句标注「稀疏」。二者相差 2× |
| **权重 / 完整训练状态** | 「模型显存」指 BF16 权重（2 B/参数）；「训练状态」指权重 + 梯度 + FP32 master + Adam m/v = **16 B/参数** |
| **聚合 / 单客户端** | 存储与网络带宽默认写**聚合值**；单客户端/单卡值必须标注「单客户端」 |
| **单向 / 双向** | PCIe、NVLink、IB 端口默认写**双向**；单向值必须标注「单向」 |
| **GB / GiB** | 正文统一用 GB/TB（十进制，10⁹/10¹² 字节）；显存规格按厂商标称 |

### 参数显存账本

| 模型 | 参数量 | BF16 权重 | FP32 训练状态（16 B/参数） | 仅权重+优化器（14 B/参数） |
|------|--------|-----------|---------------------------|---------------------------|
| 7B | 7×10⁹ | 14 GB | 112 GB | 98 GB |
| 8B | 8×10⁹ | 16 GB | 128 GB | 112 GB |
| 13B | 13×10⁹ | 26 GB | 208 GB | 182 GB |
| 70B | 70×10⁹ | **140 GB** | **1.12 TB** | **980 GB** |
| 405B | 405×10⁹ | **810 GB** | **6.48 TB** | **5.67 TB** |

> 训练 FLOPs 估算：`C ≈ 6 × N × D`（N=参数量，D=训练 token 数）。
> Llama 3 70B：6 × 70e9 × 15e12 ≈ **6.3×10²⁴ FLOPs**；405B：6 × 405e9 × 15.6e12 ≈ **3.8×10²⁵ FLOPs**。

### Llama 3 架构参数

| 模型 | hidden_dim | layers | heads | KV heads (GQA) | vocab |
|------|-----------|--------|-------|----------------|-------|
| Llama 3 8B | 4096 | 32 | 32 | 8 | 128,256 |
| Llama 3 70B | **8192** | 80 | 64 | 8 | 128,256 |
| Llama 3 405B | 16384 | 126 | 128 | 8 | 128,256 |

---

## 1. GPU 规格（默认稠密）

| GPU | 显存 | HBM 带宽 | FP16/BF16 | FP8 | FP4 | TDP |
|-----|------|---------|-----------|-----|-----|-----|
| A100 80GB SXM | 80 GB HBM2e | 2.0 TB/s | 312 TFLOPS | — | — | 400 W |
| H100 SXM | 80 GB HBM3 | 3.35 TB/s | **989 TFLOPS**（稀疏 1,979） | **1,979 TFLOPS**（稀疏 3,958） | 不支持 | 700 W |
| H200 SXM | 141 GB HBM3e | 4.8 TB/s | 989 TFLOPS | 1,979 TFLOPS | 不支持 | 700 W |
| B200 | 192 GB HBM3e | 8 TB/s | **2.25 PFLOPS**（稀疏 4.5） | **4.5 PFLOPS**（稀疏 9） | **9 PFLOPS**（稀疏 18） | ~1,000 W |
| GB300 / B300 | 288 GB HBM3e | 8 TB/s | 约 B200 的 1.1–1.2× | 约 B200 的 1.1–1.2× | 约 B200 的 1.5×（厂商口径） | **1,400 W** |
| Rubin（2026） | 288 GB+ HBM4 | ~13 TB/s（厂商口径） | 以 NVIDIA 官方发布为准 | 以官方为准 | 相对 Blackwell 约 3.3×（推理，厂商口径） | ~1,800 W |

- H100 SXM CUDA Core：**16,896**；H100 PCIe：**14,592**。
- A100 80GB / H100 80GB 均为 **5 个 HBM stack、5,120-bit**。
- H100 NVL：**94 GB/卡，双卡 188 GB**。
- H100 **没有 FP4**；FP8 从 Ada（8.9）/Hopper（9.0）开始支持，不是「Compute Capability 8.0+」。
- 单卡 FP32：H100 67 TFLOPS；TF32 稠密 494 / 稀疏 989 TFLOPS。

### 互联

| 技术 | 每卡/端口带宽 | 说明 |
|------|--------------|------|
| PCIe 5.0 x16 | **64 GB/s 单向，128 GB/s 双向** | PCIe 4.0 为 32 / 64 GB/s |
| NVLink 4（H100） | **900 GB/s 双向** | 18 link × 50 GB/s |
| NVLink 5（Blackwell） | **1.8 TB/s 双向** | 18 link × 100 GB/s |
| NVSwitch 3（Hopper） | 64 port × 50 GB/s ≈ 3.2 TB/s/芯片 | DGX H100 用 4 颗，每 GPU 18 link（5+4+4+5） |
| NVSwitch 4（Blackwell） | 72 port × 100 GB/s ≈ 7.2 TB/s/芯片 | NVL72 机柜聚合约 130 TB/s |
| IB NDR | 400 Gb/s = **50 GB/s 单向**/端口 | 8×400G = 3.2 Tb/s = 400 GB/s/服务器 |
| IB XDR | 800 Gb/s = 100 GB/s 单向/端口 | — |
| UALink 1.0 | 800 Gb/s = 100 GB/s 单向/端口 | 不是 1,600 GB/s |

- DGX A100：**6 颗 NVSwitch 2**；DGX H100：**4 颗 NVSwitch 3**。
- NVLink 域上限：H100 8 卡；NVL72 72 卡；Rubin NVL144；Rubin Ultra NVL576。跨机柜仍走 IB/以太网，不是「256 卡 NVLink 全互联」。
- 8 卡 All-Reduce 的通信量是**数百 MB 级**（70B 梯度 140 GB / 8 卡仍为百 GB 级），耗时是**毫秒级**，不是 200–500 μs。

---

## 2. 网络与集体通信

- Ring All-Reduce：**reduce-scatter N−1 步 + all-gather N−1 步 = 2(N−1) 步**；每卡通信量 `2S(N−1)/N ≈ 2S`，**与 N 基本无关**；耗时 `T ≈ 2S(N−1)/(N·B)`。
- Tree All-Reduce：时延 `O(log N)`，不是 Ring 的「对数增长」；Ring 时延 `O(N)`。
- 单卡 All-Reduce 实测带宽：单机 8×H200 NVLink 可达 **>400 GB/s**（busbw 口径）；跨机 IB NDR 每卡 **~40–50 GB/s**（受单端口 50 GB/s 限制），不是 80 GB/s。
- IB 子网：16-bit LID，单播 LID 约 **4.8 万个**，不是「理论无限」；扩展靠 subnet router。
- RoCEv2 不是 TCP，没有「TCP 回退」；ECN/PFC 负责无损，不负责链路故障检测。
- 大集群不是「几乎都用 IB」：Meta Llama 3 有一套 RoCEv2 集群；xAI Colossus 用 Spectrum-X 以太网；Google 用 ICI；AWS 用 EFA。
- UEC 1.0 发布于 **2025-06-11**，Linux Foundation 项目；LLR/CBFC 是**可选**能力，不是强制三项。
- 交换机：Tomahawk 6 **102.4 Tb/s**；Spectrum-6 **102.4 Tb/s、800G/端口**；1.6T 是下一代（CX-9 / Spectrum-7 级别）。
- 拓扑：Meta Llama 3 是三层 Fat-Tree，T0–T1 2:1、T1–T2 7:1；Google TPU 是 3D Torus；「>16,384 卡只能用 Dragonfly+」不成立。

### 1,000 卡集群网络账本

- 1,000 GPU = 125 台 8 卡服务器；每台 8×400G = 3.2 Tb/s。
- 聚合网络带宽 = 125 × 3.2 = **400 Tb/s**；1:1 非阻塞时对分带宽约 **200 Tb/s**。
- 64 端口 400G 交换机：64 × 400 Gb/s = **25.6 Tb/s** 交换容量。
- 1:1 无收敛叶脊：32 下联 / 32 上联 → 需要 **32 台 leaf、16 台 spine**（每 leaf 上联 16 台 spine 各 1 链路），不是 50 leaf / 8 spine。

---

## 3. 集群案例

| 集群 | 事实 |
|------|------|
| **Meta Llama 3** | 两套 **24,576 GPU** 集群（一套 RoCEv2、一套 IB NDR）；405B 在 **16,384 GPU** 上训练；54 天 **419 次意外中断**，约 78% 为硬件原因 |
| **xAI Colossus** | 2024 年建成约 **100K H100**，后扩至约 **200K H100/H200**；8 卡/服务器、8 服务器/机架 = 64 GPU/机架；每服务器 8×BlueField-3 + 1×ConnectX-7 = **9×400G = 3.6 Tb/s**；后端为 **Spectrum-X 以太网**，三层网络 |
| **Google TPU v7 Ironwood** | 9,216 芯片/Pod；192 GB HBM3E、7.37 TB/s；3D Torus；ICI |
| **Microsoft/OpenAI Stargate** | OpenAI + SoftBank + Oracle + MGX；4 年 **$500B**，首期 **$100B**；2029 年 **10 GW**；Microsoft 为技术伙伴；Fairwater 是独立园区 |
| **Meta Prometheus** | 1 GW 级规划，俄亥俄州；不是已建成 1 GW |

- GB300 NVL72 整柜约 **132 kW**；DGX B300 8 卡整机约 **14–15 kW**（可风冷）；「8×GB300 必须液冷」不成立，**NVL72 整柜**才必须 DLC。
- 液冷：DLC 用**水-乙二醇**，不是「导电水」；浸没式用介电液。
- CXL 池化的是**主机 DRAM**，不是 HBM；GPU 显存扩展靠 NVLink-C2C，不是 CXL。
- 16×H100 服务器价格约 **$400K–700K**，不是 $50–80K。

---

## 4. 指标定义

| 指标 | 正确定义 |
|------|---------|
| **MFU** | 实际模型 FLOPs/s ÷ **稠密**峰值 FLOPs/s。Llama 3 405B 公开值 **38–43%** |
| **GPU-Util（nvidia-smi）** | 采样期内**至少有一个 kernel 在执行**的时间比例；不是 CUDA Core 占用率，也不是算力利用率 |
| **SM Active（DCGM）** | `DCGM_FI_PROF_SM_ACTIVE`，**0–1 比例**；表示有 warp 驻留，不等于有效计算 |
| **GPU 显存利用率** | `DCGM_FI_DEV_MEM_COPY_UTIL` 是带宽利用率，不是容量占用；容量看 `FB_USED` |
| **Checkpoint 开销** | 保存耗时 ÷ checkpoint 间隔；生产常用 **1–5%**，按 MTBF 与恢复目标定 |
| **慢节点影响** | 同步训练步时 `T_step = max_i(T_i)`；单卡慢 10% → 整步约慢 10%（吞吐降约 9%），不是 10× |
| **大 batch 与通信** | All-Reduce 通信量由**参数量**决定，与 batch size 无关；大 batch 提高计算/通信比 |

---

## 5. NCCL / PyTorch 事实

- `NCCL_TIMEOUT` **不存在**；只有 `NCCL_IB_TIMEOUT`。框架超时用 `init_process_group(timeout=...)` 或 `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC`。
- `NCCL_ASYNC_ERROR_HANDLING` 是 PyTorch 旧变量；现代用 `TORCH_NCCL_ASYNC_ERROR_HANDLING`（默认 3）。
- NCCL 默认算法是 **auto**，不是 Ring。
- 不推荐设置 `NCCL_NTHREADS`；H100 是 **NVLink 4**，NVLink 5 是 Blackwell。
- CollNet = **IB SHARP**（NCCL 2.6+，跨机）；NVLS = **NVLink SHARP**（机内，Hopper+）。二者都不是「NVSwitch 4 新增」。
- NCCL 是用户态库，`dmesg` 里没有 NCCL 日志；用 `NCCL_DEBUG=INFO` / `NCCL_DEBUG_FILE`。dmesg 用来看 Xid、PCIe AER、IB 内核消息。
- nccl-tests 输出：数据行最后一列是 `#wrong`，`busbw` 只在表头；解析要按列或 `--csv`，不能 `grep busbw`。
- 标准 `DataLoader` 没有 `state_dict()`；断点续训用 PyTorch 2.4+ 的 `StatefulDataLoader`。
- `torch.distributed.checkpoint` 的 GDS 集成不是 `torch.distributed.checkpoint.GDS`；GDS 实验接口在 `torch.cuda.gds`。
- `ncu --query-metrics` 只列指标后退出，不运行应用；用 `--set full` / `--section SpeedOfLight`。

---

## 6. 时间线

| 事件 | 时间 |
|------|------|
| GPT-3 | 2020-05 |
| GPT-4 | **2023-03** |
| Llama 3 | 2024-04 |
| Megatron-LM 论文 | 2019（Shoeybi et al.） |
| Megatron-Core | 2023 |
| ZeRO | 2020 |
| FSDP 论文 | 2023 |
| PyTorch FSDP2 | 2.4 起可用，2.6 起 API 移入 `torch.distributed.fsdp` |
| UEC 1.0 | 2025-06-11 |
| NVIDIA 收购 SchedMD | **2025-12-15**（公告） |
| Slinky 开源 / v1.0.0 | 2024 / 2025-11-20 |
| Guard 论文 | MLSys 2026（arXiv:2605.17879） |

---

## 7. 推理与 GPU 共享

### KV Cache

- 公式：`KV = 2 × layers × kv_heads × head_dim × seq × batch × bytes`
- Llama 3 70B、FP16：**320 KiB/token**（`2 × 80 × 8 × 128 × 2`）；8K 单序列 ≈ **2.5 GiB**；128K 单序列 ≈ **40 GiB**
- TP=8 时，8 个 KV head 被切成 1 个/卡 → 每卡每 token ≈ **40 KiB**
- decode 每步要读「全部权重 + 全部活跃 KV」：70B 在 TP=8 下每卡权重读 **17.5 GB/step**
- 推理指标：**TTFT**（首 token 延迟，由 prefill 决定）、**TPOT**（每输出 token 延迟，由 decode 决定）；训练侧的 MFU 不适用于 decode

### GPU 共享模式

| 模式 | 隔离范围 | 训练 | 推理/多租户 |
|------|---------|------|------------|
| 整卡 | 完整 | ✅ 默认 | ✅ |
| **MIG** | SM / 显存 / L2 / 带宽（硬件分区）；实例间不能 P2P/NVLink | ❌ 不推荐（算力与通信都变小） | ✅ 优先 |
| **MPS** | 共享 SM；显存需手动设上限，无强隔离 | ⚠️ 仅特定场景 | ⚠️ 需额外约束 |
| **Time-slicing** | 时间片，无显存/故障隔离 | ❌ | ❌ 仅开发调试 |

### 调度指标

- Slurm FairShare 是**按历史用量衰减的加权公平**，不是平均分；衰减由 `PriorityDecayHalfLife` 控制
- K8s 原生只有 `PriorityClass` + 抢占；排队/配额借用（`ClusterQueue`、`cohort`）由 Kueue 提供，不是 `ResourceQuota` 的职责
- 回填的前提：`小任务墙钟 ≤ 大任务预计开始时间 − 当前时间`；未设墙钟的开发任务是 backfill 的头号元凶

### Checkpoint 间隔

- 总开销率 ≈ `S/I + (1/MTBF) × (I/2 + R)`；最优间隔 `I* ≈ sqrt(2 × S_eff × MTBF)`
- `S_eff` 是异步保存后的有效停顿；`R` 是恢复耗时；`I/2` 是平均丢失进度
- 表中 MTBF 按故障率随卡数线性增长近似，真实值须从工单统计

---

## 8. 引用规则

1. 每篇文章末尾必须有 `> **参考资料**：`，至少 3 条可点击链接。
2. 优先顺序：官方文档 / 官方博客 / 论文 arXiv / 厂商规格页 / 权威工程博客。
3. 不引用无法核验的第三方聚合站；不确定的 URL 先用 WebFetch 验证，返回 404 就换。
4. 正文中的关键数字，尽量在同一段或参考资料中给出可核验来源。
