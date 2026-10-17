# Q: 为什么要关心 GPU 利用率（MFU）？

## 为什么会有这个问题

上篇讲到多卡通信有开销。但你可能有个更大的疑问：**"我买的 GPU 到底跑满了没有？"**

一张 H100 价格约 $30,000。一个 1,000 卡集群光是 GPU 就 $3,000 万，每年电费还要几百万。如果利用率只有 20%，意味着你花 5 块钱只买到 1 块钱的算力。

但问题在于：**"利用率"这个数字极容易被误解。**

---

## 解决什么问题：引入 MFU（Model FLOPs Utilization）

**MFU** 是目前业界公认衡量集群训练效率的核心指标（由 Google 在 PaLM 论文中首次明确提出）。定义：

```
MFU = 实际达到的模型 FLOPs/s ÷ GPU 的稠密峰值 FLOPs/s
```

**分母必须是稠密（dense）峰值**。H100 的 FP16/BF16 稠密峰值是 989 TFLOPS（稀疏 1,979）——MFU 永远拿实际算力除以 989，而不是除以稀疏值，否则等于白送一倍水分。不同精度峰值不同，报 MFU 时要注明精度口径。

### 别和 nvidia-smi 的 GPU-Util 搞混了

| 指标 | `nvidia-smi GPU-Util` | MFU |
|------|----------------------|-----|
| 含义 | 采样期内**至少有一个 kernel 在执行**的时间比例 | GPU 稠密峰值的实际兑现比例 |
| 说人话 | "我有没有在干活？" | "我干活的时候有多卖力？" |
| 100% 代表什么 | 采样期内一直有 kernel 在跑，但可能是低效算子 | 跑满了理论稠密峰值 |
| 能反映通信瓶颈吗？ | ❌ 不能 | ✅ 能（通信时 MFU 会下降） |

### 例子

一个 GPU 1 秒内：前 0.8 秒全速计算（989 TFLOPS 稠密），后 0.2 秒等待梯度同步。

- `nvidia-smi GPU-Util` = 80%（因为 80% 的时间里至少有 kernel 在跑）
- 如果计算时只跑出了 50% 的稠密峰值，MFU = 80% × 50% = 40%

所以 `nvidia-smi` 100% ≠ MFU 100%。GPU-Util 只回答"忙不忙"，不回答"忙得有没有价值"。这也是为什么从业者常说："**不要只看 nvidia-smi，会骗人的。**"

---

## 影响 MFU 的主要因素

按影响力排序：

| 因素 | 影响程度 | 说明 |
|------|---------|------|
| 通信开销 | 最大 | GPU 等梯度 → MFU 直接下跌 |
| 数据加载 | 严重 | GPU 等数据 → MFU 下跌 |
| 算子效率 | 中等 | 非 Tensor Core 算子拖后腿 |
| Batch Size | 中等 | 太小 → GPU 算不满 |
| 并行策略匹配度 | 中高 | TP/PP/DP 没搭好 → 通信和计算不平衡 |
| 混合精度 | 可控 | FP8 > BF16 > FP32，算力利用不同 |

### 业界参考值

全书的统一分档口径（跨文章一致）：

| MFU 范围 | 评价 | 代表场景 |
|----------|------|---------|
| 10-20% | 差 | 裸数据并行，没有梯度累积 |
| 25-35% | 一般 | 有基本的梯度累积和 AMP 混合精度 |
| 40-50% | 良好 | 3D 并行 + Profile 针对性优化 |
| 50-65% | 优秀 | Megatron-Core + 极致调优 |
| 70%+ | 顶级 | 极少见，通常只有高度优化的 TP 组内能达到 |

公开可查的锚点：Meta 训练 Llama 3 405B 时的 MFU 为 **38-43%**（论文披露口径）——"良好"档。可见千卡以上的真实大训练，MFU 没有营销数字那么漂亮。

---

## 优缺点

### MFU 作为指标
| 优点 | 缺点 |
|------|------|
| 统一口径，跨模型/硬件可对比 | 只反映"算力利用"，不反映"训练效率"（收敛速度） |
| 帮助定位瓶颈 | 不同精度的理论峰值不同，需要注明精度 |
| 量化集群调优效果 | 小模型 MFU 天然低（算力过剩），不全是问题 |

---

## 🛠 实操指南

### 📋 决策指南：你的集群 MFU 该是多少？

不同的模型大小和硬件配置，MFU 的"及格线"不一样：

| 你的场景 | 期望 MFU | 如果低于这个值，最可能的原因 |
|---------|---------|--------------------------|
| 单 8-GPU 节点微调 7B 模型 | 30-40% | 没开 `torch.compile`，或 batch size 太小 |
| 单节点全量微调 13B | 35-45% | 检查数据加载是不是瓶颈（dataloader 停顿） |
| 2-4 节点训练 70B（FSDP2） | 35-50% | 网络通信开销太大 → 检查 NCCL busbw |
| 16+ 节点训练 70B+（Megatron） | 45-60% | 并行策略配置不合理（TP/PP/DP 没匹配好） |
| 千卡级预训练 | 38-50% | 参考 Meta Llama 3 405B 公开值 38-43%；低于 30% 要查网络与并行配置 |

**一条经验法则**：如果你的 MFU < 25%，大概率有"低级错误"——比如数据加载没做好、通信没走 RDMA、或者 Batch Size 设得太小。**先解决低级错误，再追求高级优化。**

### 🔧 自己动手验：怎么算你当前的 MFU？

```python
# 在你的训练代码中加入这段：
import torch

def compute_mfu(num_flops_per_step, step_time_seconds, num_gpus):
    """
    参数：
      num_flops_per_step: 每步的模型 FLOPs ≈ 6 × 参数量 × 每步 token 数
                          可以从 Megatron / DeepSpeed 的 log 里拿
      step_time_seconds: 你训练日志里看到的 step time
      num_gpus: GPU 数量
    返回：MFU（百分比）
    """
    gpu_peak_dense_tflops = 989  # H100 FP16/BF16 稠密峰值（MFU 分母必须是稠密值）
    theoretical_peak = num_gpus * gpu_peak_dense_tflops * 1e12
    actual = num_flops_per_step / step_time_seconds  # 集群实际 FLOPs/s
    mfu = actual / theoretical_peak * 100
    return mfu

# 举例：Llama 3 70B，1,024×H100，每步 1024 卡 × 4096 token
# num_flops_per_step ≈ 6 × (70×10^9) × (1024 × 4096) ≈ 1.76e18
# 实测 step_time = 4.3 s
# 集群实际算力 = 1.76e18 / 4.3 ≈ 4.1e17 FLOPs/s
# 理论峰值    = 1024 × 989e12 ≈ 1.01e15 FLOPs/s
# MFU = 4.1e17 / 1.01e15 ≈ 40%
#
# 注意先做量纲检查：如果算出来的 MFU > 100%，说明你的 FLOPs 数或峰值口径错了
# （比如误用了稀疏峰值，或者把乘了 2 倍的 FLOPs 又乘了一遍）
```

（一个常见的翻车现场：拿 1.38e16 FLOPs 除以 1.2 s 再除以 1024 张卡 × 2e15——分子分母的口径全对不上，算出个 0.56% 还以为集群烂透了。先确认三件事：每步 FLOPs 是 6ND、step time 是实测的、峰值是稠密的。）

**不用自己算的方法**：用 PyTorch Profiler 或 Nsight Systems：

```bash
# 用 Nsight Systems 录制一小段训练
nsys profile -o profile_output python train.py

# 打开 profile_output.qdrep → 看 GPU 计算时间 vs 通信等待时间
# GPU 计算时间 / 总时间 ≈ MFU 的上限
```

### ⚠️ 常见误区

- **误区**："nvidia-smi 显示 GPU-Util 100% = GPU 跑满了"
  - **实际**：GPU-Util 的含义是"采样期内至少有一个 kernel 在执行的时间比例"，不等于"算力跑满"。一个 GPU 可能在跑低效的 FP32 算子，利用率 100% 但 MFU 只有 20%。**要真的看算力，用 DCGM 或 Nsight Systems 看实际 FLOPs 与 Tensor Core 活动。**
- **误区**："MFU 越高越好，应该追求 70%+"
  - **实际**：小模型（<7B）的 MFU 天然低——因为 GPU 算力过剩，很大一部分时间花在 kernel launch 开销和数据搬运上。**MFU 只有在同规模模型、同硬件之间对比才有意义。**
- **误区**："调优一次，MFU 就稳定了"
  - **实际**：MFU 随训练进程波动。刚开始训练时（warmup 阶段）MFU 低，中间稳定，Checkpoint 写的时候突然下跌。**应该看一段时间的平均 MFU，而不是峰值 MFU。**

### 📊 底线数字速查

| MFU 范围 | 意味着什么 | 你应该做什么 |
|---------|-----------|------------|
| < 10% | **严重问题** | 检查：数据加载、通信是否真正走了 RDMA（IB/RoCE）而不是退化到 TCP/IP、单卡算子效率 |
| 10-20% | 差 | 打开 AMP 混合精度 / BF16，检查是否存在不必要的 CPU-GPU 同步 |
| 25-35% | 一般 | 开始优化：梯度累积、通信重叠（overlap communication） |
| 40-50% | 良好 | 微调并行策略，进一步压通信延迟 |
| 50-65% | 优秀 | 恭喜，你的集群效率已经是业界前列 |
| > 70% | 顶级 | 罕见，通常只在 TP 组内小范围达到；先核对是不是算错了口径 |

### 💡 一句话记住

> **"nvidia-smi 说利用率 100% ≠ 你的 GPU 真的在干活。MFU 才是你花钱买到了多少算力的真实账单。"**

---

## 对后续的铺垫

理解了 MFU 之后，你就会追问：**"是什么在吃我的 MFU？"**

答案是：**通信**。尤其是当你从单机扩展到多机时，通信开销会吃掉大量算力，导致 MFU 急剧下降。

这就是为什么网络成为集群设计的关键环节。

→ 继续读 [[03-网络篇/01-为什么GPU之间需要网络]]，看看"神经系统"怎么搭

---

> **参考资料**：
> - [Chowdhery et al., "PaLM: Scaling Language Modeling with Pathways" (2022)](https://arxiv.org/abs/2204.02311) — 首次明确提出 MFU 及其计算口径
> - [Grattafiore et al., "The Llama 3 Herd of Models" (2024)](https://arxiv.org/abs/2407.21783) — Llama 3 405B 训练 MFU 38-43% 的公开披露
> - [NVIDIA DCGM 用户指南](https://docs.nvidia.com/datacenter/dcgm/latest/user-guide/index.html) — `DCGM_FI_PROF_SM_ACTIVE` 等利用率指标的正确定义
> - [PyTorch Profiler 文档](https://pytorch.org/docs/stable/profiler.html) — 实测每步 FLOPs 与时间的方法
> - [NVIDIA H100 Tensor Core GPU 规格页](https://www.nvidia.com/en-us/data-center/h100/) — 稠密/稀疏峰值的官方口径
