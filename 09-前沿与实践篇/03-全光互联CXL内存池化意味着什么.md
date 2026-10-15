# Q: 全光互联、CXL 内存池化意味着什么？

## 为什么会有这个问题

GPU 越来越快，HBM 越来越贵，内存越来越不够用。

你有一个 70B 模型（按 Llama-3-70B 的规格：80 层、GQA 8 个 KV 头、head_dim 128），128K 的 context window——FP16 的 KV Cache 就要占用约 41 GB（每 token 约 0.31 MB × 131,072）。加上 140 GB 的权重……H100 的 80 GB 显存放不下，需要将数据分片到多张卡。

**核心矛盾**：GPU 的算力每代翻倍，但内存带宽和容量的增长远远落后。

**CXL（Compute Express Link）内存池化和全光互联，就是业界对"内存墙"的回答。**

## 解决什么问题

CXL 和全光互联解决的核心问题：**把内存从"绑死在单机"变成可池化、可分层、可远端访问的资源。** CXL 是一条**以主机（CPU）为中心**的一致性内存总线，解决的是 CPU 侧内存的容量、池化和分层问题；真正给 GPU 直接扩内存的是 **NVLink-C2C（Grace-Hopper，900 GB/s）**。对 AI 来说，CXL 的价值主要在**推理侧 KV Cache / 冷数据的分层存放**，而不是"训练时直接扩容显存"。

### 冯·诺依曼瓶颈的 AI 版本

```
传统架构：
  GPU ──HBM── 本地显存（有限，昂贵，焊死在 GPU 旁）
            ↑ 不够用就得做 ZeRO/FSDP → 通信开销

CXL 内存池化（以主机 CPU 为中心）：
  CPU ──CXL── 本地 DRAM（快速但有限）
      ──CXL── 远端内存池（慢一些但容量巨大）
            ↑ 推理时把 KV Cache 放到 CXL 内存池
            ↑ 训练时把冷参数卸载到 CXL 内存池（经 host 访问）
  注意：CXL 池化的是主机 DRAM，无法把各 GPU 的 HBM 池化；
        GPU 也不能直接 load/store CXL 内存（需经 host 或 P2P 转发）

给 GPU 扩内存（显存侧）：
  GPU ──NVLink-C2C── Grace CPU + LPDDR（900 GB/s，芯片间互联）

全光互联：
  GPU ──光子── 远端内存（带宽堪比本地，延迟接近本地）
            ↑ 打破"距离"和"带宽"的限制
```

### CXL 4.0 的新能力（2025 年 11 月发布）

| 特性 | CXL 3.x | CXL 4.0 | 对 AI 的意义 |
|------|---------|---------|-------------|
| 信号速率 | 64 GT/s | **128 GT/s** | 带宽翻倍 |
| PCIe 基础 | PCIe 6.0 | **PCIe 7.0** | 硬件兼容 |
| 端口捆绑 | — | **1.5 TB/s** | 多端口聚合（单条 x16 为 256 GB/s，捆绑后才到 1.5 TB/s）|
| 作用范围 | 单机架 | **多机架** | 100+ TB 内存池 |
| 延迟 | 200-500 ns | 200-500 ns | 与本地差距缩小 |

CXL 有三条子协议，理解它们才看得懂"池化"是从哪来的：

| 协议 | 作用 |
|------|------|
| **CXL.io** | 发现、配置、DMA——与 PCIe 兼容的基础通道 |
| **CXL.cache** | 设备一致性地访问主机内存（设备侧缓存主机内存）|
| **CXL.mem** | 主机一致性地访问设备内存（把设备内存当自己的内存用，池化的基础）|

能力是一版一版加上去的，别只看"3.x vs 4.0"两列：

| 版本 | 这一版才有的能力 |
|------|-----------------|
| CXL 1.1 | 单级直连（1 主机 ↔ 1 设备），没有交换机 |
| CXL 2.0 | 交换机 + **内存池化**（多主机共享一个内存池）——"池化"从这一版才开始 |
| CXL 3.0 | 多级 fabric、设备间一致性（back-invalidate）、端口捆绑 |
| CXL 3.1 | TSP（Trusted Security Protocol，链路加密）|
| CXL 3.2 | 面向内存设备的优化（更贴近内存模组的形态）|

### Marvell Photonic Fabric（全光互联）

2026 年 HCDS 最佳论文奖：Marvell 的 Photonic Fabric 技术：

```
传统电缆 vs 光互联：

电信号（CXL over copper）：
  距离限制：~2 米
  带宽：随距离下降
  功耗：每 bit ~5 pJ

光信号（Photonic Fabric）：
  距离限制：~300 米（多机架）
  带宽：不随距离下降
  功耗：每 bit ~1 pJ

实际效果（对 LLaMA3-405B）：
  · 推理并发用户密度：提升 16×
  · Time-to-first-token（100K context）：提升 100×
  · Time-to-first-token（1M context）：提升 370×
  · 推理延迟降低：>50%（Marvell 口径）
```

需要说明的是，**商业意义上的"全光互联"不只 Marvell 这一家**：NVIDIA 在 2025 年发布了 Quantum-X / Spectrum-X Photonics 交换机（**CPO，共封装光学**），把光引擎直接封装在交换 ASIC 旁边，2026 年开始上市。CPO 解决的是交换机侧的功耗与密度问题，与 Photonic Fabric 面向的"内存远端访问"是两个层面的全光化。

另一个已经落地多年的全光技术是**光交换（OCS）**：Google 的 Apollo OCS 用可重构光交换把 TPU 集群的拓扑做成可动态重构的（见 [[01-超大集群长什么样]]）。

另外，2025 年 NVIDIA 发布的 **NVLink Fusion** 允许第三方 CPU/加速器以 chiplet 方式接入 NVLink 域——它回答的正是"非 NVIDIA 芯片怎么进 NVLink 一致性域"，也是 NVLink 阵营与 CXL 阵营之间最直接的一个接口。

---

## 三大互联的共存格局

2026-2027 年的 AI 集群中，三种互联同时存在：

```
┌──────────────────────────────────────────────┐
│               AI 集群的三种互联                 │
├──────────────────────────────────────────────┤
│                                              │
│  NVLink / UALink     CXL                    │
│  GPU ⇄ GPU          CPU ⇄ 内存池            │
│  最快（1800 GB/s）   中速（256 GB/s，x16）    │
│  训练通信             内存扩展                │
│                                              │
│  ┌────────────────────────────────────┐      │
│  │       全光互联（Photonic）         │      │
│  │  用光信号替代电信号               │      │
│  │  打破距离限制，连接整个 Pod       │      │
│  │  既是 NVLink 的延伸，也是 CXL 的扩展 │      │
│  └────────────────────────────────────┘      │
│                                              │
└──────────────────────────────────────────────┘
```

| 互联 | 作用 | 带宽 | 延迟 | 距离 |
|------|------|------|------|------|
| **NVLink 5** | GPU-GPU 训练通信 | 1,800 GB/s | ~1 μs | 同机架 |
| **UALink 1.0** | 开放 GPU 互联（AMD/Intel）| ~100 GB/s/方向（200 Gbps/lane × 4 端口 ≈ 800 Gbps）| ~1 μs | 同机架 |
| **CXL 4.0** | CPU-内存池/加速器 | 256 GB/s（单条 x16）| 200-500 ns | 多机架 |
| **Photonic** | 光互联，上述全兼容 | 带宽不限 | 接近电互联 | 300m+ |

---

## 这对 AI 训练意味着什么

### CXL 内存池化的实际应用场景

**场景 1：训练时卸载不常用参数**
```python
# 传统：所有参数在 HBM 中
# CXL 方案：不常用的层放在 CXL 内存池
# 常用（HBM）：Embedding + 最后 4 层
# 不常用（CXL pool）：中间层（前向/反向时临时调入）

# 训练效率：HBM 带宽 CXL pool 带宽
# 如果只用最后 4 层做频繁更新 → 其余层可以从 CXL 读取
```

**场景 2：推理时存储 KV Cache**
```python
# 70B 模型，128K context
# KV Cache 占用：约 41 GB（FP16、GQA 8 个 KV 头）
# 批量一大、或 context 再翻倍，单卡 HBM 就装不下了

# CXL 方案：
# - 模型权重：HBM（在 GPU 旁，快速访问）
# - KV Cache：CXL 内存池（挂在主机侧，慢一点但弹性大）
# - 效果：支持更长的 context 和更大的 batch
```

**场景 3：主机侧内存池化**
```
集群有 64 台服务器，每台 2 TB 主机 DRAM：
  不池化：每台机器只能用自己的 2 TB
          → 有的任务内存不够，有的机器内存空闲
  
  CXL 池化：把各主机（CPU 侧）的 DRAM 聚合成统一池
            → 需要大内存的任务从池中按需分配
            → 大幅提高内存资源利用率

注意边界：CXL 池化的是主机 DRAM（挂接在 CPU 侧），
          不能把各 GPU 的 80 GB HBM 池化起来；
          GPU 也不能直接 load/store CXL 内存，要经 host 或 P2P 转发。
          要给 GPU 扩显存，靠的是 NVLink-C2C（Grace-Hopper）。
```

---

## 🛠 实操指南

### 📋 决策指南：CXL 和光互联适合你吗？

```
你的场景？
├─ 单机训练（8 卡）→
│   └─ CXL 不是刚需，NVLink 够用
│
├─ 长上下文推理（32K+）→
│   └─ CXL 内存池可以大幅降低 KV Cache 成本
│
├─ 显存不够但不想加 GPU →
│   └─ 扩显存靠 NVLink-C2C（Grace-Hopper，900 GB/s）；
│       CXL 只能扩主机内存，把 KV Cache / 冷数据分层放进去
│
└─ 设计下一代集群 →
    ├─ 2026：部署 CXL 3.1/CXL 4.0 就绪设备
    ├─ 2027：考虑 Photonic Fabric 光互联
    └─ 策略：保证基础设施兼容 CXL 交换机
```

### 🔧 自己动手验：检查你的系统是否支持 CXL

```bash
# 1. 检查 CPU 是否支持 CXL
cat /proc/cpuinfo | grep "model name" | head -1
# Intel: 4th Gen Xeon（Sapphire Rapids）+需要 CXL 1.1
# Intel: 6th Gen Xeon（Granite Rapids）+需要 CXL 2.0
# AMD: EPYC Genoa / Turin 支持 CXL

# 2. 检查是否检测到 CXL 内存
# 如果系统有 CXL 内存，会被识别为 NUMA 节点
numactl -H | grep -i cxl
# 如果输出为空 → 无 CXL 内存

# 3. 检查当前内存分配
numastat -v
```

### ⚠️ 常见误区

- **误区**："CXL 内存可以当 HBM 用"
  - **实际**：CXL 内存的延迟（200-500 ns）是 HBM 延迟（~100 ns）的 **2-5 倍**。CXL 适合"容量敏感但延迟容忍"的场景（KV Cache、冷参数），不适合"计算频繁访问"的场景（权重更新）。

- **误区**："光互联是未来的唯一方向"
  - **实际**：光互联在**长距离**（跨机架）和**高带宽**场景有优势，但短距离（同 Chiplet 内、同机架内）电互联的成本和能效仍然更好。**2026-2030 年是"光电混合"时代**，不是全光。

- **误区**："CXL 4.0 发布了，明年就能部署"
  - **实际**：CXL 4.0 产品采样在 2026 年底，**量产在 2027 年**。2026 年能拿到的是 CXL 3.1 产品和 PCIe 6.0 服务器。不要提前设计 CXL 4.0 的生产集群。

### 📊 底线数字速查

| 互联方案 | 带宽 | 延迟 | 距离 | 量产时间 |
|---------|------|------|------|---------|
| CXL 3.1 | 128 GB/s（x16）| 200-500 ns | 机架内 | 2026 H1 |
| CXL 4.0 | 256 GB/s（x16）| 200-500 ns | 多机架 | 2027 |
| Photonic Fabric | 不限 | ~1 μs | 300m+ | 2027+ |
| NVLink 5 | 1,800 GB/s/chip | ~1 μs | 机架内 | 2025 |

### 💡 一句话记住

> **"CXL 内存池化和全光互联是 AI 集群对'内存墙'的回答——HBM 太贵、太少、太焊死了。CXL 把主机 DRAM 池化成分层内存，让 KV Cache 和冷数据有地方可放；Photonic 光互联让你在 300 米外用接近本地的速度访问远端内存。它们补的是容量和分层，不是替代 HBM——真正给 GPU 直接扩容显存的，还是 NVLink-C2C 这一代芯片间互联。"**

---

## 对后续的铺垫

前沿科技看再多，不如动手搭一个。

前面我们聊了几千张 GPU 的集群——但**大部分团队不需要**。如果你只是想入门、做实验、跑小模型，2-4 台机器就够了。

下一节：自己动手搭建一个小集群。

→ [[04-自己动手搭建一个小集群怎么入门]]

---

> **参考资料**：
> - [Introducing the CXL 4.0 Specification — Webinar Q&A Recap](https://computeexpresslink.org/blog/introducing-the-cxl-4-0-specification-webinar-qa-recap-4386/) — CXL Consortium (2025.11)
> - [Bridging the Memory Hierarchy Gap in LLM Inference with Photonic Fabric Architecture](https://cn.marvell.com/blogs/hcds-best-paper-award-winner.html) — Marvell HCDS Best Paper (2026)；英文原博见 [HCDS Best Paper Award Winner](https://www.marvell.com/blogs/hcds-best-paper-award-winner.html)
> - [UALink has Nvidia's NVLink in the crosshairs: final specs support up to 1,024 GPUs with 200 GT/s bandwidth](https://www.tomshardware.com/tech-industry/ualink-has-nvidias-nvlink-in-the-crosshairs-final-specs-support-up-to-1-024-gpus-with-200-gt-s-bandwidth) — Tom's Hardware (2025-04)
> - [Marvell to acquire XConn Technologies for $540M](https://www.nasdaq.com/articles/marvell-technology-buy-xconn-technologies-about-540-mln) (2026)
> - [UALink and CXL 4.0: The Open Standards Reshaping GPU Cluster Architecture](https://introl.com/blog/ualink-cxl-4-gpu-interconnect-memory-pooling-guide-2025) — Introl（厂商营销博客，需交叉核实）(2026)
