# Q: NVIDIA 收购 SchedMD 意味着什么？

## 为什么会有这个问题

2025 年 12 月 15 日，NVIDIA 宣布收购 SchedMD——Slurm 项目当前的主要维护公司（Slurm 本身由 LLNL 于 2001/2002 年创造，SchedMD 成立于 2010 年）。消息一出，HPC 和 AI 社区炸开了锅：

- "NVIDIA 要'杀'掉 Slurm？"
- "Kubernetes 要统一天下了？"
- "SchedMD 这家 15 年的公司被收编了？（Slurm 项目 20+ 年，SchedMD 公司 15 年）"

**事实是：Slurm 不会死，但它在 AI 集群中的角色正在发生根本性变化。**

> 一个常被忽略的背景：**NVIDIA 与 Slurm 的官方合作早于本次收购十余年**——NVIDIA 很早就通过 **DGX SuperPOD + Base Command Manager（BCM）** 官方集成 Slurm（同时支持 K8s/Run:ai）。所以"拥抱 Slurm"并不是这次收购才开始的。

## 解决什么问题：AI 集群的"调度分裂"

2025 年前，AI 集群面临一个尴尬的局面：

```
同一套 GPU 硬件，需要两套调度系统：

Slurm → 管理训练（批处理作业）
Kubernetes → 管理推理（长运行服务）

问题是：
  - 两套系统维护两套用户、两套权限
  - GPU 资源被硬性划分：50% 给 Slurm 池，50% 给 K8s 池
  - 训练池空闲时，推理池的 GPU 被浪费——反之亦然
  - 运维团队需要掌握两种调度器的技能
```

**业界的推测（非官方口径）**：有人认为 NVIDIA 收 SchedMD，是为了让 Slurm 成为 Kubernetes 的一个工作负载，用 K8s 做统一底座。**但这只是分析，不是官方表态**——NVIDIA 的收购公告只承诺"保持 Slurm 开源与厂商中立、继续为数百家客户提供支持、HPC + AI 双线并进"，**没有任何"把 Slurm 包进 K8s"的表述**；而且 Slinky 在收购前就已存在（1.0.0 发布于 2025-11-20）、CoreWeave 的 SUNK 在 2024 年初就已上线——"Slurm 跑在 K8s 上"不是这次收购带来的转变。下面这张图是**分析视角**，不是官方路线图。

### 收购的分析视角（业界推测，非官方口径）

```
收购前：
  Slurm ← 独立运营，由 SchedMD 维护
  Kubernetes ← CNCF 社区
  NVIDIA ← 卖 GPU，不管调度

收购后：
  NVIDIA ← 控制 Slurm 的发展方向
    └→ 开源 Slinky：Slurm on Kubernetes
        └→ Slurm 跑在 K8s 上 → K8s 成为统一底座
            └→ NVIDIA GPU Operator 统一管理 GPU 资源
                └→ 同一套 GPU 设备插件、同一套网络配置
```

### Slinky（Slurm on Kubernetes）

Slinky 是 **SchedMD 自己的开源项目**（v1.0.0 于 2025-11-20 发布，**早于 2025-12-15 的收购**；KubeCon Europe 2025 已有演讲）：

```
架构：
  ┌─────────────────────────────────┐
  │       Kubernetes 集群            │
  │  ┌──────────────────────────┐   │
  │  │   Slurm Pod              │   │
  │  │  ┌─ slurmctld ─────────┐ │   │
  │  │  │ 作业调度 + 排队管理 │ │   │
  │  │  └────────────────────┘ │   │
  │  │  ┌─ slurmd ───────────┐ │   │
  │  │  │ 节点代理（每 GPU 节点）│ │   │
  │  │  └────────────────────┘ │   │
  │  └──────────────────────────┘   │
  │  ┌──────────────────────────┐   │
  │  │  K8s Pod（推理服务）    │   │
  │  │  vLLM / TGI / TensorRT  │   │
  │  └──────────────────────────┘   │
  └─────────────────────────────────┘
```

**性能参考**（来自 NVIDIA 官方博客，而非"内部测试"）：
- 已验证 8,000 GPU 规模的 Slurm on Kubernetes 部署
- 相比裸金属 Slurm，**性能损失可忽略**
- 训练和推理可以共享 GPU 池 → 减少闲置 GPU、提高复用率

### CoreWeave SUNK

CoreWeave（云 GPU 提供商）在 2024 年初就已推出生产级混合方案 SUNK：

```
SUNK = CoreWeave 的"Slurm on Kubernetes"方案
       （CoreWeave's system for running Slurm on Kubernetes）

特点：
  - 同一集群同时跑 Slurm 训练 + K8s 推理
  - 用户通过 sbatch 提交训练任务
  - 推理服务通过 kubectl 管理
  - GPU 资源自动在训练和推理之间动态分配
  - 减少"闲置 GPU"（训练池空闲时自动给推理用）
```

---

## 这对生态意味着什么

收购引发的最集中的争议是**厂商中立性**：Slurm 长期是跨厂商、跨站点的公共 HPC 调度器，如今它的主要维护方归于一家 GPU 厂商，社区自然会担心它的"中立性"还能维持多久——这正是 HPCwire 2026 年 1 月那篇文章的主旨。NVIDIA 回应称会保持 Slurm 开源与厂商中立（见文末参考链接）。

### 对 Slurm 用户

| 变化 | 影响 |
|------|------|
| Slurm 继续开源 | ✅ 不会被杀 |
| SchedMD 团队保留 | ✅ 核心开发者继续维护 |
| 新功能优先支持 AI 场景 | ✅ 更好的 GPU 拓扑感知、NCCL 集成 |
| NVIDIA GPU Operator 深度集成 | ✅ 开箱即用的 GPU 调度 |
| Slinky 成为默认部署方式 | ⚠️ 新部署可能从"裸金属 Slurm"转向"K8s + Slinky" |

### 对 Kubernetes 用户

| 变化 | 影响 |
|------|------|
| K8s 成为 AI 集群的统一底座 | ✅ 训练 + 推理终于统一 |
| KAI Scheduler 持续开源 | ✅ NVIDIA 没有放弃 K8s 原生方案 |
| GPU 动态分时共享 | ✅ 训练空闲 → 推理自动用上 |
| 不需要同时学两个调度器 | ✅ 运维团队只需精通 K8s |

### 对 NVIDIA

```
NVIDIA 的战略清晰了：
  "GPU 硬件 → 统一调度层 → 你的 AI 工作负载"
              ↓
         Kubernetes
              ↓
   Slinky（Slurm）| KAI Scheduler | GPU Operator
              ↓
    你的训练任务 → 你的推理服务 → 你的数据流水线
```

---

## 🛠 实操指南

### 📋 决策指南：收购后你的集群策略该不该变？

```
你当前的集群用了什么调度器？
├─ 裸金属 Slurm →
│   ├─ 不动：Slurm 继续维护，现有作业无需迁移
│   ├─ 可考虑：新增推理服务时引入 K8s + Slinky 混合
│   └─ 长期：数据中心扩建时考虑 Slinky 方式
│
├─ Kubernetes →
│   ├─ 不动：KAI Scheduler / Volcano 继续使用
│   ├─ 可考虑：如果训练调度不满意，试试 Slinky
│   └─ 好消息：GPU Operator 集成更深了
│
└─ 正在搭建 →
    └─ 推荐方向：Kubernetes + Slinky/KAI Scheduler
        ├─ 训练团队习惯 sbatch → Slinky 提供
        ├─ 推理团队习惯 K8s → 原生提供
        └─ GPU 管理 → GPU Operator
```

### 🔧 自己动手验：试试 Slinky 的基本概念

```bash
# 1. 检查你是否已经在用 Slinky
kubectl get pods -n slurm-system 2>/dev/null || echo "没有检测到 Slinky"

# 2. 在 Kubernetes 上模拟 Slurm 作业提交
# 安装 slurm-client
apt-get install slurm-client

# 提交训练任务到 Slinky 集群
sbatch --partition=gpu --gpus=8 --wrap="torchrun --nproc_per_node=8 train.py"

# 3. 查看 GPU 利用率变化（训练 + 推理混合时的效果）
nvidia-smi dmon -s puct -d 30
# 如果训练空闲时 GPU 利用率仍 > 50% → 推理服务在复用 GPU
```

### ⚠️ 常见误区

- **误区**："NVIDIA 收购 SchedMD 后 Slurm 会闭源"
  - **实际**：Slurm 仍然是 GPL v2 开源项目。NVIDIA 在收购公告中明确承诺**继续开源**。Slinky 也是开源的。

- **误区**："收购后所有集群都必须用 K8s"
  - **实际**：没有"必须"。Slurm 仍然可以直接跑在裸金属上。Slinky 只是多了"跑在 K8s 上"的选项——**不是替代，是补充**。

- **误区**："Slinky 就是 Slurm 容器化，没什么新意"
  - **实际**：核心创新在于**Slurm 和 K8s 共享 GPU 资源池**。在此之前，训练和推理的 GPU 是硬性隔离的，Slinky 让它们可以动态共享，大幅提升利用率。

### 📊 底线数字速查

| 指标 | 裸金属 Slurm | 纯 K8s | Slinky（Slurm on K8s）|
|------|-------------|--------|----------------------|
| 训练性能 | baseline | ~1% loss | ≈0% loss |
| 推理性能 | N/A | baseline | baseline |
| GPU 利用率 | 60-70%（训练池） | 70-80% | **80-90%**（动态共享）|
| 运维复杂度 | 1 套系统 | 1 套系统 | **~1 套系统**（K8s 统一）|
| 用户学习成本 | 低（HPC 团队） | 低（DevOps 团队） | **低（各有各的接口）**|
| 部署成熟度 | 20 年验证 | 5 年验证 | 2025-2026 新推出 |

### 💡 一句话记住

> **"NVIDIA 收购 SchedMD，官方承诺是'保持 Slurm 开源与厂商中立、继续支持数百家客户'。至于'要用 K8s 把 Slurm 包起来'——那更多是业界分析（毕竟 Slinky 收购前就存在、SUNK 2024 年初已上线），并非官方表述。无论哪种解读，2026 年'Slurm on K8s'都在成为 AI 集群的重要选项。"**

---

## 对后续的铺垫

有了调度器，有了 Slurm on K8s，但还有一个关键问题：**分布式训练要求"所有 GPU 同时到位"才能启动**。

这就是 Gang Scheduling（集合调度）要解决的问题。

→ [[05-Gang Scheduling]]

---

> **参考资料**：
> - ["NVIDIA Acquires Open-Source Workload Management Provider SchedMD" — NVIDIA 官方公告 (2025-12-15)](https://blogs.nvidia.com/blog/nvidia-acquires-schedmd/)
> - ["What Does NVIDIA's Acquisition of SchedMD Mean for Slurm?" — HPCwire (2026)](https://www.hpcwire.com/2026/01/06/what-does-nvidias-acquisition-of-schedmd-mean-for-slurm/)
> - [CoreWeave SUNK 文档](https://docs.coreweave.com/products/sunk/)
> - [Slinky 项目 — GitHub 组织 SlinkyProject](https://github.com/SlinkyProject)
> - [Slurm Slinky 官方文档](https://slurm.schedmd.com/slinky.html)
> - [NVIDIA: "Running Large-Scale GPU Workloads on Kubernetes with Slurm"](https://developer.nvidia.com/blog/running-large-scale-gpu-workloads-on-kubernetes-with-slurm/)
> - [Crusoe: "Slurm on Crusoe Managed Kubernetes" (2026)](https://www.crusoe.ai/resources/blog/slurm-on-crusoe-managed-kubernetes-how-we-built-managed-gpu-training-infrastructure)
