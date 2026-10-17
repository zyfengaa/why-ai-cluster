# Q: NVIDIA 收购 SchedMD 意味着什么？

## 为什么会有这个问题

2025 年 12 月 15 日，NVIDIA 宣布收购 SchedMD——Slurm 的创造者和维护公司。消息一出，HPC 和 AI 社区炸开了锅：

- "NVIDIA 要'杀'掉 Slurm？"
- "Kubernetes 要统一天下了？"
- "SchedMD 这家老店被收编了？"

**事实是：Slurm 不会死，但它在 AI 集群中的角色正在发生根本性变化。**

## 解决什么问题：AI 集群的"调度分裂"

收购之前，AI 集群长期面临一个尴尬的局面：

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

**NVIDIA 的思路**：收购 SchedMD，同时让 Slurm 可以作为 Kubernetes 的一个工作负载运行（Slinky）。

### 基本事实卡（先对齐时间线）

| 事实 | 时间/内容 |
|------|----------|
| SchedMD 成立 | **2010 年**（总部在美国犹他州 Lehi，长期是 Slurm 的核心维护方） |
| Slinky 开源 | **2024 年**（SchedMD 推出的"Slurm on Kubernetes"组件集） |
| Slinky v1.0.0 | **2025 年 11 月 20 日**发布（slurm-operator + slurm-bridge） |
| CoreWeave SUNK 开源 | **2024 年初**（SUNK = Slurm on Kubernetes） |
| NVIDIA 收购公告 | **2025 年 12 月 15 日** |
| Slurm 许可证 | GNU GPL / LGPL 开源，收购后 NVIDIA 承诺继续以开源、厂商中立方式维护 |

注意顺序：**Slinky 和 SUNK 都诞生在收购之前**。它们是"Slurm on K8s"这个趋势的产物，不是收购的产物；收购更像是 NVIDIA 对这个趋势的"下注确认"。

### 收购的逻辑

```
收购前：
  Slurm ← 独立运营，由 SchedMD 维护
  Kubernetes ← CNCF 社区
  NVIDIA ← 卖 GPU + Run:ai（K8s 侧调度/配额）

收购后：
  NVIDIA ← 影响 Slurm 的发展方向
    └→ Slinky：Slurm on Kubernetes（SchedMD 官方方案）
        └→ Slurm 跑在 K8s 上 → K8s 有机会成为统一底座
            └→ GPU 设备插件/DRA、网络配置可以共用一套
```

### Slinky（Slurm on Kubernetes）

Slinky 是 SchedMD 官方的开源项目，2024 年开源，**v1.0.0 于 2025-11-20 发布**：

```
两个核心组件：
  slurm-operator  → 在 K8s 集群内用 CRD（Cluster、NodeSet 等）管理一整套 Slurm：
                     slurmctld / slurmdbd / slurmd 都以 Pod 形式运行
  slurm-bridge    → 反向桥接：把 K8s 工作负载交给 Slurm 控制面做调度决策，
                     让 Slurm 语义（队列、公平分享）作用于 K8s 资源

v1.0.0 的关键能力：
  - hybrid 支持（Slurm 集群可挂接 K8s 之外的节点）
  - slurm-bridge 新增 DRA Extended Resources、TaintToleration、VolumeBinding 插件
  - 支持 Slurm 25.11 的 GRES 细粒度节点资源分配
```

### CoreWeave SUNK

CoreWeave（云 GPU 提供商）**2024 年初**开源了 SUNK：

```
SUNK = Slurm on Kubernetes（SlUrm oN Kubernetes）

特点：
  - Slurm 节点以 K8s Pod 形式运行（NodeSet 定义计算节点，Syncer 同步状态）
  - 同一集群同时跑 Slurm 训练 + K8s 推理
  - 用户通过 sbatch 提交训练任务，推理服务通过 kubectl 管理
  - 自研 SUNK Pod Scheduler 仲裁 Slurm 与 K8s 工作负载对同一资源池的竞争
  - 训练空闲时，GPU 可以让给推理负载
```

---

## 这对生态意味着什么

### 先说该警惕的

收购不等于闭源，但也不等于毫无风险。社区的现实担忧是**厂商中立性**：NVIDIA 同时拥有 Slurm（HPC 侧）和 Run:ai/KAI（K8s 侧），未来 Slurm 是否会优先适配 NVIDIA 硬件、AMD/Gaudi/TPU 生态是否会被边缘化，是 HPC 用户长期观察的问题（收购后 CRN、HPCwire 等媒体报道了这些担忧，也提到了 Flux 等替代方案作为潜在出路）。这是"需要观察的博弈"，不是"已经发生的灾难"。

### 对 Slurm 用户

| 变化 | 影响 |
|------|------|
| Slurm 继续开源 | ✅ NVIDIA 公告明确承诺继续以开源、厂商中立方式开发发布 |
| SchedMD 团队保留 | ✅ 核心开发者继续维护 |
| 新功能优先支持 AI 场景 | ⚠️ 大概率如此（更好的 GPU 拓扑感知、异构集群支持），但受益方首先是 NVIDIA 硬件 |
| Slinky 成为一等公民 | ✅ Slurm on K8s 从第三方拼装变成官方方案 |
| 其他加速器生态 | ⚠️ 需要靠社区与客户合同压力维持中立性 |

### 对 Kubernetes 用户

| 变化 | 影响 |
|------|------|
| K8s 有机会成为 AI 集群的统一底座 | ✅ 训练（Slurm）+ 推理终于能共用资源池 |
| KAI Scheduler 持续开源 | ✅ NVIDIA 没有放弃 K8s 原生方案 |
| GPU 动态共享 | ✅ 训练空闲 → 推理复用（通过 SUNK/Slinky 的调度仲裁） |
| 不需要同时学两套资源管理 | ⚠️ 理论如此；实际仍要理解 Slurm 的队列/公平分享语义 |

### 对 NVIDIA

```
NVIDIA 的战略清晰了：
  "GPU 硬件 → 统一调度层 → 你的 AI 工作负载"
              ↓
         Kubernetes（或裸金属 Slurm）
              ↓
   Slinky（Slurm）| KAI Scheduler | GPU Operator / DRA
              ↓
    你的训练任务 → 你的推理服务 → 你的数据流水线
```

---

## 🛠 实操指南

### 📋 决策指南：收购后你的集群策略该不该变？

```
你当前的集群用了什么调度器？
├─ 裸金属 Slurm →
│   ├─ 不动：Slurm 继续开源维护，现有作业无需迁移
│   ├─ 可考虑：新增推理服务时引入 K8s + Slinky 混合
│   └─ 长期：数据中心扩建时评估 Slinky 方式
│
├─ Kubernetes →
│   ├─ 不动：KAI Scheduler / Volcano / Kueue 继续使用
│   ├─ 可考虑：如果训练调度不满意，试试 Slinky
│   └─ 好消息：GPU 管理链路（GPU Operator/DRA）集成会更深
│
└─ 正在搭建 →
    └─ 推荐方向：Kubernetes + Slinky/KAI Scheduler
        ├─ 训练团队习惯 sbatch → Slinky 提供
        ├─ 推理团队习惯 K8s → 原生提供
        └─ GPU 管理 → GPU Operator / DRA
```

### 🔧 自己动手验：确认你的集群里有没有 Slurm-on-K8s

```bash
# 1. 检查是否部署了 Slinky（slurm-operator）
kubectl get pods -n slurm-system 2>/dev/null || echo "没有检测到 slurm-operator"
kubectl get clusters.slinky.schedmd.com 2>/dev/null

# 2. 检查是否部署了 CoreWeave SUNK
kubectl get slurmclusters 2>/dev/null

# 3. 确认本机 Slurm 版本与维护方
sinfo --version
scontrol show config | grep -i -E "SLURM_VERSION|SLURMCTLD"

# 4. 观察 GPU 利用率（训练 + 推理混部场景）
nvidia-smi dmon -s puct -d 30
# 训练空闲时段利用率仍不归零 → 可能有推理负载在复用 GPU
```

### ⚠️ 常见误区

- **误区**："NVIDIA 收购 SchedMD 后 Slurm 会闭源"
  - **实际**：Slurm 仍然是 GPL 系开源项目，NVIDIA 在收购公告中明确承诺**继续开源、厂商中立**。Slinky 也是开源的（SchedMD 维护）。真正的风险不在许可证，而在**发展方向的偏向**——这要靠社区持续监督。

- **误区**："收购后所有集群都必须用 K8s"
  - **实际**：没有"必须"。Slurm 仍然可以直接跑在裸金属上——这是绝大多数超算的形态。Slinky 只是多了"跑在 K8s 上"的选项——**不是替代，是补充**。

- **误区**："Slinky/SUNK 是收购之后才出现的新东西"
  - **实际**：Slinky 2024 年开源（2025-11-20 才到 v1.0.0），SUNK 2024 年初就开源了。**收购是把已有趋势收编并加速**，不是发明了这个趋势。

- **误区**："收购宣布的那一刻，Slurm 的功能就大变了"
  - **实际**：Slurm 的发布节奏、支持流程在短期内不会变。监管审批、团队整合都需要时间，观察窗口应以年计。

### 📊 底线数字速查

| 指标 | 裸金属 Slurm | 纯 K8s | Slurm on K8s（Slinky/SUNK）|
|------|-------------|--------|----------------------|
| 训练性能 | baseline | 接近 baseline（需正确配置网络/拓扑） | 接近裸金属（Pod 化开销为主） |
| 推理性能 | N/A | baseline | baseline |
| GPU 资源池 | 训练/推理通常硬隔离 | 服务为主 | 训练 + 推理可共享同一资源池 |
| 运维复杂度 | 1 套系统 | 1 套系统 | K8s 底座 + Slurm 语义 |
| 用户学习成本 | 低（HPC 团队） | 低（DevOps 团队） | 各用各的接口（sbatch / kubectl）|
| 部署成熟度 | 20+ 年验证 | 数年验证 | 2024 开源，2025-11 v1.0 |

### 💡 一句话记住

> **"NVIDIA 收购 SchedMD，不是要'杀'掉 Slurm，也不是发明了 Slurm on K8s——Slinky 和 SUNK 早在收购前就已开源。收购是用官方资源把'Slurm 跑在 K8s 上'这条路铺平。Slurm 的开源身份没有变，需要盯住的是方向，而不是许可证。"**

---

## 对后续的铺垫

有了调度器，有了 Slurm on K8s，但还有一个关键问题：**分布式训练要求"所有 GPU 同时到位"才能启动**。

这就是 Gang Scheduling（集合调度）要解决的问题。

→ [[05-Gang Scheduling]]

---

> **参考资料**：
> - [NVIDIA: "NVIDIA Acquires SchedMD"（2025-12-15 官方公告）](https://blogs.nvidia.com/blog/nvidia-acquires-schedmd/)
> - [HPCwire: "What Does Nvidia's Acquisition of SchedMD Mean for Slurm?" (2026-01-06)](https://www.hpcwire.com/2026/01/06/what-does-nvidias-acquisition-of-schedmd-mean-for-slurm/)
> - [CRN: "Nvidia Bought Slurm's Creator: It Makes Sense To One Partner, Another Is Concerned" (2025)](https://www.crn.com/news/software/2025/nvidia-bought-slurm-s-creator-it-makes-sense-to-one-partner-another-is-concerned)
> - [Slinky 项目官网](https://slinky.ai) / [slurm-operator（GitHub）](https://github.com/SlinkyProject/slurm-operator)
> - [slurm-announce 邮件列表：Slinky v1.0.0 发布（2025-11-20）](https://lists.schedmd.com/mailman3/hyperkitty/list/slurm-announce@lists.schedmd.com/message/EADOMYFCUGV6EDKVQBTFHQE3QTO5QDLS/)
> - [CoreWeave SUNK 官方文档](https://docs.coreweave.com/products/sunk)
