# Q: Kubernetes 为什么在 AI 场景崛起？

## 为什么会有这个问题

Slurm 统治了 HPC 20 年。但在 2022-2026 年间，越来越多的 AI 团队转向 Kubernetes——OpenAI、Meta、Google 的 AI 基础设施都在 K8s 上运行。

一个为"微服务"设计的容器调度器，为什么能"入侵"AI 领域？

## 解决什么问题

Kubernetes 进入 AI 场景，不是因为它在"批处理"上比 Slurm 强，而是因为 **AI 训练不再是纯批处理——它变成了一个端到端的系统**。

### 从"提交作业"到"AI 流水线"

2020 年的 AI 训练：
```bash
# 就三步——简单直接
数据准备 → sbatch train.slurm → 等待输出
```

2026 年的 AI 流水线：
```
数据流水线 → 数据增强 → 分布式训练 → Checkpoint → 
模型评估 → A/B 测试 → 推理部署 → 在线监控 → 自动回滚
```

**这个"端到端"的流程里，Kubernetes 有天然优势。**

### Kubernetes 在 AI 场景的核心能力

| 能力 | 为什么对 AI 重要 | Kubernetes 做法 |
|------|----------------|-----------------|
| **容器化** | 环境一致性：开发环境 = 生产环境 | Docker + Pod |
| **弹性伸缩** | 推理服务根据流量自动扩缩 | HPA / VPA |
| **自愈** | 节点挂了自动重启，训练不会丢失 | Pod 自动调度 |
| **多租户** | 多个团队共享集群，互不干扰 | Namespace + RBAC |
| **统一平台** | 训练 + 推理 + 数据流水线都在同一平台 | 一套 YAML 走天下 |

### Meta 的实践：Kubernetes on Bare Metal

Meta 的 GPU 集群是 Kubernetes 在 AI 场景的标杆案例：

```
Grand Teton 平台：
  - 24,576 GPU 集群，用于 Llama 3 训练
  - 直接在裸金属上跑 Kubernetes（不是虚拟化）
  - PyTorch + Kubernetes + 自定义调度器
  - 统一的训练 + 推理平台
  
关键决策：
  为什么不用 Slurm？
  - Meta 内部所有基础设施都基于 K8s
  - 训练只是"一个 Pod 的长时间版本"
  - 统一平台降低运维复杂度
```

---

## Kubernetes 在 AI 场景的三大挑战

### 挑战 1：Gang Scheduling（集合调度）

Kubernetes 原生调度是"逐个 Pod"进行的。对于一个需要 64 卡的分布式训练任务，如果 63 个 Pod 启动了，第 64 个没资源——那 63 个 Pod 都只能在空转。

**解决方案（2026）**：

| 方案 | 来源 | 原理 |
|------|------|------|
| **Volcano** | 华为开源 | 支持 PodGroup 和 Gang Scheduling |
| **KAI Scheduler** | NVIDIA（2025 开源） | 基于 Run:ai 技术，Kubernetes 原生 |
| **Kueue** | Google 开源 | 面向批处理的排队和配额管理 |

### 挑战 2：拓扑感知

Kubernetes 默认不知道什么是 NVLink、NVSwitch。分配 GPU 时可能：
- 把 8 卡任务分到 2 台不同节点的 4 张卡上（而不是同一台 8 卡机器）
- 把 4 个需要 NVLink 通信的 GPU 分到不同 PCIe 根上

**解决方案**：
```yaml
# NVIDIA GPU Operator + Topology Manager
apiVersion: v1
kind: Pod
metadata:
  name: training-pod
  annotations:
    nvidia.com/gpu-request-topology: "nvlink"
spec:
  containers:
  - name: trainer
    image: pytorch:24.03
    resources:
      limits:
        nvidia.com/gpu: 8
  schedulerName: kai-scheduler  # 拓扑感知调度器
```

### 挑战 3：网络性能

K8s 的容器网络（CNI）默认会引入额外的网络跳转，对需要 RDMA/InfiniBand 通信的分布式训练来说，这是不可接受的。

**解决方案**：
```yaml
# 使用 hostNetwork 绕过 CNI
spec:
  hostNetwork: true  # 直接使用宿主机网络栈
  containers:
  - name: trainer
    ports:
    - containerPort: 29500  # NCCL 通信端口
```

---

## 🛠 实操指南

### 📋 决策指南：什么时候选 Kubernetes？

```
你的工作负载类型？
├─ 纯大规模训练（> 512 卡，单任务跑 > 1 周）→ Slurm 可能更合适
│
├─ 训练 + 推理混合 →
│   └─ 统一平台 > 分离管理 → Kubernetes
│
├─ 推理为主 →
│   └─ Kubernetes（弹性 + 自愈天然优势）
│
└─ 你的团队背景是 DevOps/云原生 →
    └─ Kubernetes（在团队熟悉的工具上扩展 AI 能力）
```

### 🔧 自己动手验：用 Kubernetes 跑一个训练任务

```bash
# 1. 检查集群是否支持 GPU
kubectl get nodes -o custom-columns=:.metadata.name,:.status.allocatable.nvidia\\.com/gpu

# 2. 提交一个 PyTorch 训练任务
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: pytorch-train
  annotations:
    k8s.v1.cni.cncf.io/networks: ib-net  # InfiniBand 网络
spec:
  containers:
  - name: trainer
    image: pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel
    command: ["torchrun", "--nproc_per_node=2", "train.py"]
    resources:
      limits:
        nvidia.com/gpu: 2
    env:
    - name: NCCL_IB_DISABLE
      value: "0"
    - name: NCCL_SOCKET_IFNAME
      value: "eth0"
  hostNetwork: true  # 直接使用宿主机网络
  restartPolicy: Never
EOF

# 3. 查看训练状态
kubectl logs -f pytorch-train
```

### ⚠️ 常见误区

- **误区**："Kubernetes 不适合大规模训练，性能损失太大"
  - **实际**：**如果配置正确（hostNetwork + GPU Operator + Topology Manager），Kubernetes 的网络和 GPU 性能损失可以控制在 1% 以内**。Meta 的 24K GPU 集群跑在 K8s 上就是最好的证明。

- **误区**："Kubernetes 的容器化对训练来说太重了"
  - **实际**：容器化恰恰是训练团队最需要的——**环境一致性和可复现性**。2026 年，Slurm 集群也在用 Pyxis/Enroot 容器。

- **误区**："有 GPU Operator 就万事大吉了"
  - **实际**：GPU Operator 只解决"让 K8s 知道有 GPU"的问题。**拓扑感知、Gang Scheduling、RDMA 网络**都需要额外配置。

### 📊 底线数字速查

| 能力 | Kubernetes | Slurm | 胜者 |
|------|-----------|-------|------|
| 批处理调度 | ⚠️ 需插件 | ✅ 原生 | Slurm |
| 推理服务 | ✅ 原生 | ❌ | K8s |
| 弹性伸缩 | ✅ 原生 | ❌ | K8s |
| 自愈 | ✅ 原生 | ❌ | K8s |
| 最大集群 | ~5,000 节点 | ~30,000 节点 | Slurm |
| 训练启动速度 | ~5-30 s | ~1-2 s | Slurm |
| 统一平台 | ✅ 端到端 | ❌ 纯调度 | K8s |
| 学习曲线（AI 团队）| 陡 | 平 | Slurm |

### 💡 一句话记住

> **"Kubernetes 进入 AI 领域不是因为它在训练调度上比 Slurm 强，而是因为 AI 已经从'提交作业'变成了'端到端流水线'——Kubernetes 能统一管理训练、推理、监控、A/B 测试，而 Slurm 只能管训练这一个环节。"**

---

## 对后续的铺垫

但 Kubernetes 和 Slurm 不是"二选一"——2025-2026 年的最大趋势是两者结合。

NVIDIA 收购 SchedMD（Slurm 的母公司）后，开源了 **Slinky（Slurm on Kubernetes）**——把 Slurm 作为 K8s 的工作负载运行。这就是下一节的内容。

→ [[04-NVIDIA收购SchedMD意味着什么]]

---

> **参考资料**：
> - "Kubernetes vs SLURM for AI Workloads" — Shakti Cloud (2026)
> - "Why the top AI labs run Kubernetes on bare metal" — Spectro Cloud (2026)
> - "Slurm vs. Kubernetes in the Age of AI" — HPCwire (2026)
> - NVIDIA GPU Operator 文档 (2026)
> - KAI Scheduler 文档 (2026)
