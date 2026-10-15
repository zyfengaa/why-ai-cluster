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
| **K8s 原生 gang scheduling（KEP-4671）** | Kubernetes 社区 | 内置于核心调度器；v1.37 达 beta 但**默认关闭**，需显式开启 |
| **Volcano** | 华为开源 | 支持 PodGroup 和 Gang Scheduling |
| **KAI Scheduler** | NVIDIA（源自 Run:ai 技术） | Kubernetes 原生，队列与拓扑感知 |
| **Kueue** | Google 开源 | 面向批处理的排队和配额管理，与原生调度器配合 |

> 截至 2026 年，原生 gang scheduling 虽然已进核心并达到 beta，但默认关闭；**生产上的推荐架构仍是 Kueue 负责排队与配额 + Volcano/coscheduling（或 KAI Scheduler）负责 Pod 组的 all-or-nothing 调度**。

> 背景：**NVIDIA 已于 2024-12-30 以约 7 亿美元完成对 Run:ai 的收购**——KAI Scheduler 正是源自 Run:ai 的调度技术，这也是理解它"为什么是 K8s 原生风格"的关键前情。

### 挑战 2：拓扑感知

Kubernetes 默认不知道什么是 NVLink、NVSwitch。分配 GPU 时可能：
- 把 8 卡任务分到 2 台不同节点的 4 张卡上（而不是同一台 8 卡机器）
- 把 4 个需要 NVLink 通信的 GPU 分到不同 PCIe 根上

**解决方案**：GPU 的拓扑信息由 **NVIDIA GPU Feature Discovery（GFD）** 采集并写成**节点标签**（如 `nvidia.com/gpu.product`；多节点 NVLink 场景下还有 `nvidia.com/gpu.clique`），调度侧则通过 **topologyKey 亲和约束**或 **DRA（Dynamic Resource Allocation）/ ComputeDomain** 来表达"这些 GPU 必须落在同一 NVLink 域"。

```yaml
# 通过节点亲和 + GFD 写入的拓扑标签约束
apiVersion: v1
kind: Pod
metadata:
  name: training-pod
spec:
  containers:
  - name: trainer
    image: pytorch:24.03
    resources:
      limits:
        nvidia.com/gpu: 8
  affinity:
    # 用 GFD 写入的拓扑标签，把需要 NVLink 通信的 Pod 约束到同一 NVLink 域
    podAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
      - topologyKey: nvidia.com/gpu.clique   # GFD 写入的 NVLink 域标签
        labelSelector:
          matchLabels:
            app: trainer
  schedulerName: kai-scheduler  # 拓扑感知调度器
```

> 注意：像 `nvidia.com/gpu-request-topology` 这样的**注解并不存在**——GPU 拓扑不是靠 Pod 注解声明，而是靠 GFD 写入节点标签、再在调度侧用 topologyKey 或 DRA 表达。

补充一句：**Dynamic Resource Allocation（DRA）已在 Kubernetes 1.34 正式 GA（2025-09）**，成为 GPU 等加速器资源分配的现代机制。它用 `ResourceClaim` 描述"我要什么样的设备"，让调度器能按设备属性（型号、MIG 配置、NVLink 域）而不是只靠整数计数来分配 GPU；NVIDIA 也在 DRA 之上提供了 ComputeDomain（把多节点 NVLink 域暴露给作业）等能力。

### 挑战 3：网络性能

K8s 的默认容器网络（CNI）走 overlay/IP，对需要 RDMA/InfiniBand 通信的分布式训练来说，性能和语义都不合适。**但解法不是 `hostNetwork: true`**——把 Pod 塞进宿主机网络栈只能拿到普通 TCP/IP，**并不能换来 RDMA/NCCL over IB**，反而会破坏 CNI 的网络隔离与多租户。

**解决方案**：用 **Multus + SR-IOV / IB Device Plugin / RDMA shared device plugin**（GPU Operator 已内置），把 IB 网卡作为 **secondary network 直通**给 Pod：

```yaml
# 通过 Multus 把 IB 网卡作为 secondary network 挂给 Pod
apiVersion: v1
kind: Pod
metadata:
  name: training-pod
  annotations:
    k8s.v1.cni.cncf.io/networks: ib-net   # 引用 NetworkAttachmentDefinition
spec:
  containers:
  - name: trainer
    resources:
      limits:
        nvidia.com/gpu: 8
        rdma/rdma_shared_device_a: 1        # 通过 RDMA shared device plugin 请求 RDMA 设备
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
  restartPolicy: Never
EOF

# 3. 查看训练状态
kubectl logs -f pytorch-train
```

### ⚠️ 常见误区

- **误区**："Kubernetes 不适合大规模训练，性能损失太大"
  - **实际**：**如果配置正确（Multus/SR-IOV + GPU Operator + Topology Manager），Kubernetes 的网络和 GPU 性能损失可以控制在 1% 以内**。Meta 的 24K GPU 集群跑在 K8s 上就是最好的证明。

- **误区**："Kubernetes 的容器化对训练来说太重了"
  - **实际**：容器化恰恰是训练团队最需要的——**环境一致性和可复现性**。2026 年，Slurm 集群也在用 Pyxis/Enroot 容器。

- **误区**："有 GPU Operator 就万事大吉了"
  - **实际**：GPU Operator 只解决"让 K8s 知道有 GPU"的问题。**拓扑感知、Gang Scheduling、RDMA 网络**都需要额外配置。

### 📊 底线数字速查

| 能力 | Kubernetes | Slurm | 胜者 |
|------|-----------|-------|------|
| 批处理调度 | ⚠️ 需插件 | ✅ 原生 | Slurm |
| 推理服务 | ✅ 原生 | ❌ | K8s |
| 弹性伸缩 | ✅ 原生 | ⚠️ 需 `--requeue` + checkpoint | K8s |
| 自愈 | ✅ 原生 | ⚠️ 需显式 `--requeue`，无声明式自愈 | K8s |
| 最大集群 | ~5,000 节点 | 10 万+ 节点 | Slurm |
| 训练启动速度 | ~5-30 s | ~1-2 s | Slurm |
| 统一平台 | ✅ 端到端 | ❌ 纯调度 | K8s |
| 学习曲线（AI 团队）| 陡 | 平 | Slurm |

### 💡 一句话记住

> **"Kubernetes 进入 AI 领域不是因为它在训练调度上比 Slurm 强，而是因为 AI 已经从'提交作业'变成了'端到端流水线'——Kubernetes 能统一管理训练、推理、监控、A/B 测试，而 Slurm 只能管训练这一个环节。"**

---

## 对后续的铺垫

但 Kubernetes 和 Slurm 不是"二选一"——2025-2026 年的最大趋势是两者结合。

**Slinky（Slurm on Kubernetes）** 是 **SchedMD 自己的开源项目**——1.0.0 于 **2025-11-20** 发布、KubeCon Europe 2025（4 月）已有演讲，**早于 NVIDIA 在 2025-12-15 宣布收购 SchedMD**。它把 Slurm 作为 K8s 的工作负载运行。（注意措辞：Slurm 是项目，SchedMD 是其主要的开发公司。）这就是下一节的内容。

→ [[04-NVIDIA收购SchedMD意味着什么]]

---

> **参考资料**：
> - ["What Does NVIDIA's Acquisition of SchedMD Mean for Slurm?" — HPCwire (2026)](https://www.hpcwire.com/2026/01/06/what-does-nvidias-acquisition-of-schedmd-mean-for-slurm/)
> - [Volcano PodGroup 文档](https://volcano.sh/docs/concepts/podgroup/)
> - [Kueue 文档](https://kueue.sigs.k8s.io/)
> - [Kubernetes v1.34：DRA 正式 GA](https://kubernetes.io/blog/2025/09/01/kubernetes-v1-34-dra-updates/)
> - [Kubernetes 大规模集群最佳实践（cluster-large）](https://kubernetes.io/docs/setup/best-practices/cluster-large/)
> - [NVIDIA GPU Operator 文档](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/)
> - [KAI Scheduler 文档](https://github.com/kai-scheduler/KAI-Scheduler)
