# Q: Kubernetes 为什么在 AI 场景崛起？

## 为什么会有这个问题

Slurm 统治了 HPC 20 年。但在 2022-2026 年间，越来越多的 AI 团队转向 Kubernetes——OpenAI、Meta 等公司的 AI 基础设施大量使用 K8s（Google 内部大规模使用的是自家的 Borg，Kubernetes/GKE 是它对外输出的产品）。

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
| **自愈** | 进程崩溃、节点故障后自动重新调度 | Pod 自动重建 |
| **多租户** | 多个团队共享集群，互不干扰 | Namespace + RBAC |
| **统一平台** | 训练 + 推理 + 数据流水线都在同一平台 | 一套 YAML 走天下 |

### 自愈 ≠ 训练自愈：一个必须记住的前提

Kubernetes 的"自愈"恢复的是 **Pod 进程**，不是**训练状态**。节点挂掉后，新调度的 Pod 是一个全新的容器——CUDA 上下文、优化器状态、DataLoader 位置全部丢失，训练从头开始。

**所以 K8s 的自愈要对训练生效，必须配合：**

1. **Checkpoint 恢复**：训练任务启动时自动检测最新 checkpoint，从那里续训（`torch.distributed.checkpoint`、DeepSpeed 的 resume 机制）
2. **弹性/容错训练**：TorchElastic（`torchrun` 的 `--max-restarts`）允许 worker 变动后重新集合；配合作业框架（JobSet、KAI）的自动重跑
3. **健康检查与止损**：把"卡死的节点"及时踢出重调度，而不是让 NCCL 超时等待

一句话：**K8s 负责"把 Pod 再拉起来"，训练框架负责"从断点继续"。两者缺一不可。**

### Meta 的实践：Kubernetes on Bare Metal

Meta 的 GPU 集群是 Kubernetes 在 AI 场景的标杆案例（Meta 在公开分享中多次介绍）：

```
Grand Teton 平台：
  - 24,576 GPU 集群（一套用于 Llama 3 训练）
  - 直接在裸金属上跑 Kubernetes（不是虚拟化）
  - PyTorch + Kubernetes + 自定义调度器
  - 统一的训练 + 推理平台
  
关键决策：
  为什么不用 Slurm？
  - Meta 内部基础设施大量基于 K8s
  - 统一平台降低运维复杂度
  - 训练作业用自定义调度器 + TorchElastic 补齐批处理能力
```

---

## Kubernetes 在 AI 场景的三大挑战

### 挑战 1：多机训练的编排单位

首先要纠正一个常见误解：**Pod 是 Kubernetes 的最小调度单元，一个 Pod 永远不能跨节点**。所谓"多机训练"，是编排 **多个 Pod**（通常一个节点一个 Pod），让它们组成一个作业一起调度、一起生灭。

**解决方案（2026）**：

| 方案 | 来源 | 原理 |
|------|------|------|
| **JobSet** | Kubernetes SIG（jobset.sigs.k8s.io） | 把多个 Job 组成一个 JobSet，支持全部 Job 就绪才启动、失败全部重跑 |
| **MPI Operator** | Kubeflow | 把 MPI 风格的多 Pod 作业抽象成 MPIJob CRD |
| **Volcano** | CNCF 孵化项目 | PodGroup + gang scheduling |
| **KAI Scheduler** | NVIDIA（2025 开源，基于 Run:ai 技术） | 分层队列 + gang + 弹性 |
| **Kueue** | kubernetes-sigs 批处理排队项目 | 基于 Workload 的配额与排队管理 |
| **Ray** | Anyscale | RayCluster CRD，Head + Worker Pod 拓扑 |

### 挑战 2：拓扑感知

Kubernetes 默认不知道什么是 NVLink、NVSwitch。分配 GPU 时可能：
- 把 8 卡任务分到 2 台不同节点的 4 张卡上（而不是同一台 8 卡机器）
- 把 4 个需要 NVLink 通信的 GPU 分到不同 NUMA 域上

**解决方案**（示意：用 GPU Feature Discovery 打的节点标签做亲和）：

```yaml
# NVIDIA GPU Operator 会部署 GPU Feature Discovery（GFD），
# 在有 NVLink 域（MNNVL）的节点上打上 nvidia.com/gpu.clique=<ClusterUUID>.<CliqueID> 标签
# （示例字段，是否生成取决于 GPU Operator/GFD 配置与硬件；非 MNNVL 硬件不会打这个标签）
apiVersion: v1
kind: Pod
metadata:
  name: training-pod
spec:
  affinity:
    podAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
      - labelSelector:
          matchExpressions:
          - key: job-name
            operator: In
            values: ["training-job"]
        topologyKey: nvidia.com/gpu.clique   # 让同一作业的 Pod 落在同一个 NVLink 域内
  containers:
  - name: trainer
    image: pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime
    resources:
      limits:
        nvidia.com/gpu: 8
  schedulerName: kai-scheduler   # 拓扑感知调度器（示例：NVIDIA KAI Scheduler）
```

此外，新一代的方案是 **Kubernetes DRA（Dynamic Resource Allocation）**：Kubernetes 1.34 起 DRA 核心进入 GA（`resource.k8s.io/v1`），NVIDIA 的 DRA 驱动可以基于拓扑约束（如 NVLink 域、ComputeDomain）分配 GPU，这是 device plugin 之后的演进方向。

### 挑战 3：网络性能

K8s 的容器网络（CNI）默认会引入额外的网络跳转，对需要 RDMA/InfiniBand 通信的分布式训练来说，这是不可接受的。

**常见方案与坑**：

```yaml
# 方案 A：hostNetwork —— 简单粗暴，但有代价
spec:
  hostNetwork: true  # Pod 直接共享宿主机网络命名空间
  containers:
  - name: trainer
    ports:
    - containerPort: 29500  # NCCL 通信端口
```

**注意：`hostNetwork: true` 和 Multus 多网卡方案互斥**。hostNetwork 下 Pod 共享主机的网络命名空间，Multus 附加的第二块 RDMA 网卡接口不再有意义，也无法给不同 Pod 分配独立的 RDMA 设备。生产环境更推荐：

| 方案 | 说明 |
|------|------|
| **RDMA shared device plugin** | 把 RDMA 设备以 Kubernetes 资源形式暴露（`rdma/hca: 1`），Pod 按需申请，多 Pod 可共享/独占 HCA |
| **SR-IOV Network Device Plugin + Multus** | 把物理网卡 SR-IOV 切分 VF，通过 Multus 把 VF 附加到 Pod（独立网络命名空间，不影响主机栈） |
| **Macvlan / IPvlan CNI** | 让 Pod 直接拿到子网内的真实 IP，延迟接近宿主机，适合 RoCE |

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

# 2. 提交一个 PyTorch 训练任务（单节点示例）
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: pytorch-train
spec:
  restartPolicy: Never
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
EOF

# 3. 查看训练状态
kubectl logs -f pytorch-train
```

> 生产环境请用 JobSet / MPI Operator / Ray 等编排多 Pod 作业，并给训练 Pod 挂上 RDMA 设备资源（rdma shared device plugin 或 SR-IOV），而不是依赖 hostNetwork。

### ⚠️ 常见误区

- **误区**："Kubernetes 不适合大规模训练，性能损失太大"
  - **实际**：**如果配置正确（RDMA 设备直通 + GPU Operator + 拓扑感知调度），Kubernetes 的网络和 GPU 开销可以做到接近裸金属**。Meta 等公司在裸金属 K8s 上跑大规模训练的公开分享就是例证。

- **误区**："Kubernetes 的自愈能让训练不中断"
  - **实际**：自愈只是把 Pod 重新拉起来，**训练状态必须靠 checkpoint 恢复**。没有 checkpoint 的训练作业，Pod 自愈等于从头重跑。

- **误区**："一个 Pod 可以跨多个节点跑"
  - **实际**：Pod 是最小调度单元，**永远绑定在一个节点上**。多机训练必须编排多个 Pod（JobSet / MPI Operator / Volcano / Ray）。

- **误区**："有 GPU Operator 就万事大吉了"
  - **实际**：GPU Operator（device plugin 模式）只解决"让 K8s 知道有 GPU"的问题（MIG 切分、time-slicing 配置也在这层）。**拓扑感知、Gang Scheduling、RDMA 网络**都需要额外配置；更细粒度的设备分配要走 DRA。

### 📊 底线数字速查

| 能力 | Kubernetes | Slurm | 胜者 |
|------|-----------|-------|------|
| 批处理调度 | ⚠️ 需插件 | ✅ 原生 | Slurm |
| 推理服务 | ✅ 原生 | ❌ | K8s |
| 弹性伸缩 | ✅ 原生 | ❌ | K8s |
| 自愈 | ✅ 进程级（训练需 checkpoint 配合） | ❌ | K8s |
| 已验证集群规模 | 数千节点 | 65,536 节点 | Slurm |
| 训练启动速度 | 秒到分钟级（含镜像拉取） | 秒级 | Slurm |
| 统一平台 | ✅ 端到端 | ❌ 纯调度 | K8s |
| 学习曲线（AI 团队）| 陡 | 平 | Slurm |

### 💡 一句话记住

> **"Kubernetes 进入 AI 领域不是因为它在训练调度上比 Slurm 强，而是因为 AI 已经从'提交作业'变成了'端到端流水线'——Kubernetes 能统一管理训练、推理、监控、A/B 测试，而 Slurm 只能管训练这一个环节。"**

---

## 对后续的铺垫

但 Kubernetes 和 Slurm 不是"二选一"——2024 年以来的最大趋势是两者结合。

NVIDIA 收购 SchedMD（Slurm 的维护公司）后，**SchedMD 的 Slinky（Slurm on Kubernetes）**——把 Slurm 作为 K8s 的工作负载运行——进入了 1.0。这就是下一节的内容。

→ [[04-NVIDIA收购SchedMD意味着什么]]

---

> **参考资料**：
> - [Kubernetes 官方文档：Device Plugins](https://kubernetes.io/docs/concepts/extend-kubernetes/compute-storage-net/device-plugins/)
> - [NVIDIA DRA Driver for GPUs：ComputeDomain Workloads](https://dra-driver-nvidia-gpu.sigs.k8s.io/docs/guides/compute-domain-workloads/)
> - [NVIDIA GPU Operator 文档](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/)
> - [Kueue Overview（kubernetes-sigs）](https://kueue.sigs.k8s.io/docs/overview/)
> - [JobSet 文档（Kubernetes SIG）](https://jobset.sigs.k8s.io/docs/overview/)
> - [Kubeflow MPI Operator](https://github.com/kubeflow/mpioperator)
> - [RDMA shared device plugin](https://github.com/k8snetworkplumbingwg/rdma-shared-device-plugin)
> - [SR-IOV Network Device Plugin](https://github.com/k8snetworkplumbingwg/sriov-network-device-plugin)
> - [NVIDIA KAI Scheduler（GitHub）](https://github.com/NVIDIA/KAI-Scheduler)
