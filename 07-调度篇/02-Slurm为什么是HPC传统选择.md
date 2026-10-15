# Q: Slurm 为什么是 HPC 的传统选择？

## 为什么会有这个问题

当你走进任何一个国家级超算中心——太湖之光、Frontier、Perlmutter——问他们用什么调度器，答案几乎都是 **Slurm**。它从 2002 年诞生，已经统治 HPC 调度 20 多年。

在 AI 浪潮中，Slurm 被"挖"出来管理 GPU 集群。不是因为它最时尚，而是因为 **大规模分布式训练的需求和 HPC 批处理作业几乎一模一样**。

## 解决什么问题

Slurm（Simple Linux Utility for Resource Management）解决的核心问题：

1. **大规模批处理调度**：在几万个节点上调度数百万个 CPU 小时的作业
2. **资源隔离与公平分享**：多用户、多项目共享集群时保证公平
3. **作业依赖与流水线**：作业 A 完成了才启动作业 B
4. **拓扑感知分配**：知道哪些 GPU/节点在同一个网络域内

### Slurm 的设计哲学

```
"资源到位前不执行，资源释放前不让道"
```

这背后是 Slurm 和 Kubernetes 最本质的区别，但**并不是"预约式 vs 抢占式"**——Slurm 的 reservation 是需手动 `scontrol create reservation` 的**可选功能**，默认同样是"资源可用即调度 + backfill"；Slurm 也支持抢占（`PreemptType`/`PreemptMode`），而 K8s 原生的 Pod 抢占语义其实很弱。真正的区别有两点：**默认分配的原子性**（Slurm 对多节点作业天然 all-or-nothing，K8s 原生则逐个 Pod 调度）与**负载模型**（批处理队列 vs 声明式期望状态）。

### Slurm 的核心组件

| 组件 | 功能 | 类比 |
|------|------|------|
| **slurmctld** | 中央控制器，负责调度决策 | 大脑 |
| **slurmd** | 每个节点上的守护进程，执行任务 | 手脚 |
| **slurmdbd** | 数据库守护进程，记录作业历史 | 记账员 |
| **sacctmgr** | 账户管理，配置 QOS 和公平分享 | HR 部门 |

### 用户最常用的命令

```bash
# 提交作业
sbatch --gpus=8 --ntasks=8 --nodes=1 train.slurm

# 交互式作业（调试用）
srun --gpus=8 --ntasks=8 --nodes=1 --pty bash

# 查看队列
squeue -u $USER

# 查看节点状态
sinfo -o "%n %G %T %c %m"

# 取消作业
scancel <jobid>

# 查看作业详情
scontrol show job <jobid>
```

### Slurm 在 AI 场景中的天然优势

1. **Gang Scheduling（集合调度）**：Slurm 天然支持"全部资源到位才启动"——这对分布式训练至关重要。你不会希望 512 卡的任务只拿到 500 张卡就启动。

2. **MPI 原生集成**：`srun` 天然支持 MPI 任务，NCCL 的通信初始化直接兼容 Slurm 的节点分配。

3. **拓扑感知**：`topology.conf` 可以配置 NVSwitch 域、InfiniBand 交换机层级，Slurm 分配时尽量把任务放在拓扑最优的节点上。

4. **万卡级生产验证**：Frontier（37,888 GPU）、Summit（27,648 GPU）、LUMI 等超算全部用 Slurm。

```bash
# Slurm 拓扑配置示例（topology.conf）
# 定义交换机层级
SwitchName=leaf01 Nodes=node[001-016]
SwitchName=leaf02 Nodes=node[017-032]
SwitchName=spine01 Switches=leaf01,leaf02

# Slurm 会尽量把任务分配到同一 leaf 交换机下的节点
```

---

## Slurm 的局限

| 局限 | 表现 | 影响 |
|------|------|------|
| **无容器原生支持** | 需借助 Pyxis/Enroot 或 Singularity | 开发和部署环境不一致 |
| **自愈能力弱于 K8s** | 无 Pod 级控制器；节点故障需显式启用 `--requeue` + 应用侧 checkpoint 才能自动重跑 | 无 K8s 式的声明式自愈 |
| **弹性弱于 K8s** | 任务启动后资源固定；弹性要靠 `--requeue` + checkpoint 重跑实现 | 无 HPA 式动态扩缩 |
| **无原生推理服务** | Slurm 不是为长运行服务设计的 | 推理服务需要额外方案 |
| **API 不够 RESTful** | 官方 API 不够云原生 | 和 CI/CD 集成困难 |

---

## 🛠 实操指南

### 📋 决策指南：什么时候选 Slurm？

```
你团队的技术背景？
├─ HPC/超算出身 → 闭眼选 Slurm，学习成本为零
├─ DevOps/云原生出身 →
│   └─ 训练为主，少量推理 → Slurm（简单可靠）
│       └─ 推理为主，少量训练 → Kubernetes（更合适）
└─ 混合团队 →
    └─ Slurm + Kubernetes 混合方案（Slinky/SUNK）
```

```bash
# 一个标准的 Slurm 训练作业脚本
#!/bin/bash
#SBATCH --job-name=train-llama
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --mem=512G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu-a100
#SBATCH --exclusive

# 加载环境
module load cuda/12.4
module load nccl/2.21
# 非交互 batch 脚本里 conda activate 前必须先初始化 shell hook，否则会报 "conda: command not found"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llama-training

# NCCL 调优
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0,mlx5_1
export NCCL_NET_GDR_LEVEL=5

# 启动训练
torchrun \
  --nproc_per_node=8 \
  --nnodes=$SLURM_NNODES \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1):29500 \
  train.py \
  --model-size 70b \
  --batch-size 4
```

### 🔧 自己动手验：检查 Slurm 集群的健康状态

```bash
# 1. 查看集群总资源
sinfo -o "%P %D %t %c %m %G"
# 输出示例：
# PARTITION NODES STATE CPUS MEMORY GPUS
# gpu-a100* 16 idle 128 1024G 8
# gpu-h100 8 mixed 128 2048G 8

# 2. 查看排队情况
squeue -o "%.18i %.9P %.20j %.8u %.8T %.10M %.9l %.6D %.20R" -S i

# 3. 查看 GPU 分配分布
scontrol show node | grep -E "NodeName|Gres|AllocTRES"
# 能看到每台节点的 GPU 分配情况

# 4. 检查作业为什么排队
scontrol show job <jobid> | grep -E "Reason|Priority|State|Nodes"
# Reason=Resources → 等资源
# Reason=Priority → 优先级低，等更高优先级的跑完
```

### ⚠️ 常见误区

- **误区**："Slurm 太老了，不适合 AI"
  - **实际**：Frontier（37,888 张 AMD MI250X GPU）用 Slurm 管理，CoreWeave、Oracle OCI 等大规模 GPU 云也在用 Slurm 调度训练任务（可参见 SchedMD 官方客户列表）。**老 ≠ 不适合**，Slurm 的大规模调度能力在 AI 领域同样关键。

- **误区**："Slurm 不支持容器"
  - **实际**：Slurm + Pyxis/Enroot 是 NVIDIA 推荐的集群容器方案，性能无损。2026 年 Pyxis 已经成为 Slurm AI 集群的事实标准容器运行时。

- **误区**："学习 Slurm = 学习一套过时工具"
  - **实际**：Slurm 仍然是几乎所有国家超算中心和大型 GPU 集群的默认调度器。**会 Slurm 意味着可以直接上手 90% 的 GPU 集群**。

### 📊 底线数字速查

| 指标 | Slurm | Kubernetes | 备注 |
|------|-------|-----------|------|
| 最大集群规模 | 10 万+ 节点（公开记录 122,880 节点） | ~5,000 节点 | Slurm 上限更高 |
| 作业启动延迟 | ~1-2 s | ~5-30 s（含 Pod 调度） | Slurm 更快 |
| 批处理调度 | ✅ 原生 | ⚠️ 需扩展组件 | Slurm 天然优势 |
| 推理服务 | ❌ 需要额外方案 | ✅ 原生 | K8s 天然优势 |
| 弹性伸缩 | ⚠️ 需 `--requeue` + checkpoint | ✅ 原生 | K8s 优势 |
| 自愈 | ⚠️ 需显式 `--requeue`，无声明式自愈 | ✅ 原生 | K8s 优势 |
| 拓扑感知 | ✅ native | ⚠️ 需配置 | Slurm 更直接 |
| 学习曲线 | 低（HPC）↔ 高（云原生） | 低（云原生）↔ 高（HPC） | 取决于背景 |

### 💡 一句话记住

> **"Slurm 是 AI 集群调度器的'老兵'——2002 年诞生，经过万卡超算 20 年的考验。它的设计哲学是'资源到位前不执行'，天然适合大规模批处理训练。2026 年，TOP500 中仍有约 60-65% 的系统跑在 Slurm 上，但它的角色正从'唯一选项'变为'训练调度的最佳选项'。"**

---

## 对后续的铺垫

Slurm 是 HPC 的传统选择。但 Kubernetes 在 AI 场景中正在快速崛起——尤其是推理服务和多租户场景。

它们不是对手，而是互补。2026 年最大的趋势是 **Slurm on Kubernetes（Slinky/SUNK）**——把两者结合起来。

→ [[07-调度篇/03-Kubernetes为什么在AI场景崛起]]
→ [[04-NVIDIA收购SchedMD意味着什么]]

---

> **参考资料**：
> - ["What Does NVIDIA's Acquisition of SchedMD Mean for Slurm?" — HPCwire (2026)](https://www.hpcwire.com/2026/01/06/what-does-nvidias-acquisition-of-schedmd-mean-for-slurm/)
> - [Crusoe: "Slurm on Crusoe Managed Kubernetes" (2026)](https://www.crusoe.ai/resources/blog/slurm-on-crusoe-managed-kubernetes-how-we-built-managed-gpu-training-infrastructure)
> - [Slurm 官方文档](https://slurm.schedmd.com/documentation.html)
> - [Slurm 抢占（Preemption）文档](https://slurm.schedmd.com/preempt.html)
> - [Slurm 多因子优先级（Multifactor Priority）文档](https://slurm.schedmd.com/priority_multifactor.html)
> - [Slurm 大规模集群管理指南（Large Cluster Administration）](https://slurm.schedmd.com/big_sys.html)
> - [MLSysBook Vol.II: "Fleet Orchestration" (2026)](https://mlsysbook.ai/vol2/fleet_orchestration/fleet_orchestration.html)
