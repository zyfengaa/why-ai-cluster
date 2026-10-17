# Q: Slurm 为什么是 HPC 的传统选择？

## 为什么会有这个问题

当你走进任何一个国家级超算中心——Frontier、Aurora、LUMI、Perlmutter——问他们用什么调度器，答案几乎都是 **Slurm**。它从 2002 年开始开发，已经统治 HPC 调度 20 多年。

也有例外：**Fugaku（富岳）用的是富士通自研的 TCS（Fujitsu Technical Computing Suite）调度器**（`pjsub`/`pjstat` 命令），它与富士通 Tofu-D 互联深度绑定——Slurm 只被用在 Fugaku 的前处理/后处理环境上。所以准确地说：Slurm 统治了"绝大多数"超算，而不是"全部"。

在 AI 浪潮中，Slurm 被"挖"出来管理 GPU 集群。不是因为它最时尚，而是因为 **大规模分布式训练的需求和 HPC 批处理作业几乎一模一样**。

## 解决什么问题

Slurm（Simple Linux Utility for Resource Management）解决的核心问题：

1. **大规模批处理调度**：在几万个节点上调度数百万个 CPU 小时的作业
2. **资源隔离与公平分享**：多用户、多项目共享集群时保证公平
3. **作业依赖与流水线**：作业 A 完成了才启动作业 B
4. **拓扑感知分配**：知道哪些 GPU/节点在同一个网络域内

官方口径的规模上限很明确：**Slurm 的几乎所有组件都已经过（模拟方式）验证，可支持最多 65,536 个计算节点的集群**（见 SchedMD 官方 Large Cluster Administration Guide）。

### Slurm 的设计哲学

```
"资源到位前不执行，资源释放前不让道"
```

这是 Slurm 和 Kubernetes 最大的区别：**Slurm 默认把作业申请的资源当成一个整体，凑齐才启动；Kubernetes 默认逐个 Pod 调度**（虽然它也有抢占机制，但那不是"全部到位才启动"的语义）。

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

# 查看节点状态（可运行示例）
# %P 分区  %a 可用性  %D 节点数  %t 节点状态（紧凑）  %C CPU（已分配/空闲/其他/总计）  %m 内存  %G GRES（GPU 等）
sinfo -o "%P %a %D %t %C %m %G"
# 输出示例：
# PARTITION AVAIL NODES STATE  CPUS(A/I/O/T) MEMORY   GRES
# gpu-a100* up    16    idle   0/128/0/128   1024000  gpu:a100:8
# gpu-h100  up    8     mix    64/64/0/128   2048000  gpu:h100:8

# 取消作业
scancel <jobid>

# 查看作业详情
scontrol show job <jobid>
```

### Slurm 在 AI 场景中的天然优势

1. **整块分配天然满足"全部到位才启动"**：Slurm 默认按作业申请的节点/卡数一次性分配资源，凑不齐就不启动。你不会希望 512 卡的任务只拿到 500 张卡就启动。（注意：Slurm 官方语境里的 "gang scheduling" 是另一个概念——时间片轮转，详见本篇 05。）

2. **MPI 原生集成**：`srun` 天然支持 MPI 任务，NCCL 的通信初始化直接兼容 Slurm 的节点分配。

3. **拓扑感知**：`topology.conf` 可以配置网络交换机层级（机内 NVSwitch/PCIe 拓扑 + 机间 IB leaf/spine 层级），Slurm 分配时尽量把任务放在拓扑最优的节点上。

4. **万卡级生产验证**：Frontier（约 3.8 万张 AMD MI250X）、Aurora、LUMI 等顶级超算全部用 Slurm；按 NVIDIA 收购 SchedMD 时公布的数据，TOP500 中约 65% 的系统使用 Slurm。

### GPU 作业的关键配置：GRES、cgroup 与独占

```bash
# ---------- slurm.conf / gres.conf 侧（管理员） ----------
# slurm.conf 中声明 GRES 类型与节点 GPU 数量
# GresTypes=gpu
# NodeName=gpu-node[01-16] Gres=gpu:h100:8 CPUs=128 RealMemory=2048000

# cgroup.conf 中开启设备隔离，防止作业越权使用未分配的 GPU
# ConstrainDevices=yes

# ---------- 用户侧 ----------
# 申请每节点 8 张 GPU（由 GRES 提供）
sbatch --nodes=4 --ntasks-per-node=8 --gpus-per-node=8 train.sh

# 指定 GPU 型号与数量
sbatch --gpus=h100:4 train.sh

# 每张 GPU 配 16 个 CPU 核心、独立内存配额
sbatch --gpus=8 --cpus-per-gpu=16 --mem-per-gpu=64G train.sh

# 独占节点：整台机器都给这个作业，不留其他作业的 CPU/GPU
sbatch --exclusive --nodes=2 --gpus-per-node=8 train.sh
```

`--gpus-per-node` / `--gpus` 是 `--gres=gpu:N` 的语法糖；`--cpus-per-gpu` / `--mem-per-gpu` 保证通信线程（NCCL、Dataloader）有足够的 CPU 和 pinned memory；`--exclusive` 在大规模训练中最常用——宁可整节点独占，也不要和别的作业共享一台机器互相抢带宽。

```bash
# Slurm 拓扑配置示例（topology.conf，树状网络拓扑）
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
| **无容器原生支持** | 需借助 Pyxis/Enroot 或 Apptainer（原 Singularity） | 开发和部署环境不一致 |
| **无自愈能力** | 节点挂了，任务就挂了 | 需要外部脚本监控和重提交 |
| **无弹性调度** | 任务一旦启动，资源固定 | 不能动态扩缩 |
| **无原生推理服务** | Slurm 不是为长运行服务设计的 | 推理服务需要额外方案 |
| **API 不够 RESTful** | 官方 REST API（slurmrestd）成熟度仍在提升 | 和 CI/CD 集成需要额外工作 |

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
# 一个标准的 Slurm 多机训练作业脚本
#!/bin/bash
#SBATCH --job-name=train-llama
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=128
#SBATCH --mem=512G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu-a100
#SBATCH --exclusive

# 加载环境
module load cuda/12.4
module load nccl/2.21
conda activate llama-training

# NCCL 调优
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0,mlx5_1
export NCCL_NET_GDR_LEVEL=5

# 关键点：必须用 srun 启动 torchrun，让每个节点跑一个 launcher 进程，
# torchrun 通过 Slurm 分配的节点列表做 rendezvous。
# 不要只在单个节点上直接跑 torchrun --nnodes=4 —— 那样绕过了 Slurm 的
# 作业分配，其余 3 个节点上的 GPU 不会被使用，也不会有跨机通信。
HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)

srun --nodes=4 --ntasks-per-node=1 torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc-per-node=$SLURM_GPUS_ON_NODE \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$HEAD_NODE:29500 \
  train.py \
  --model-size 70b \
  --batch-size 4
```

> 说明：这里每个 Slurm task 是"每节点一个 torchrun 进程"。torchrun 自身再在节点内拉起 8 个训练进程。另一种写法是不用 torchrun，让 `srun --ntasks-per-node=8` 直接启动 8 个带 rank 环境变量的训练进程（MPI 风格），两者都常见，关键是**进程必须由 `srun` 在 Slurm 分配的资源内启动**。

### 🔧 自己动手验：检查 Slurm 集群的健康状态

```bash
# 1. 查看集群总资源
sinfo -o "%P %D %t %C %m %G"
# 输出示例：
# PARTITION NODES STATE CPUS(A/I/O/T) MEMORY GRES
# gpu-a100* 16    idle  0/128/0/128   1024000 gpu:a100:8
# gpu-h100  8     mix   64/64/0/128   2048000 gpu:h100:8

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
  - **实际**：Frontier（2022 年起首个 ExaFLOPS 级超算）用 Slurm 管理数万张 AMD MI250X GPU；按 NVIDIA/SchedMD 公告口径，TOP500 约 65% 的系统运行 Slurm，大量大型 GPU 集群同样如此。**老 ≠ 不适合**，Slurm 的大规模调度能力在 AI 领域同样关键。

- **误区**："Slurm 不支持容器"
  - **实际**：Slurm + Pyxis/Enroot 是常见的集群容器方案，2026 年很多 AI 集群用 Pyxis 作为事实标准容器运行时；Apptainer（原 Singularity）在超算侧同样流行。

- **误区**："学习 Slurm = 学习一套过时工具"
  - **实际**：Slurm 仍然是绝大多数国家超算中心和大型 GPU 集群的默认调度器。**会 Slurm 意味着可以直接上手大多数 GPU 集群**。

### 📊 底线数字速查

| 指标 | Slurm | Kubernetes | 备注 |
|------|-------|-----------|------|
| 已验证集群规模 | **65,536 节点**（官方模拟验证口径） | 数千节点（取决于发行版与调优） | Slurm 上限更高 |
| 作业启动延迟 | 秒级（资源到位即可启动） | 秒到分钟级（含 Pod 调度与镜像拉取） | Slurm 更快 |
| 批处理调度 | ✅ 原生 | ⚠️ 需扩展组件 | Slurm 天然优势 |
| 推理服务 | ❌ 需要额外方案 | ✅ 原生 | K8s 天然优势 |
| 弹性伸缩 | ❌ 不支持 | ✅ 原生 | K8s 优势 |
| 自愈 | ❌ 不支持 | ✅ 原生 | K8s 优势 |
| 拓扑感知 | ✅ 原生（topology.conf） | ⚠️ 需配置 | Slurm 更直接 |
| 学习曲线 | 低（HPC）↔ 高（云原生） | 低（云原生）↔ 高（HPC） | 取决于背景 |

### 💡 一句话记住

> **"Slurm 是 AI 集群调度器的'老兵'——2002 年开始开发，经过万卡超算 20 多年的考验，官方验证过 65,536 节点的规模。它的设计哲学是'资源到位前不执行'，天然适合大规模批处理训练。2026 年，Slurm 仍然运行着 TOP500 中约 65% 的系统，但它的角色正从'唯一选项'变为'训练调度的最佳选项'。"**

---

## 对后续的铺垫

Slurm 是 HPC 的传统选择。但 Kubernetes 在 AI 场景中正在快速崛起——尤其是推理服务和多租户场景。

它们不是对手，而是互补。2024 年以来最大的趋势是 **Slurm on Kubernetes（Slinky/SUNK）**——把两者结合起来。

→ [[07-调度篇/03-Kubernetes为什么在AI场景崛起]]
→ [[04-NVIDIA收购SchedMD意味着什么]]

---

> **参考资料**：
> - [Slurm Workload Manager 官方文档](https://slurm.schedmd.com/documentation.html)
> - [Slurm Large Cluster Administration Guide（65,536 节点验证口径）](https://slurm.schedmd.com/big_sys.html)
> - [sinfo 手册页（格式说明符）](https://manpages.ubuntu.com/manpages/noble/man1/sinfo.1.html)
> - [Fujitsu: "Operations Management Software of Supercomputer Fugaku"（Fugaku 自研调度器）](https://www.fujitsu.com/global/documents/about/resources/publications/technicalreview/2020-03/article10.pdf)
> - [TOP500 List — November 2025（El Capitan #1 / Frontier #2）](https://www.top500.org/lists/top500/list/2025/11/)
> - [NVIDIA: "NVIDIA Acquires SchedMD"（TOP500 约 65% 使用 Slurm）](https://blogs.nvidia.com/blog/nvidia-acquires-schedmd/)
