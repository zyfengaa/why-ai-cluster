# Q: Gang Scheduling 是什么？

## 为什么会有这个问题

你提交了一个需要 512 张 GPU 的训练任务。调度器说："好的，我看看……现在有 510 张空闲。先给你 510 张，等你起来了我再加 2 张。"

你的任务启动了 510 张 GPU。但 NCCL 初始化时发现"说好 8 台机器，怎么只到了 7.9 台？"——要么初始化失败，要么通信拓扑不对，性能暴跌。

**这就是分布式训练的"鸡生蛋"问题：没有全部到位，不如一个都不启动。**

## 解决什么问题

Gang Scheduling（集合调度）的核心规则：**要么全部资源一起给，一个都不少；要么一个都不给，继续等。**

### 为什么普通调度不行

```
普通调度器（逐个分配）：
  时间 0：任务 A 请求 4 台机器 × 8 GPU = 32 GPU
  时间 1：调度器分配 3 台机器（24 GPU）→ 任务 A 的 3 台启动
  时间 2：调度器找到第 4 台机器 → 但前 3 台已经在等（NCCL 超时）
  时间 3：任务 A 的 NCCL 初始化失败 → 全部 3 台空转浪费

Gang Scheduling（集合分配）：
  时间 0：任务 A 请求 4 台机器 × 8 GPU = 32 GPU
  时间 1：调度器预留 3 台（24 GPU），还差 1 台 → 继续等
  时间 2：调度器找到第 4 台机器 → 32 GPU 全部到位
  时间 3：任务 A 启动 → 4 台同时开始 → NCCL 成功初始化
                 ↑ 虽然等了 2 个时间单位，但没有 GPU 浪费
```

### 为什么"等一点都不浪费"

这是 Gang Scheduling 反直觉的地方——**等资源看似浪费了时间，但避免了更大的浪费**：

```
普通调度：
  30 分钟等待（找最后那台机器）+ 30 分钟空转（NCCL 重试）
  = 60 分钟浪费

Gang Scheduling：
  30 分钟等待（资源集齐才启动）+ 0 分钟空转
  = 30 分钟等待
  + 节省了 30 分钟 GPU 空转
```

### Gang Scheduling 在不同调度器中的实现

| 调度器 | 实现方式 | 状态（2026） |
|--------|---------|-------------|
| **Slurm** | 原生支持（资源粒度分配） | 成熟，20 年验证 |
| **Kubernetes 原生（KEP-4671）** | 内置于核心调度器 | v1.37 达 beta，**默认关闭** |
| **Kubernetes + Volcano** | PodGroup CRD | 生产级 |
| **Kubernetes + KAI Scheduler** | 公平分享排队 | 生产级（NVIDIA 推荐）|
| **Kubernetes + Kueue** | Workload 排队 | beta |

> 注意：**K8s 原生 gang scheduling（KEP-4671）已在 v1.37 进入核心并达到 beta，但默认关闭**——所以别再说"K8s 必须靠 Volcano/KAI/Kueue 才能做 gang 调度"。生产上仍需 Volcano/KAI/Kueue 等组件来提供更完整的队列、配额和拓扑能力，但原生能力已经存在。

### Slurm 的 Gang Scheduling

```bash
# Slurm 原生就是 Gang Scheduling
# --nodes=4 表示"必须在全部 4 个节点同时可用时才启动"
sbatch --nodes=4 --ntasks-per-node=8 --gpus-per-node=8 train.sh

# Slurm 不会在只有 3 个节点时启动任务
# 这就是它天然适合分布式训练的原因
```

### Kubernetes 的 Gang Scheduling（以 Volcano 为例）

```yaml
apiVersion: scheduling.volcano.sh/v1beta1
kind: PodGroup
metadata:
  name: training-group
spec:
  minMember: 32          # 最少需要 32 个 Pod
  minResources:           # 最少需要的资源
    nvidia.com/gpu: "256"
  queue: training-queue  # 排队队列
---
apiVersion: v1
kind: Pod
metadata:
  name: trainer-0
  annotations:
    # Volcano 用"注解"而非标签关联 PodGroup，键为 scheduling.k8s.io/group-name
    scheduling.k8s.io/group-name: training-group
spec:
  schedulerName: volcano  # 使用 Volcano 调度器
  ...
```

---

## Gang Scheduling 的代价

Gang Scheduling 不是免费午餐——它有几个核心问题：**资源碎片、饥饿和死锁**。

### 资源碎片

```
假设集群有 4 台 8-GPU 机器（共 32 GPU）：

任务 A：需要 16 GPU（2 台机器）→ 等
任务 B：需要 8 GPU（1 台机器）→ 等

场景：任务 A 先来，占了 1 台（8 GPU），需要再 1 台但机器 B 被占用了
      → 任务 A 卡在 Gang Scheduling 上
      → 任务 B 只需要 8 GPU，但 Gang Scheduling 的等待让 B 也跑不了
      → 32 GPU 利用率 = 25%（只有 1 台被预留）
```

### 饥饿

```
一个大任务占用了"很大比例的预留资源"，但一直凑不齐：
  - 任务 A 需要 256 GPU
  - 集群有 256 GPU
  - 任务 A 预留了 255 GPU，等 1 台空闲
  - 但其他任务只能用剩下的 1 GPU
  - 集群利用率 ≈ 99%（255 预留），但实际吞吐 = 0
  
这就是"大任务饥饿"问题
```

### 死锁

比饥饿更隐蔽的是**同优先级 gang 互等导致的死锁**——这是 gang scheduling 的核心难点：

```
假设集群有 20 GPU，两个任务同时排队：

  任务 A：需要 12 GPU → 先抢到 10 GPU，等剩余 2 GPU
  任务 B：需要 12 GPU → 先抢到 10 GPU，等剩余 2 GPU
  → A 等 B 释放，B 等 A 释放，谁都不肯先放弃已占用的 10 GPU
  → 20 GPU 全部被"预留"，但两个任务都永远启动不了
```

根源在于**预留（reservation）把"等待"变成了"占用"**。规避手段：

| 手段 | 原理 |
|------|------|
| **禁止部分预留** | 一个 gang 要么整体占住、要么完全不占（Slurm 的默认原子分配、Volcano 的 `minMember` 都是这个思路） |
| **超时回退** | 等待超过阈值就释放已预留资源、让出位置（Slurm `--switches` 的时间窗、Volcano 的超时都是此类） |
| **优先级 + 抢占** | 让高优先级大任务可以"挤掉"低优先级任务，打破互等 |
| **Backfill** | 用小作业填补预留之外的缝隙，减少资源空转 |

### 解决方案

| 策略 | 原理 | 效果 |
|------|------|------|
| **超配（Overcommit）** | 预留的资源也可以临时给其他任务用 | 提高利用率，但抢占时有代价 |
| **Backfill** | 大任务等待时，小任务"填空" | Slurm 原生支持 |
| **优先级抢占** | 高优先级任务可以"抢"低优先级的资源 | 适合生产环境 |
| **弹性训练** | 训练任务可以适应资源变化 | 最高端方案（Torch Elastic） |

---

## 🛠 实操指南

### 📋 决策指南：需要 Gang Scheduling 吗？

```
你的训练任务规模？
├─ < 8 GPU（单机）→ 不需要 Gang Scheduling
│   └─ 单机内 NCCL 自动处理
│
├─ 8-64 GPU（多机小规模）→ 建议开启
│   └─ 但 Backfill 更重要（小任务填空）
│
├─ 64-512 GPU（中等规模）→ 必须 Gang Scheduling
│   └─ + 优先级队列（生产任务优先）
│
└─ > 512 GPU（大规模）→ 必须 Gang Scheduling
    └─ + 弹性训练（允许任务随资源变化调整）
```

### 🔧 自己动手验：判断 Gang Scheduling 是否在工作

```bash
# 1. 检查 Slurm 集群是否有任务在"等资源"
squeue -o "%.18i %.9P %.20j %.8u %.8T %.10M %.9l %.6D" | grep PD
# 排队的任务中，如果有很多"Resources"原因 → 可能缺资源

# 2. 检查任务是否因为资源不足而"死等"
scontrol show job <jobid> | grep Reason
# Reason=Resources → 等资源
# Reason=NonZeroExitCode → 之前异常退出

# 3. 检查 Kubernetes Volcano/KAI Scheduler 的 PodGroup 状态
kubectl describe podgroup <name>
# Status.Pending → 在等
# Status.Running → 已全部到位

# 4. 检查有多少 GPU 被预留但未被使用
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv
# 如果有 GPU utilization=0% 但 memory.used>0 → 可能是 Gang Scheduling 预留
```

```python
# 模拟 Gang Scheduling vs 非 Gang Scheduling 的利用率
def simulate_gang_vs_nogang(n_gpus=64):
    """模拟两种调度策略在大规模混部场景下的表现"""
    import random
    
    tasks = []
    for _ in range(30):
        tasks.append({
            'gpus': random.choice([8, 16, 32, 64]),
            'duration': random.randint(10, 100),
            'submitted': random.randint(0, 200),
        })
    
    # 非 Gang：来了就分配
    time_gang = 0
    time_nogang = 0
    
    for t in tasks:
        # 非 Gang：直接算
        time_nogang += t['duration']
        
        # Gang：等资源
        wait_time = t['submitted'] * 0.1  # 等资源时间
        time_gang += t['duration'] + wait_time
    
    print(f"非 Gang 调度总运行时间（模拟）: {time_nogang} 单位")
    print(f"Gang 调度总运行时间（含等待）: {time_gang} 单位")
    print(f"Gang 调度额外等待: {(time_gang/time_nogang -1)*100:.0f}%")
    print()
    print("但是非 Gang 调度会有 NCCL 重试和初始化失败的风险")
    print("实际浪费远大于等待时间")
```

### ⚠️ 常见误区

- **误区**："Gang Scheduling 就是让大任务优先"
  - **实际**：Gang Scheduling 是"要么全给，要么不给"，不是"大任务优先"。优先级和 Gang 是两个独立维度。一个 8 卡的高优任务和一个 512 卡的低优任务同时排队，高优的 8 卡可能先跑。

- **误区**："Gang Scheduling 总是提高利用率"
  - **实际**：**在大集群混部场景中，Gang Scheduling 可能降低利用率**——因为预留的资源不能被其他任务用。这就是为什么需要 Backfill 和超配策略作为补充。

- **误区**："所有分布式训练框架都需要 Gang Scheduling"
  - **实际**：弹性训练（Torch Elastic、PyTorch DCP）可以容忍资源变化。如果使用弹性训练 + FSDP，任务可以在资源不足时自动调整，不需要严格的 Gang Scheduling。

### 📊 底线数字速查

| 指标 | 无 Gang Scheduling | 有 Gang Scheduling | 最佳实践 |
|------|-------------------|-------------------|---------|
| 资源碎片率 | 低（< 5%） | 中（5-15%） | 配合 Backfill |
| 大任务等待时间 | 低（直接启动） | 高（可能等很久） | 需要超配策略 |
| 小任务等待时间 | 低 | 可能被大任务"堵住" | 优先级 + 限制大任务预留时间 |
| GPU 利用率（混部） | 60-70% | 70-85%（+Backfill） | 调度策略决定 |
| 训练稳定性 | 低（NCCL 崩溃风险） | 高 | DevOps 成本 |

### 💡 一句话记住

> **"Gang Scheduling 就是'要么全班同学都到了才上课，一个都不能少'。它的反直觉之处在于——等资源看起来是在浪费时间，但避免了 NCCL 初始化失败后 GPU 空转的更大浪费。2026 年，64 GPU 以上的训练任务，Gang Scheduling 是必须的。"**

---

## 对后续的铺垫

Gang Scheduling 解决了"所有 GPU 同时到位"的问题。但还有一个更精细的维度：**这些 GPU 在物理上是怎么连接的？**

如果 8 张 GPU 分布在 2 台不用 NVSwitch 的服务器上，NCCL All-Reduce 的性能会暴跌。这就是拓扑感知调度要解决的问题。

→ [[06-拓扑感知调度]]

---

> **参考资料**：
> - [Shankar: "How PCIe, NVLink, and NUMA Topology Affect GPU Scheduling" — Dev.to (2026)](https://dev.to/daya-shankar/how-pcie-nvlink-and-numa-topology-affect-gpu-scheduling-outcomes-l52)
> - [Volcano 调度器文档](https://github.com/volcano-sh/volcano)
> - [KAI Scheduler 文档 (NVIDIA, 2025-2026)](https://github.com/NVIDIA/KAI-Scheduler)
> - [Kueue 项目 (Google, 2026)](https://github.com/kubernetes-sigs/kueue)
> - [Slurm `sched/gang`（时间片调度）插件文档](https://slurm.schedmd.com/gang_scheduling.html)：注意这个插件做的是**多个作业分时共享同一批资源**（time-slicing），与本文讲的"all-or-nothing、全部到位才启动"是**两个不同概念**。Slurm 的"全部到位才启动"来自其**多节点原子分配的默认行为**，并不需要这个插件。
