# Q: Gang Scheduling 是什么？

## 为什么会有这个问题

你提交了一个需要 512 张 GPU 的训练任务。调度器说："好的，我看看……现在有 510 张空闲。先给你 510 张，等你起来了我再加 2 张。"

你的任务启动了 510 张 GPU。但 NCCL 初始化时发现"说好 64 台机器，怎么只到了 63.8 台？"——要么初始化失败，要么通信拓扑不对，性能暴跌。

**这就是分布式训练的"鸡生蛋"问题：没有全部到位，不如一个都不启动。**

## 解决什么问题

先正名：**"Gang Scheduling" 这个词在两个世界里指两件不同的事**，混着说会出大问题。

| 语境 | 含义 | 关键机制 |
|------|------|---------|
| **Kubernetes 语境（本文主角）** | **全组原子分配（all-or-nothing）**：一个作业的全部 Pod 要么同时调度成功，要么一个都不启动 | PodGroup / Workload + 配额预留 |
| **Slurm 官方语境** | **时间片轮转（time-slicing）**：多个作业被同时分配到重叠资源上，调度器用 suspend/resume 让它们轮流运行 | `PreemptMode=suspend,gang`（见 [Slurm Gang Scheduling 文档](https://slurm.schedmd.com/gang_scheduling.html)） |

Slurm 满足"全部到位才启动"靠的**不是** gang scheduling，而是它默认的**整块资源分配**：作业申请 64 台节点，Slurm 凑不齐 64 台就不启动它。这一点和 Kubernetes 默认的"逐 Pod 调度"形成对比，也是 Slurm 天然适合分布式训练的原因。下文如无特别说明，"Gang Scheduling" 指 Kubernetes 语境的全组原子分配。

### 为什么逐个调度不行

```
逐个调度（Kubernetes 默认行为）：
  时间 0：任务 A 请求 4 台机器 × 8 GPU = 32 GPU
  时间 1：调度器逐个绑定 Pod → 3 台机器（24 GPU）上的 Pod 先启动
  时间 2：第 4 台机器被任务 B 占了 → A 的 24 个 Pod 继续占着资源等待
  时间 3：A 的 NCCL 初始化超时 → 重试 → 仍然凑不齐
  结果：24 GPU 被"启动但永远跑不起来"的 Pod 长期占住，
        任务 B 正常运行，其他任务也被这 24 GPU 挤占 → 部分死锁

全组分配（Gang Scheduling）：
  时间 0：任务 A 请求 4 台机器 × 8 GPU = 32 GPU
  时间 1：调度器确认当前只有 24 GPU 可用 → A 整组不启动，一个 Pod 都不绑定
  时间 2：任务 B（8 GPU）小，先满足 → 跑完释放
  时间 3：32 GPU 全部可用 → A 整组原子启动 → NCCL 成功初始化
```

逐个调度的真正危害不是"启动慢"，而是**部分启动的组会占住资源既不干活也不释放**——这正是死锁的温床。Gang 的本质是把"能不能跑"的判断提前到绑定之前。

### 为什么"等一点都不浪费"

这是 Gang Scheduling 反直觉的地方——**等资源看似浪费了时间，但避免了更大的浪费**：

```
逐个调度：
  24 GPU 被半启动的组占住（不产出）+ NCCL 超时重试
  = 持续性的资源占用 + 可能的反复失败

Gang Scheduling：
  整组排队等待（不占资源）→ 集齐后 100% 一次到位
  = 等待期间资源完全可用（可以被小任务 backfill）
```

### 各调度器里的实现（2026）

| 系统 | 实现方式 | 状态（2026） |
|------|---------|-------------|
| **Slurm** | 默认整块分配，天然满足"全部到位才启动"；官方 "gang scheduling" 是另一回事（时间片轮转） | 成熟，20+ 年验证 |
| **Kubernetes + Volcano** | PodGroup CRD（`scheduling.volcano.sh/v1beta1`）+ `scheduling.k8s.io/group-name` 注解 | 生产级（CNCF 孵化项目） |
| **Kubernetes + KAI Scheduler** | 分层队列 + gang + 弹性 | 生产级（NVIDIA 开源） |
| **Kueue** | Workload 是排队/准入的基本单位：配额凑齐才放行（Pod 在此之前保持 suspended） | kubernetes-sigs 项目，大量生产使用（API v1beta2） |
| **scheduler-plugins coscheduling** | PodGroup CRD（`scheduling.x-k8s.io/v1alpha1`）+ `scheduling.x-k8s.io/pod-group` 标签 | SIG 官方插件，生产可用 |

### Slurm 的做法

```bash
# Slurm 默认就是整块分配：--nodes=4 表示"全部 4 个节点同时可用时才启动"
sbatch --nodes=4 --ntasks-per-node=8 --gpus-per-node=8 train.sh

# Slurm 不会在只有 3 个节点时启动这个任务——这就是它天然适合分布式训练的原因
# （整块分配 + backfill 是 Slurm 的默认组合，不需要额外开启 gang）

# 官方的 "gang scheduling"（PreemptMode=suspend,gang）是时间片轮转：
# 让高优先级作业可以挂起（suspend）低优先级作业并抢占，之后按时间片轮流运行。
# 它解决的是"优先级抢占时作业不丢"，不是"全部到位才启动"。
```

### Kubernetes 的 Gang Scheduling（以 Volcano 为例）

```yaml
# Volcano v1.10+ 验证的写法（PodGroup API 长期稳定在 v1beta1）
apiVersion: scheduling.volcano.sh/v1beta1
kind: PodGroup
metadata:
  name: training-group
spec:
  minMember: 32          # 最少需要 32 个 Pod 一起调度
  minResources:          # 组内最少需要的资源总量
    nvidia.com/gpu: "256"
  priorityClassName: high-priority
  queue: training-queue  # 排队队列
---
apiVersion: v1
kind: Pod
metadata:
  name: trainer-0
  annotations:
    scheduling.k8s.io/group-name: training-group   # 标准 PodGroup 关联注解
spec:
  schedulerName: volcano   # 使用 Volcano 调度器
  containers:
  - name: trainer
    image: pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime
    resources:
      limits:
        nvidia.com/gpu: 8
  # ... 其余 31 个 Pod 同样带这个注解
```

要点：
- Pod 通过 **注解 `scheduling.k8s.io/group-name`** 关联到 PodGroup——这是 Kubernetes 生态约定俗成的标准 gang 注解，Volcano、aibrix 等项目都沿用同一写法。
- 在 Deployment/Job 里，把注解写在 `spec.template.metadata.annotations` 上，由 Volcano 的 controller 自动创建 PodGroup。
- **不要写 `volcano.sh/podgroup-name` 这样的注解——它不存在**，写了不会报错但 gang 语义完全不生效，这是最常见的踩坑点。

### 替代与补充方案

| 方案 | 原理 | 何时选它 |
|------|------|---------|
| **Kueue** | Workload 声明 `podSets`（角色×副本×资源），配额凑齐才整体 admit，之前 Pod 保持 suspended | 想保留 kube-scheduler，只要排队/配额/准入 |
| **KAI Scheduler** | 分层队列 + 公平分享 + gang + 弹性 | NVIDIA 生态，GPU 分时/弹性需求 |
| **coscheduling plugin** | kube-scheduler 插件，Pod 用标签 `scheduling.x-k8s.io/pod-group` 组队，`minMember` 达标才放行 | 不想引入独立调度器，只要 gang |
| **Slurm `--exclusive` + 拓扑约束** | 整块分配 + `--switches`/topology.conf 限制节点位置 | 纯 Slurm 集群 |
| **弹性训练** | TorchElastic/`torchrun` 允许 worker 数变化 | 可容忍资源变化、优先吞吐弹性的场景 |

---

## Gang Scheduling 的代价

Gang Scheduling 不是免费午餐——它有一个核心问题：**资源碎片和饥饿**。

### 碎片与"占住不起"

```
一个组凑不齐时，它一个 Pod 都不会启动（这是 Gang 的正确行为）。
但如果配额/预留机制实现不当，或调度器只为"未来可能凑齐"而做大量预留：

  大任务 G 需要 64 台机器，当前集群只有 60 台可用
  → G 整组排队（正确，不占资源）
  → 但如果调度器为 G "预订"了潜在资源、或队列策略不允许小任务插入
  → 小任务也进不来 → 集群吞吐趋近于零
  
  极端情形（backfill 缺失的集群）：
  G 的最后几台机器要等一个长作业跑完才能释放，
  而 G 又把队列堵死 → 其他所有任务陪等
```

判断标准很简单：**看"排队中的资源"有多少真的在等一个可能凑不齐的组**。等待本身无害，占着不释放才有害。

### 饥饿

```
一个大组长期凑不齐 → 反复重试 → 每次重试都要"独占式"确认可用资源
  → 队列里比它小的任务如果排队策略不优，会被一直压在后面

应对：优先级队列 + 老化（aging，等待越久优先级越高）
      + 限制单个组的最大申请量 + backfill
```

### 解决方案

| 策略 | 原理 | 效果 |
|------|------|------|
| **Backfill** | 大任务等待时，小任务"填空"跑掉 | Slurm 原生支持（sched/backfill，conservative 模式） |
| **超配（Overcommit）** | 预留的资源也可以临时给其他任务用 | 提高利用率，但抢占时有代价 |
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
# Reason=Priority → 优先级低

# 3. 检查 Kubernetes Volcano 的 PodGroup 状态
kubectl describe podgroup <name>
# Phase=Pending → 组还没凑齐，整组等待
# Phase=Running → 已全部到位
# minMember 与当前 Running 的 Pod 数对比 → 是否全部启动

# 4. 检查有多少 GPU 被预留但未被使用
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv
# 如果有 GPU utilization=0% 但 memory.used>0 → 可能是 Gang 预留或半启动的组
```

```python
# 模拟：逐个调度 vs Gang 的资源占用差异（机制示意，非精确数字）
def simulate_deadlock(n_machines=8, gpus_per_machine=8):
    """大组 5 台机器，小组 1 台机器；当前只有 4 台空闲"""
    idle_machines = 4
    # 逐个调度：大组先到，抢走 4 台，永远凑不齐第 5 台
    big_group_holds = min(4, 5)      # 4 台被占住，作业无法启动
    small_group_blocked = idle_machines - big_group_holds  # 0 台剩给小组
    print(f"逐个调度：大组占 {big_group_holds} 台但无法启动，小组可用 {small_group_blocked} 台")

    # Gang：大组整组等待，不绑定任何 Pod
    gang_holds = 0
    small_group_ok = idle_machines - gang_holds
    print(f"Gang 调度：大组等待（占 {gang_holds} 台），小组可用 {small_group_ok} 台并正常运行")

simulate_deadlock()
```

### ⚠️ 常见误区

- **误区**："Slurm 的 gang scheduling 就是 all-or-nothing 集合调度"
  - **实际**：Slurm 官方语境的 gang scheduling 是**时间片轮转**（suspend/resume 轮流运行）。你要的"全部到位才启动"是 Slurm 默认的**整块分配**，两者别混为一谈——面试和生产配置里都会踩这个坑。

- **误区**："Gang Scheduling 就是让大任务优先"
  - **实际**：Gang Scheduling 是"要么全给，要么不给"，不是"大任务优先"。优先级和 Gang 是两个独立维度。一个 8 卡的高优任务和一个 512 卡的低优任务同时排队，高优的 8 卡可能先跑。

- **误区**："Gang Scheduling 总是提高利用率"
  - **实际**：**在大集群混部场景中，不当的 Gang 实现可能降低利用率**——半启动的组占住资源不释放、或预留把队列堵死。这就是为什么需要 Backfill 和超配策略作为补充。

- **误区**："所有分布式训练框架都需要 Gang Scheduling"
  - **实际**：弹性训练（Torch Elastic、`torchrun`）可以容忍资源变化。如果使用弹性训练 + FSDP，任务可以在资源不足时自动调整，不需要严格的 Gang Scheduling。

### 📊 底线数字速查

| 指标 | 无 Gang Scheduling | 有 Gang Scheduling | 说明 |
|------|-------------------|-------------------|------|
| NCCL 初始化成功率 | 半启动的组会反复超时失败 | 接近 100%（全组一起启动） | Gang 的核心价值 |
| 半启动组的资源占用 | 有（占住资源等不到同伴） | 无（整组等待，不绑定） | 死锁的根源差异 |
| 资源碎片率 | 低 | 略高（整组门槛挡住小任务） | 配合 Backfill 化解 |
| 大任务等待时间 | 表面更短（先启动再说） | 可能更久（等整组凑齐） | 但等待期间资源可被小任务使用 |
| 小任务等待时间 | 可能被半启动的大组堵死 | 需要 backfill / 优先级保证 | 队列策略决定 |
| 训练稳定性 | 低（NCCL 崩溃风险） | 高 | 生产训练标配 |

### 💡 一句话记住

> **"Gang Scheduling 就是'要么全班同学都到了才上课，一个都不能少'。它的反直觉之处在于——等资源看起来是在浪费时间，但避免了'半启动的组占住资源永远跑不起来'的更大浪费。另外记住：K8s 的 gang 是原子分配，Slurm 的 gang 是时间片轮转，两个词长得一样，说的是两件事。2026 年，64 GPU 以上的训练任务，Gang Scheduling 是必须的。"**

---

## 对后续的铺垫

Gang Scheduling 解决了"所有 GPU 同时到位"的问题。但还有一个更精细的维度：**这些 GPU 在物理上是怎么连接的？**

如果 8 张 GPU 分布在 2 台不同的服务器上，NCCL All-Reduce 就要走跨机网络，性能会暴跌。这就是拓扑感知调度要解决的问题。

→ [[06-拓扑感知调度]]

---

> **参考资料**：
> - [Slurm Gang Scheduling 官方文档（时间片轮转语义）](https://slurm.schedmd.com/gang_scheduling.html)
> - [Slurm Preempt / PreemptMode 文档](https://slurm.schedmd.com/preempt.html)
> - [Volcano 官方文档](https://volcano.sh/en/docs/) / [volcano-sh/volcano（GitHub）](https://github.com/volcano-sh/volcano)
> - [Kubernetes SIG Scheduler Plugins：Coscheduling 插件](https://scheduler-plugins.sigs.k8s.io/docs/plugins/coscheduling/)
> - [Kueue Overview（kubernetes-sigs）](https://kueue.sigs.k8s.io/docs/overview/)
> - [NVIDIA KAI Scheduler（GitHub）](https://github.com/NVIDIA/KAI-Scheduler)
