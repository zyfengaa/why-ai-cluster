# Q: 调度器装好了，队列为什么还是排不好？

## 为什么会有这个问题

你已经装好了 Slurm 或 K8s，任务能提交、能跑。但生产里最常见的抱怨不是"调度器不能用"，而是：

- 大任务永远排不上，小任务一直在跑；
- 一个组不知不觉占了一半 GPU，别的组只能等；
- 推理服务和训练抢同一批卡，互相拖慢；
- 一张 80GB 的卡上只跑了一个 7B 推理，利用率 20%；
- 凌晨两点的开发任务把白天的训练挤掉了。

**"能调度"和"调度得好"是两件事。** 前者靠装软件，后者靠四个旋钮：优先级与公平、配额、回填与预留、GPU 共享。

## 解决什么问题：四个旋钮

```
谁先跑？   → 优先级与公平（priority / fair-share）
谁能占多少？→ 配额与限额（QOS / quota / TRES）
怎么不浪费？→ 回填与预留（backfill / reservation）
一张卡怎么分？→ GPU 共享（MIG / MPS / time-slicing）
```

---

## 旋钮一：优先级与公平

### Slurm 的多因子优先级

Slurm 的优先级不是一个数，而是多个因子的加权和：

```
Priority = SiteFactor × (Age + FairShare + JobSize + Partition + QOS + TRES + Nice)

Age        等待越久越高
FairShare  历史用量越少越高（按衰减半衰期计算）
JobSize    大任务加权（可关掉）
Partition  分区权重
QOS        服务质量等级权重
TRES       资源类型的计费权重（如 GPU 比 CPU 贵）
Nice       用户手动调整量
```

权重在 `slurm.conf` 里配置（`PriorityWeightAge`、`PriorityWeightFairshare`、`PriorityWeightQOS`…），**默认配置不代表适合你**。

```bash
# 看某个作业各因子的取值
sprio -l -j <jobid>

# 看全局权重配置
scontrol show config | grep -i priority

# 看账户的公平份额与历史用量
sshare -U -l
```

**FairShare 不是"平均分"**：它按历史用量做衰减（`PriorityDecayHalfLife`，默认约一周量级）。一个组昨天用光了集群，今天它的优先级会被压低，但随着时间推移逐渐恢复。这正是"公平"和"平均"的区别。

### K8s 的对应物

Kubernetes 没有 FairShare，它用 **PriorityClass + 抢占**，再靠 Kueue 补上排队与配额：

| 需求 | K8s 原生 | Kueue |
|---|---|---|
| 优先级 | `PriorityClass`（数值越大越优先） | 继承 |
| 抢占 | kube-scheduler 抢占 | `preemption` 策略（Reclaim / Preempt / Borrow） |
| 配额 | `ResourceQuota`（硬上限） | `ClusterQueue` 的 `nominalQuota` |
| 借用 | ❌ | `cohort` + `borrowingLimit` / `lendingLimit` |
| 排队 | ❌（Pod 直接 Pending） | `LocalQueue` / `Workload` |

一句话：**K8s 原生解决"优先级"，Kueue 解决"排队与配额借用"。** 只装 ResourceQuota 不叫公平调度。

---

## 旋钮二：配额与限额

### Slurm：QOS + TRES 计费

```bash
# 建一个 QOS，限制组内 GPU 总量、单用户上限、最长墙钟
sacctmgr add qos train \
  GrpTRES=gpu=128 \
  MaxTRESPerUser=cpu=64,gpu=16 \
  MaxJobsPerUser=20 \
  MaxWall=24:00:00

# 让 GPU 比 CPU "贵"，防止用 CPU 刷公平份额
# slurm.conf（QOS 或 Partition 上）
# TRESBillingWeights=cpu=0.1,mem=0.01,gpu=1.0

# 用量报表（按账户/用户/分区）
sreport cluster AccountUtilizationByUser Start=2026-01-01 End=2026-01-31
```

**关键点**：`GrpTRES` 管总量，`MaxTRESPerUser` 管个人，`TRESBillingWeights` 决定"公平份额怎么算钱"。三者缺一，都会出现"有人把集群刷满但账面看不出来"。

### Kueue：配额 + 借用

```yaml
# 以你部署的 Kueue 版本为准（v1beta1 / v1beta2 字段略有差异）
apiVersion: kueue.x-k8s.io/v1beta2
kind: ClusterQueue
metadata:
  name: gpu-cluster-queue
spec:
  namespaceSelector: {}
  resourceGroups:
  - coveredResources: ["cpu", "memory", "nvidia.com/gpu"]
    flavors:
    - name: h100
      resources:
      - name: nvidia.com/gpu
        nominalQuota: 64
        borrowingLimit: 32      # 可以从同 cohort 的其他队列借多少
        lendingLimit: 16        # 最多借给别人多少
  cohort: gpu-cohort             # 同 cohort 的队列之间可以互相借用
---
apiVersion: kueue.x-k8s.io/v1beta2
kind: LocalQueue
metadata:
  name: team-a
  namespace: team-a
spec:
  clusterQueue: gpu-cluster-queue
```

**借用机制的代价**：借来的配额可以被原主"收回"（抢占）。所以生产任务要放在 `nominalQuota` 内，开发任务才用借用额度。

---

## 旋钮三：回填与预留

### Backfill：小任务填缝

大任务等资源时，调度器允许小任务插队——**前提是小任务能在"大任务预计开始时间"之前跑完**。

```bash
# 开启 backfill（slurm.conf）
# SchedulerType=sched/backfill
# SchedulerParameters=bf_max_job_start=10,bf_window=1440
```

**判据**：

```
小任务墙钟 ≤ 大任务预计开始时间 − 当前时间
```

否则小任务会把大任务往后推。实际踩坑最多的是**开发任务没设 `MaxWall`**——一个"跑 5 分钟"的 notebook 占了 8 小时，大任务排到明天。

### Reservation：给截止日期留资源

```bash
# 为某个时间窗预留 64 卡
scontrol create reservation starttime=2026-11-20T09:00:00 \
  duration=24:00:00 nodes=node[001-008] user=team-a
```

预留会直接降低利用率。只该用在**有硬截止日期**的场景（发布会、比赛、审计），不要当成常规手段。

---

## 旋钮四：GPU 共享

这是最容易被忽略、但对推理和开发场景收益最大的一层。

| 模式 | 隔离维度 | 典型用途 | Slurm | K8s |
|---|---|---|---|---|
| **整卡** | 完整 | 大训练 | 默认 | 默认 |
| **MIG** | SM / 显存 / L2 / 带宽（硬件分区） | 推理、小模型、多租户 | GRES 按 MIG profile 申请 | device plugin / DRA |
| **MPS** | SM 共享；显存需手动设上限，无强隔离 | 多个小 kernel、开发 | 无原生调度，需节点侧起 MPS daemon | 需 MPS daemon |
| **Time-slicing** | 时间片，**无隔离** | 开发调试 | 无原生 | device plugin `replicas` |

### MIG 的边界

```bash
# 查看可用 profile（H100 80GB 支持 7 个实例，profile 形如 1g.10gb）
nvidia-smi mig -lgip
nvidia-smi mig -cgi 1g.10gb -C     # 创建实例
nvidia-smi mig -lgi                # 查看已创建的实例
```

- **优点**：显存和算力是硬隔离的，一个租户 OOM 不会影响另一个；
- **代价**：每个实例的 SM/带宽都变小；**同一 GPU 上不同 MIG 实例之间不能做 P2P/NVLink 通信**；
- **不适合**：需要大显存或跨卡通信的训练任务；**适合**：推理、小模型、多租户开发。
- K8s 里资源名是 `nvidia.com/mig-1g.10gb`（取决于你的 MIG 策略和 GPU Operator 配置）。

### MPS 与 time-slicing 的坑

- **MPS**：多个进程共享同一张卡的 SM；一个 client 崩溃可能影响同卡其他 client；显存不天然隔离（需要 `CUDA_MPS_PINNED_DEVICE_MEM_LIMIT` 之类的约束）。适合"多个小 kernel 互相填缝"，不适合强隔离多租户。
- **Time-slicing**：只是时间片轮转，**没有显存隔离、没有故障隔离、延迟抖动大**。只适合开发调试，不要用来跑生产推理。

---

## 🛠 实操指南

### 📋 决策指南：你的负载该怎么配？

```
你的主要负载？
├─ 大规模训练（单任务 > 8 卡）→
│   └─ 整卡 + gang + 拓扑感知；不要用 MIG/MPS
│
├─ 推理 / 小模型（单任务 ≤ 1 卡）→
│   ├─ 多租户、要求隔离 → MIG
│   └─ 单租户、追求吞吐 → 整卡多实例 / MPS
│
├─ 开发 / Notebook → time-slicing（可接受隔离缺失）
│
└─ 训练 + 推理混部 →
    ├─ 队列层：Kueue / Slurm QOS 做配额
    ├─ 资源层：推理用 MIG，训练用整卡
    └─ 时间层：白天推理优先，夜间训练 backfill
```

### 🔧 自己动手验：队列健康度

```bash
# 1. 排队时间分布（P50/P95）——比"平均等待"更能说明问题
squeue -h -o "%.M" | sort -n | awk '{a[NR]=$1} END {
  print "P50:", a[int(NR*0.5)], " P95:", a[int(NR*0.95)] }'

# 2. 每个账户的 GPU 占用
squeue -h -o "%a %b" | awk '{g[$1]+=$2} END {for (k in g) print k, g[k]}' | sort -k2 -nr

# 3. 被 backfill 推迟的大任务（看 Reason 与 StartTime）
squeue -t PD -o "%.10i %.10P %.20j %.8u %.10M %.20r %.20S" | grep Resources

# 4. 检查是否有作业没设墙钟（backfill 的常见元凶）
squeue -h -o "%.10i %.20l" | awk '$2 ~ /UNLIMITED/'

# 5. Kueue 队列状态
kubectl get clusterqueue -o wide
kubectl describe workload <name>   # 看是被 quota 还是 preemption 卡住
```

### ⚠️ 常见误区

- **误区**："FairShare 就是平均分"
  - **实际**：FairShare 是按历史用量衰减的加权公平。用量大的组会被压低优先级，但会随时间恢复；完全平均会惩罚"刚提交就被调度"的组，效率更差。

- **误区**："MIG 能让训练更快"
  - **实际**：MIG 是把一张卡切成更小的实例，**单实例的算力和带宽都变小**。它的价值是隔离和多租户，不是加速。

- **误区**："回填总是提高利用率"
  - **实际**：回填任务如果没有墙钟限制，会把大任务无限推迟。回填的前提是"小任务能在窗口内跑完"，不是"有缝就填"。

- **误区**："K8s 的 ResourceQuota 就是公平调度"
  - **实际**：ResourceQuota 是硬上限，不是公平分享，也不支持借用。要排队、配额借用、抢占，需要 Kueue 或同类组件。

- **误区**："time-slicing 可以跑生产推理"
  - **实际**：没有显存隔离、没有故障隔离，上下文切换还会带来延迟抖动。它只适合开发调试。

---

## 📊 底线数字速查

| 指标 | 健康 | 警告 | 危险 |
|---|---|---|---|
| 队列等待 P95 | < 1 个作业时长 | 1-3 个 | > 3 个 |
| 集群利用率 | 70-85% | 60-70% / 85-90% | < 60% / > 90% |
| 未设墙钟的作业占比 | 0 | < 5% | > 10% |
| 单账户 GPU 占用 | < 总配额 | = 配额 | 持续超配额（借用） |
| MIG 实例利用率 | > 50% | 20-50% | < 20% |
| 抢占/回填失败率 | < 5% | 5-15% | > 15% |

---

## 💡 一句话记住

> **"调度好不好，不看能不能提交，看四个旋钮：优先级与公平、配额与限额、回填与预留、GPU 共享。Slurm 的 FairShare 是带衰减的加权公平，不是平均分；MIG 是硬件隔离、MPS 是共享、time-slicing 是分时——训练别用后两个跑生产，推理优先用 MIG。"**

---

## 对后续的铺垫

队列解决了"谁先跑、占多少"。但资源分配好之后，训练到底跑得好不好、有没有被慢节点拖住，属于运维观测要回答的问题。

→ [08-运维观测篇/01-集群监控到底要监控什么](../08-运维观测篇/01-集群监控到底要监控什么.md)

---

> **参考资料**：
> - [Slurm：Multifactor Job Priority](https://slurm.schedmd.com/priority_multifactor.html) — 优先级因子与权重
> - [Slurm：QOS 与 sacctmgr](https://slurm.schedmd.com/sacctmgr.html) — GrpTRES / MaxTRES / TRESBillingWeights
> - [Slurm：GRES 与 MIG](https://slurm.schedmd.com/gres.html) — GPU/MIG 资源请求
> - [Kueue 官方文档](https://kueue.sigs.k8s.io/docs/) — ClusterQueue / cohort / borrowing
> - [NVIDIA MIG 用户指南](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/) — 实例 profile 与限制
> - [NVIDIA MPS 文档](https://docs.nvidia.com/deploy/mps/) — 多进程服务与隔离边界
