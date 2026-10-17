# 调度篇 — 谁来决定 GPU 归谁用

## 本篇章回答的问题

| Q&A | 核心链条 |
|------|---------|
| [Q1: 为什么需要资源调度器？](01-为什么需要资源调度器.md) | 手工分配的混乱 → 资源抽象/排队/隔离/拓扑感知 → 经验指标区间 |
| [Q2: Slurm 为什么是 HPC 的传统选择？](02-Slurm为什么是HPC传统选择.md) | 整块资源分配 + 公平分享 + 拓扑配置，65,536 节点验证规模，GRES/cgroup/`--gpus-per-node` 实战 |
| [Q3: Kubernetes 为什么在 AI 场景崛起？](03-Kubernetes为什么在AI场景崛起.md) | 端到端流水线 → 容器/弹性/自愈（自愈≠训练自愈，需 checkpoint）→ gang/拓扑/网络三大挑战 |
| [Q4: NVIDIA 收购 SchedMD 意味着什么？](04-NVIDIA收购SchedMD意味着什么.md) | 2025-12-15 收购 → Slinky（2024 开源）/SUNK 时间线 → 开源不变、方向需观察 |
| [Q5: Gang Scheduling 是什么？](05-Gang%20Scheduling.md) | K8s 的 gang = 全组原子分配；Slurm 的 gang = 时间片轮转——两个概念分开写 |
| [Q6: 拓扑感知调度为什么重要？](06-拓扑感知调度.md) | NVLink（900 GB/s 双向）vs PCIe 5.0（128 GB/s 双向）≈7× → 机内拓扑 + 机间 rail/leaf 两层 → `--switches` 与 DRA/亲和 |

## 一句话导读

调度器解决的是"**几百个任务抢几千张卡时，谁先用、用哪些卡、怎么不打架**"。这一章按"为什么需要 → 两大阵营（Slurm/K8s）→ 两者融合（Slurm on K8s）→ 两个关键机制（gang scheduling、拓扑感知）"的顺序展开，全部命令和 YAML 均按真实 Slurm/Kubernetes 语法核对过。

## 前置知识

- 了解数据并行 / 张量并行 / 流水线并行，知道"8 卡一个 NVLink 域"意味着什么（[05-并行策略篇](../05-并行策略篇/README.md)）
- 了解 NCCL All-Reduce 的通信路径（[03-网络篇](../03-网络篇/README.md)）
- 知道什么是 checkpoint（[06-训练框架篇](../06-训练框架篇/README.md)）

## 阅读建议

- **只想要选型结论**：读 Q1 的决策树 + Q2/Q3 各自的"底线数字速查"表即可。
- **要动手搭集群**：精读 Q2 的 Slurm 作业脚本（`srun` 启动 `torchrun`）、Q3 的多 Pod 编排与网络方案、Q5 的 Volcano/Kueue 写法、Q6 的 `--switches` 与标签亲和示例。
- **注意两个易混概念**：K8s 的 gang scheduling（全组原子分配）≠ Slurm 的 gang scheduling（时间片轮转），Q5 开头专门做了区分；PCIe/NVLink/IB 带宽全书默认说双向，单向必须标注（Q6 开头）。
