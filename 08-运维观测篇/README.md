# 运维观测篇 — 集群跑起来之后

## 本篇章回答的问题

| Q&A | 核心链条 |
|------|---------|
| Q1: 集群监控到底要监控什么？ | 指标分层：MFU 为王，XID/ECC/NVLink 抓健康 |
| Q2: 为什么 GPU 利用率 100% 不一定好？ | GPU-Util 是 kernel 执行时间占比，不是算力利用率 |
| Q3: NCCL 调优到底在调什么？ | 算法/协议/信道 + 用对传输层，auto 优先 |
| Q4: 为什么"慢节点"是大集群头号杀手？ | T_step = max_i(T_i)，灰节点难定位 |
| Q5: 训练断了怎么办？ | Checkpoint 状态完整性 + 频率 + 原子性 + 弹性恢复 |
| Q6: Profiling 工具怎么用？ | Nsight Systems → Nsight Compute → PyTorch Profiler |

**核心因果链**：监控发现异常（GPU-Util 假忙 / MFU 低）→ 归因到通信与慢节点（NCCL 调优 + Straggler 检测）→ 故障终会发生（Checkpoint 恢复 + 弹性训练）→ 持续定位根因（Profiling 工具）

## 阅读建议

- 按 01 → 06 顺序读：先建立"看什么指标"的框架，再逐层下钻到通信、慢节点、恢复和 Profiling。
- 动手优先：02 的 StepTimer、03 的 nccl-tests 解析、04 的定位流程、05 的 DCP 示例都可以直接搬进自己的训练代码。
- 数字口径全书统一（MFU、checkpoint 大小、开销 1-5% 等），详见根目录 [FACTS.md](../FACTS.md)。
- 前置知识：并行策略（[05-并行策略篇](../05-并行策略篇/README.md)）与网络基础（[03-网络篇](../03-网络篇/README.md)）；读 05 前建议先过一遍 [06-训练框架篇](../06-训练框架篇/README.md) 中的 FSDP/ZeRO 分片概念。
