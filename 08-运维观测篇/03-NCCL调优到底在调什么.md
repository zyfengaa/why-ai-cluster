# Q: NCCL 调优到底在调什么？

## 为什么会有这个问题

你输入 `torchrun --nproc_per_node=8 train.py`，NCCL 自动初始化，训练启动了。一切看起来不错——直到你发现吞吐比别人低 30%。

有人告诉你："调一下 NCCL 环境变量就好了。"

于是你打开 NVIDIA 的 NCCL 文档，看到 **40+ 个环境变量**——NCCL_ALGO、NCCL_PROTO、NCCL_BUFFSIZE、NCCL_NTHREADS、NCCL_MIN_NCHANNELS……每个变量 3-5 种取值。

**NCCL 调优到底在调什么？**

## 解决什么问题

NCCL（NVIDIA Collective Communications Library）调优的核心目标只有一个：**让多 GPU 之间的集体通信（All-Reduce、AllGather、ReduceScatter）花的时间最短。**

### NCCL 通信的三要素

```
算法（NCCL_ALGO）：怎么切分数据来通信？
协议（NCCL_PROTO）：用什么方式传输数据？
信道（NCCL_MIN_NCHANNELS / NCCL_MAX_NCHANNELS）：开多少条并发的通信流？
```

### 算法选择：Ring vs Tree vs NVLS / CollNet

**Ring（环状算法）**：

```
GPU0 → GPU1 → GPU2 → GPU3 → ... → GPU0

特点：
  - 每个 GPU 只和邻居通信
  - 带宽利用率最高（接近 100%），**大消息**下最优
  - 延迟随 GPU 数线性增长
  - NCCL 默认算法之一
```

**Tree（树状算法）**：

```
         GPU0
       ↗    ↖
    GPU1    GPU2
   ↗   ↖   ↗   ↖
GPU3  GPU4 GPU5 GPU6

特点：
  - **小消息（≲ 256 KB）**下优于 Ring，延迟随 GPU 数只做对数增长
  - 在跨机场景延迟更低
  - 适合跨 leaf 交换机的通信
```

**NVLS（NVLink SHARP）**——第三代 NVSwitch（Hopper/H100 代）新增：

```
利用 NVSwitch 内建的 SHARP（可扩展分级聚合归约）能力：
  - 归约在交换机内部完成
  - GPU 不需要参与聚合
  - 带宽最高，延迟最低
  - 需要第三代 NVSwitch（Hopper/H100 代）
```

> 别和 **CollNet** 混淆：CollNet 是把归约卸载到**支持 SHARP 的 InfiniBand 交换机**上（跨机网内归约，由 `NCCL_COLLNET_ENABLE` 控制），属于网络特性；NVLS 则是 **NVSwitch（NVLink 域内）** 的 SHARP，属于机内特性。两者都叫"网内归约"，但硬件载体完全不同。

| 算法 | 适合规模 | 带宽效率 | 延迟 | 硬件要求 |
|------|---------|---------|------|---------|
| Ring | 任意规模，大消息最优 | 最高（接近 100%） | 随 GPU 数线性增长 | 无 |
| Tree | 任意规模，小消息（≲ 256 KB）最优 | 低于 Ring | 随 GPU 数对数增长 | 无 |
| NVLS | 单机 NVLink 域内 | 最高（网内归约） | 极低 | NVLS / 第三代 NVSwitch（Hopper/H100 代）|
| CollNet | 跨机（IB） | 高（网内归约） | 低 | 支持 SHARP 的 InfiniBand 交换机 |

### 协议选择：Simple vs LL vs LL128

| 协议 | 原理 | 适合场景 |
|------|------|---------|
| **Simple** | 直接通过 NVLink 或 IB 传输 | 通用方案，IB 跨机选这个 |
| **LL**（Low-Latency）| 使用少量数据包降低延迟 | 小数据量（< 1 MB）|
| **LL128** | 128 字节对齐的低延迟协议 | NVLink 域内传输 |

```bash
# 经验法则
# IB 跨机：NCCL_PROTO=Simple
# NVLink 域内：NCCL_PROTO=LL128（默认自动选）
# 小消息：NCCL_PROTO=LL
```

---

## 核心调优变量

### 必设变量（生产环境基线）

```bash
# ─── 传输层 ───
export NCCL_IB_DISABLE=0                    # 开启 InfiniBand/RoCE
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1       # 指定 IB HCA（前缀 "=" 表示精确匹配，如 =mlx5_0:1；留空则自动探测全部 HCA）
export NCCL_SOCKET_IFNAME=eth0              # 引导接口；IB 环境建议留空让 NCCL 自动选择（会优先 ib*），强行写 eth0 反而可能更差
export NCCL_NET_GDR_LEVEL=5                 # 允许跨 NUMA/SYS 距离使用 GDR（>4 一律按 SYS 解释；无固定默认值，NCCL 按架构自选）

# ─── 算法 ───
# NCCL_ALGO 一般不需要手动指定，让 NCCL 自动选
# 如果想强制 Ring：NCCL_ALGO=Ring
# 如果想强制 Tree：NCCL_ALGO=Tree

# ─── Buffer 和并行度 ───
export NCCL_BUFFSIZE=16777216               # 16MB（默认 4MB）
export NCCL_NTHREADS=512                    # 通信线程数（H100/NVLink5 建议 512）
export NCCL_MIN_NCHANNELS=4                 # 最小通信信道数（自 NCCL 2.17 起废弃，建议改用 NCCL_MIN_CTAS）

# ─── 超时和错误处理 ───
export TORCH_NCCL_ASYNC_ERROR_HANDLING=3    # 异步错误处理，默认值 3=teardown（最严格）
# 注意：NCCL_ASYNC_ERROR_HANDLING 已废弃，PyTorch 会告警提示改用 TORCH_NCCL_ASYNC_ERROR_HANDLING；
# 把默认的 3 改成 1 其实是"削弱"了默认行为，不再是"必设"。
#
# 超时不是 NCCL 的环境变量，而是 PyTorch 进程组的参数（默认即 1800 s）：
#   from datetime import timedelta
#   dist.init_process_group("nccl", timeout=timedelta(seconds=1800))
```

### 调试变量（不出问题不要开）

```bash
export NCCL_DEBUG=WARN         # INFO 只在诊断时才用（日志量大，建议重定向到文件）
export NCCL_DEBUG_SUBSYS=INIT  # 只看初始化阶段
```

### RoCE 场景的 QoS（PFC / ECN / TC）

跑 RoCE 时，除了上面的变量，还必须配置 **TC（Traffic Class）与无损网络参数**——这是 RoCE 相比 IB 最容易被漏掉的一项：

```bash
export NCCL_IB_TC=106      # RoCE 流量打到的 TC（必须和交换机上的 lossless 队列一一对应）
export NCCL_IB_SL=0        # IB 的 Service Level
```

- **PFC（Priority Flow Control）**：按优先级做链路级流控，防止 RoCE 丢包。
- **ECN（Explicit Congestion Notification）**：端到端拥塞标记，比 PFC 更平滑，避免 PFC 风暴。
- TC 两边（主机与交换机）必须一致，否则"无损"承诺失效，RoCE 会掉速。

---

## 🛠 实操指南

### 📋 决策指南：NCCL 调优步骤

```
第一步：跑 nccl-tests 基准测试
├─ 只有知道"理论上该多快"，才知道"实际上慢了多少"
├─ all_reduce_perf -b 8 -e 8G -f 2 -g 8
└─ 参考：8×H100 NVLink = 460-480 GB/s busbw

第二步：如果基准测试慢了 → 检查基础配置
├─ NVLink 是否全 Active？ → nvidia-smi nvlink -s
├─ NCCL 是否用了 IB？ → NCCL_DEBUG=INFO 看 "NET/IB" 字样
├─ GDR 是否开启？ → NCCL_DEBUG=INFO 看 "GDR: Enabled"
└─ NCCL 选择了什么算法？ → 看 "Ring" 或 "Tree" 输出

第三步：如果基准测试正常但训练慢 → 应用层问题
├─ 计算/通信比是否失衡？
├─ 是否可以做通信和计算重叠？
└─ 是否可以用 FSDP/ZeRO 减少通信量？

第四步：如果训练不稳定，经常挂 → 错误处理
├─ TORCH_NCCL_ASYNC_ERROR_HANDLING=3（默认值，最严格的 teardown）
├─ init_process_group(timeout=timedelta(seconds=1800))（默认即 1800 s）
└─ 看 NCCL_DEBUG=INFO 的输出 + Flight Recorder（见下）
```

### 🔧 自己动手验：完整的 NCCL 调优流程

```bash
# 1. 先跑基准测试，建立基线
all_reduce_perf -b 8 -e 8G -f 2 -g 8 | tee nccl_baseline.log
# 关键看 busbw 行（bus bandwidth，总线带宽）
# 注意区分 algbw 与 busbw：algbw 是"算法视角"的有效吞吐，busbw 是"总线视角"的实际搬运量。
# Ring AllReduce 下每个 GPU 既发又收，等效搬运量是数据量的 2 倍，所以 busbw ≈ 2 × algbw。
# 跨节点带宽对比、跟 NVLink/IB 理论值对比，都要用 busbw，别拿 algbw 去比。
grep "busbw" nccl_baseline.log | tail -5

# 2. 检查 NCCL 的配置
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT \
  python -c "import torch; torch.distributed.init_process_group('nccl')" 2>&1 | \
  grep -E "NCCL_VERSION|NCCL_ALGO|NCCL_PROTO|IB_HCA|GDR"

# 3. 确认 NCCL 用的是 IB 不是 TCP
# 如果看到 "NET/Socket" → 降级到 TCP 了（性能暴跌 10-50×）
# 应该看到 "NET/IB"

# 4. 逐个检查环境变量的影响
# 注意：NCCL_ALGO / NCCL_PROTO 没有名为 "Auto" 的合法取值——
# "让 NCCL 自动选" = 不设置该变量（写成 Auto 会被当作无效值）。
for algo in unset Ring Tree; do
    for proto in unset Simple LL128; do
        [ "$algo" = unset ] && unset NCCL_ALGO || export NCCL_ALGO=$algo
        [ "$proto" = unset ] && unset NCCL_PROTO || export NCCL_PROTO=$proto
        all_reduce_perf -b 128M -e 128M -f 2 -g 8 | grep "busbw" | tail -1
    done
done

# 5. 训练 hang 住了怎么排查？
#    a) 看进程组超时：init_process_group(timeout=timedelta(seconds=1800))，
#       超时会抛 "Watchdog caught collective operation timeout" 之类的错误，
#       并在栈里指出卡在哪一步 collective（哪个 rank 没到齐）。
#    b) 用 NCCL_DEBUG=INFO 的日志逐段看：先看初始化段（选了哪个算法/协议、
#       走 IB 还是 Socket、GDR 是否开启），再看 collective 段，判断是初始化卡住
#       还是某个 collective 卡住。
#    c) 用 NCCL Flight Recorder（Meta 公开推荐，定位 hang 的标准工具）：
export TORCH_NCCL_DEBUG_INFO_DUMP=/tmp/nccl_fr_
#       它把 NCCL 最近的状态存进环形缓冲区，超时/崩溃时一次性落盘，
#       比事后才开 INFO 抓现场更可靠。
# 提醒：NCCL 是用户态库，不往 dmesg 写日志；要看 NCCL 的报错请用上面的 INFO 输出与 Flight Recorder。

# 6. 导出拓扑，确认 NCCL 看到的链路符合预期
NCCL_TOPO_DUMP_FILE=/tmp/nccl_topo.xml NCCL_DEBUG=INFO \
  all_reduce_perf -b 8 -e 8G -f 2 -g 8
# 生成的 nccl_topo.xml 描述 NCCL 探测到的 GPU/NIC/PCIe/NVLink 拓扑；
# 也可以用 NCCL_TOPO_FILE 指定一份手写拓扑文件覆盖自动探测结果（常见于容器/虚拟化环境）。
```

### ⚠️ 常见误区

- **误区**："NCCL 环境变量设置得越多越好"
  - **实际**：**建议只设置 5-10 个关键变量，其他的让 NCCL 自动选择**。乱设变量可能互相冲突，反而降低性能。NCCL 官方的建议是"从默认值开始，逐一修改，每次只改一个"。

- **误区**："NCCL_DEBUG=INFO 可以一直开"
  - **实际**：INFO 级别的日志量很大，需要重定向到文件（否则会淹没训练日志）。只在诊断问题时使用，正常运行设为 WARN。

- **误区**："NCCL_ALGO=Ring 总是最好的"
  - **实际**：**取舍按消息大小**，不是按 GPU 数——小消息（≲ 256 KB）Tree 更优，大消息 Ring 带宽最高。不需要手动指定，让 NCCL 自动选即可。

- **误区**："NCCL 调优只在 HPC 集群上需要"
  - **实际**：**云 GPU 集群上 NCCL 调优更重要**。云网络的抖动更大、拓扑更复杂、NCCL 更可能错误地选择了 TCP 回退。云端慢节点的比例也更高。

### 📊 底线数字速查

**NCCL All-Reduce 性能参考（busbw）**：

| 配置 | 期望 busbw | 慢于这个值说明有问题 |
|------|-----------|-------------------|
| 8×H100 SXM5（NVLink） | 460-480 GB/s | < 370 GB/s |
| 8×A100 SXM4（NVLink） | 230-250 GB/s | < 185 GB/s |
| 8×H100 PCIe | 35-45 GB/s | < 28 GB/s |
| 2 节点 H100 + NDR 400 IB | 350-380 GB/s | < 280 GB/s |
| 16 节点 H100 + NDR 400 | 240-280 GB/s | < 190 GB/s |

**环境变量速查**：

| 变量 | 建议值 | 说明 |
|------|--------|------|
| NCCL_IB_DISABLE | 0 | 开 IB/RoCE |
| NCCL_IB_HCA | 指定具体 HCA | 前缀 "=" 表示精确匹配；留空则自动探测 |
| NCCL_NET_GDR_LEVEL | 5（=SYS）| 允许跨 NUMA/SYS 距离用 GDR；无固定默认值 |
| TORCH_NCCL_ASYNC_ERROR_HANDLING | 3（默认）| 已取代废弃的 NCCL_ASYNC_ERROR_HANDLING |
| init_process_group(timeout=...) | 1800 s | PyTorch 进程组超时，默认即 1800 s（非 NCCL 变量）|
| NCCL_BUFFSIZE | 16777216 | 16MB |
| NCCL_NTHREADS | 512（H100）/ 256（A100）| 通信线程 |
| NCCL_DEBUG | WARN（生产）| INFO 只诊断用 |

### 💡 一句话记住

> **"NCCL 调优其实就三个步骤：先跑 nccl-tests 知道'理论上该多快'，然后检查 NCCL 有没有正确使用 IB 和 GDR（TCP 回退会让性能暴跌 50 倍），最后确认异步错误处理与超时设置（防止训练莫名其妙挂掉）。NCCL_ALGO / NCCL_PROTO 默认交给 NCCL 自动选通常就不错，但按 collective 定向覆盖（如 `NCCL_PROTO="^LL128;allreduce:LL128"`）和用 nccl-tests 扫参，都是官方支持的标准动作——先建立基线，再有针对性地调。"**

---

## 对后续的铺垫

NCCL 调优解决了"通信不高效"的问题。但还有一个更难发现、也更致命的问题：**慢节点（Straggler）。**

一个慢节点就能拖慢整个 All-Reduce——因为 NCCL 的同步通信要求所有 GPU 都完成才能继续。一个卡慢了，几百张卡一起等。

→ [[04-为什么说慢节点是大集群的头号杀手]]

---

> **参考资料**：
> - [NCCL 官方文档（User Guide）](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/) (NVIDIA, 2026) — **首要来源**
> - [NCCL 环境变量全表（Environment Variables）](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html) (NVIDIA)
> - [NCCL-tests 基准测试工具](https://github.com/NVIDIA/nccl-tests) (NVIDIA)
> - [PyTorch NCCL 专用环境变量（TORCH_NCCL_*）](https://docs.pytorch.org/docs/stable/torch_nccl_environment_variables.html) (PyTorch)
> - [NCCL Tuning for Multi-GPU LLM Training: Environment Variables, Topology, and Cloud Guide (2026)](https://www.spheron.network/blog/nccl-tuning-multi-gpu-llm-training-2026/) — Spheron（厂商营销博客，benchmark 为自报数据，仅供参考）(2026)
> - [NCCL-tests in practice — the mandatory pre-launch health check](https://docs.alayanew.com/en/docs/tech-blog/2026-05-06-nccl-tests) — Alya (2026)
> - [NCCL Debugging & Tuning: NCCL_ALGO (Ring, Tree, CollNet)](https://rajatpandit.com/ai-engineering/debugging-nccl-ring-failures/) — Rajat Pandit (2026)
> - [NCCL Topology Dump and Tuning on Kubernetes](https://kubernetes.recipes/recipes/ai/nccl-topology-dump-tuning-kubernetes/) — K8s Recipes (2026)
