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
信道（NCCL_NCHANNELS）：开多少条并发的通信流？
```

### 算法选择：Ring vs Tree vs NVLS / CollNet

先纠正一个常见误解：**NCCL 的默认算法选择是 auto**——NCCL 自己根据消息大小、拓扑和传输介质（NVLink/网络）在 Ring、Tree、NVLS、CollNet 之间选择，并不是"默认 Ring"。手动指定 `NCCL_ALGO` 只在你明确知道自己在做什么时才用。

**Ring（环状算法）**：

```
GPU0 → GPU1 → GPU2 → GPU3 → ... → GPU0

特点：
  - 每个 GPU 只和邻居通信
  - 带宽利用率高，适合大消息
  - 总时延随 GPU 数线性增长，小消息/大规模时吃亏
```

**Tree（树状算法）**：

```
         GPU0
       ↗    ↖
    GPU1    GPU2
   ↗   ↖   ↗   ↖
GPU3  GPU4 GPU5 GPU6

特点：
  - 时延 O(log N)，小消息、大规模场景优于 Ring
  - 多级树（Tree + Tree）适合跨机、跨交换机
```

**NVLS（NVLink SHARP）**——机内 NVSwitch 归约：

```
利用 NVSwitch 的 SHARP（在交换芯片内做聚合归约）能力：
  - 归约在 NVSwitch 内部完成，GPU 少收一份数据
  - 带宽效率高、省显存带宽
  - 需要 NVSwitch + Hopper 及以上（H100 是 NVSwitch 3 / NVLink 4）
```

**CollNet（网络内归约，IB SHARP）**——跨机场景：

```
利用 InfiniBand 交换机的 SHARP 能力：
  - 归约在网络交换机内部完成，不占用端到端带宽跑全量数据
  - NCCL 2.6 起支持
  - 需要 IB 交换机开启 SHARP 服务
```

> 注意区分：**CollNet = IB SHARP（跨机、网络内归约）**；**NVLS = NVLink SHARP（机内、NVSwitch 归约）**。两者都不是"NVSwitch 4 新增"——H100 平台本来就是 NVSwitch 3 / NVLink 4，NVSwitch 4 是 Blackwell 世代的东西。

| 算法 | 适合场景 | 时延特性 | 硬件要求 |
|------|---------|---------|---------|
| Ring | 大消息 | 随 GPU 数线性增加 | 无 |
| Tree | 小消息、大规模 | 对数增长 | 无 |
| NVLS | 机内大消息归约 | 低 | NVSwitch + Hopper+ |
| CollNet | 跨机大消息归约 | 低 | IB SHARP |

### 协议选择：Simple vs LL vs LL128

| 协议 | 原理 | 适合场景 |
|------|------|---------|
| **Simple** | 直接通过 NVLink 或 IB 传输 | 通用方案，大消息、IB 跨机常用 |
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
export NCCL_IB_HCA=mlx5_0,mlx5_1            # 指定 IB HCA（可带端口：mlx5_0:1,mlx5_1:1）
export NCCL_SOCKET_IFNAME=eth0              # TCP 带外/引导流量用的网卡（按名字匹配）
export NCCL_NET_GDR_LEVEL=PHB               # GPU Direct RDMA 的 PCI 拓扑要求

# ─── RoCE 部署常调的 IB 细节 ───
export NCCL_IB_GID_INDEX=3                  # RoCE 的 GID index（RoCEv2 常用 3）
export NCCL_IB_TC=106                       # 流量等级（Traffic Class），配合交换机 ECN
export NCCL_IB_SL=0                         # 服务等级（IB 场景）
export NCCL_IB_MTU=4096                     # MTU（RoCE 场景要和交换机配置一致）

# ─── 拓扑与调试辅助 ───
export NCCL_TOPO_FILE=/path/to/topo.xml     # 显式指定拓扑 XML（自动探测不准时用）
export NCCL_GRAPH_DUMP_FILE=/tmp/graph.xml  # 把 NCCL 自动生成的通信图 dump 出来排查

# ─── 算法 ───
# NCCL_ALGO 一般不需要手动指定，默认 auto，让 NCCL 按消息大小和拓扑自动选
# 如果想强制：NCCL_ALGO=Ring / Tree / NVLS / CollNet（需硬件支持）

# ─── Buffer ───
export NCCL_BUFFSIZE=8388608                # 默认 4MB；IB 跨机大消息训练常调到 8MB+
```

> **不要把上面这些当"万能加速开关"。** 它们的作用是**让 NCCL 正确地找到并使用硬件**（对的网卡、对的 HCA、对的无损网络配置），而不是"设了就更快"。没部署 RoCE 就别设 `NCCL_IB_TC`；自动探测正常就别设 `NCCL_TOPO_FILE`。

### 框架侧的超时与错误处理（PyTorch）

```python
# NCCL 本身没有 NCCL_TIMEOUT 这个变量（不存在，别设）。
# 算子级超时由框架控制：
import datetime, torch
torch.distributed.init_process_group(
    backend="nccl",
    timeout=datetime.timedelta(minutes=30),   # 集体通信超时
)
```

```bash
# PyTorch 的 NCCL 相关控制变量（注意前缀是 TORCH_，不是 NCCL_）：
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1    # 异步错误处理；现代 PyTorch 用 TORCH_ 前缀
                                            # （旧变量 NCCL_ASYNC_ERROR_HANDLING 已废弃；
                                            #  新版本默认值为 3，即 SkipCleanUp 语义）
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800  # watchdog 心跳超时（秒），配合 Flight Recorder 定位 hang
```

```bash
# NCCL 自己保留的超时变量只有 IB 传输层的：
export NCCL_IB_TIMEOUT=18                   # IB RC 传输超时（IB 协议参数，一般不用动）
```

### 调试变量（不出问题不要开）

```bash
export NCCL_DEBUG=WARN         # INFO 只有在诊断时才用（额外日志有开销，生产保持 WARN）
export NCCL_DEBUG_SUBSYS=INIT  # 只看初始化阶段
export NCCL_DEBUG_FILE=/tmp/nccl_%h_%p.log  # 日志写文件（%h=主机名 %p=进程号），多机排障时必备
```

---

## 🛠 实操指南

### 📋 决策指南：NCCL 调优步骤

```
第一步：跑 nccl-tests 基准测试
├─ 只有知道"理论上该多快"，才知道"实际上慢了多少"
├─ all_reduce_perf -b 8 -e 8G -f 2 -g 8
└─ 参考：8×H100 NVLink = 460-480 GB/s busbw

第二步：如果基准测试慢了 → 检查基础配置
├─ NVLink 链路是否正常？ → nvidia-smi nvlink -s（速率）/ -e（错误）
├─ NCCL 是否用了 IB？ → NCCL_DEBUG=INFO 看 "NET/IB" 字样
├─ GDR 是否开启？ → NCCL_DEBUG=INFO 看 "GDR" 相关行
└─ NCCL 选择了什么算法？ → 看 INFO 日志里的算法/协议选择输出

第三步：如果基准测试正常但训练慢 → 应用层问题
├─ 计算/通信比是否失衡？
├─ 是否可以做通信和计算重叠？
└─ 是否可以用 FSDP/ZeRO 减少通信量？

第四步：如果训练不稳定，经常挂 → 错误处理
├─ TORCH_NCCL_ASYNC_ERROR_HANDLING 开启
├─ init_process_group(timeout=...) 设够大
├─ TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC 配合 watchdog
└─ 看 NCCL_DEBUG=WARN 的输出
```

### 🔧 自己动手验：完整的 NCCL 调优流程

```bash
# 1. 先跑基准测试，建立基线（输出前加 hostname，多机时能对号入座）
all_reduce_perf -b 8 -e 8G -f 2 -g 8 | \
  awk -v h="$(hostname)" 'NR>2 && $1 ~ /^[0-9]+$/ {print h, $1, $(NF-1)}'
# 列结构：size  count_type  redop  root  time  algbw  busbw  wrong
#   $1 = 消息大小，$(NF-1) = busbw（倒数第二列；最后一列是 #wrong，别取错！）
#   $(NF-2) 是 algbw。busbw 才是多卡带宽的正确口径（≈ algbw × 2(N-1)/N）

# 或者干脆用 CSV 输出，按列名解析：
all_reduce_perf -b 8 -e 8G -f 2 -g 8 --csv | \
  awk -v h="$(hostname)" -F, 'NR==1{for(i=1;i<=NF;i++)h2[$i]=i;next}
    $1 ~ /^[0-9]+$/ {print h, $1, $h2["busbw"]}'

# 2. 检查 NCCL 的配置
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT \
  python -c "import torch; torch.distributed.init_process_group('nccl')" 2>&1 | \
  grep -E "NCCL version|NCCL_ALGO|NCCL_PROTO|IB_HCA|GDR"

# 3. 确认 NCCL 用的是 IB 不是 Socket
# 如果看到 "NET/Socket" → 降级到 TCP socket 传输了，跨机带宽会低一个数量级
# 应该看到 "NET/IB"

# 4. 逐个检查环境变量的影响（每次只改一个）
for algo in Ring Tree NVLS; do
    NCCL_ALGO=$algo all_reduce_perf -b 128M -e 128M -f 2 -g 8 | \
      awk 'NR>2 && $1 ~ /^[0-9]+$/ {print "algo='"$algo"'", $1, $(NF-1)}'
done

# 5. 排查挂死/超时
# NCCL 是纯用户态库，dmesg 里查不到 NCCL 日志！正确的姿势：
NCCL_DEBUG=INFO NCCL_DEBUG_FILE=/tmp/nccl_%h_%p.log python train.py
# dmesg 的正确用途：看 Xid、PCIe AER、IB 内核驱动（mlx5）报错
dmesg | grep -iE "xid|aer|mlx5"
```

### ⚠️ 常见误区

- **误区**："NCCL 环境变量设置得越多越好"
  - **实际**：**建议只设置 5-10 个关键变量，其他的让 NCCL 自动选择**。乱设变量可能互相冲突，反而降低性能。NCCL 官方的建议是"从默认值开始，逐一修改，每次只改一个"。`NCCL_NTHREADS` 这类内部参数**新版 NCCL 已不推荐用户设置**——网上流传的"H100 用 512"属于过时配方，设错可能导致性能异常甚至崩溃，保持默认即可。

- **误区**："NCCL_DEBUG=INFO 可以一直开着"
  - **实际**：INFO 级别的日志有可观开销，还会淹没真正有用的行。只在诊断问题时使用，正常运行设为 WARN。

- **误区**："NCCL 默认就是 Ring，我不用管算法"
  - **实际**：**默认是 auto**——NCCL 按消息大小、拓扑（NVLink/IB）、硬件能力（NVLS/CollNet）自动选择。手动指定只在排障对比时用。

- **误区**："NCCL 挂了先看 dmesg"
  - **实际**：**NCCL 是用户态库，内核日志里没有它的输出**。看 NCCL 自己的日志要用 `NCCL_DEBUG=INFO` + `NCCL_DEBUG_FILE`。`dmesg` 的角色是查**内核层**的错误：Xid、PCIe AER、mlx5 驱动报错。

- **误区**："NCCL 调优只在 HPC 集群上需要"
  - **实际**：**云 GPU 集群上 NCCL 调优更重要**。云网络的抖动更大、拓扑更复杂、NCCL 更可能选错传输（掉到 Socket）。云端慢节点的比例也更高。

### 📊 底线数字速查

**NCCL All-Reduce 性能参考（busbw）**：

| 配置 | 期望 busbw | 慢于这个值说明有问题 |
|------|-----------|-------------------|
| 8×H100 SXM5（NVLink） | 460-480 GB/s | < 370 GB/s |
| 8×A100 SXM4（NVLink） | 230-250 GB/s | < 185 GB/s |
| 8×H100 PCIe | 35-45 GB/s | < 28 GB/s |
| 2 节点 H100 + NDR 400 IB | 350-380 GB/s | < 280 GB/s |
| 16 节点 H100 + NDR 400 | 240-280 GB/s | < 190 GB/s |

**环境变量速查（全部真实存在）**：

| 变量 | 建议值 | 说明 |
|------|--------|------|
| NCCL_IB_DISABLE | 0 | 开 IB/RoCE |
| NCCL_IB_HCA | 指定具体 HCA | 排除不该用的网卡 |
| NCCL_IB_GID_INDEX | 3（RoCE 常用） | RoCEv2 的 GID index |
| NCCL_IB_TC / NCCL_IB_SL | 106 / 0（示例） | RoCE 流量等级 / IB 服务等级 |
| NCCL_IB_MTU | 4096 | 与交换机配置一致 |
| NCCL_IB_TIMEOUT | 默认 | IB RC 传输超时，一般不动 |
| NCCL_SOCKET_IFNAME | 指定网卡名 | 带外/引导流量走对网卡 |
| NCCL_TOPO_FILE | 按需 | 显式拓扑 XML |
| NCCL_GRAPH_DUMP_FILE | 按需 | dump 通信图排查 |
| TORCH_NCCL_ASYNC_ERROR_HANDLING | 1（现代 PyTorch） | 异步错误处理；旧前缀 NCCL_ 已废弃 |
| TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC | 秒数 | watchdog 心跳超时 |
| NCCL_DEBUG | WARN（生产） | INFO 只诊断用 |
| NCCL_DEBUG_FILE | 路径模板 | 日志落盘，多机排障 |
| NCCL_NTHREADS | **不设** | 新版 NCCL 不推荐用户设置 |

> 环境变量里**不存在** `NCCL_TIMEOUT`；进程级/算子级超时分别用 `init_process_group(timeout=...)` 和 `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC`。

### 💡 一句话记住

> **"NCCL 调优其实就三步：先跑 nccl-tests 知道'理论上该多快'（busbw 在倒数第二列，别 grep 错）；然后检查 NCCL 有没有正确使用 IB/NVLink（掉到 Socket 传输性能会掉一个数量级）；最后把框架侧的异步错误处理和超时配置好。环境变量是'纠错工具'不是'加速开关'——默认 auto 的算法选择，多数时候比你手动的直觉准。"**

---

## 对后续的铺垫

NCCL 调优解决了"通信不高效"的问题。但还有一个更难发现、也更致命的问题：**慢节点（Straggler）。**

一个慢节点就能拖慢整个 All-Reduce——因为 NCCL 的同步通信要求所有 GPU 都完成才能继续。一个卡慢了，几百张卡一起等。

→ [[04-为什么说慢节点是大集群的头号杀手]]

---

> **参考资料**：
> - [NCCL 用户指南：环境变量](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html) — NVIDIA
> - [nccl-tests GitHub](https://github.com/NVIDIA/nccl-tests) — NVIDIA
> - [NCCL Inspector：AI 工作负载的通信可观测性](https://developer.nvidia.com/blog/enhancing-communication-observability-of-ai-workloads-with-nccl-inspector/) — NVIDIA Technical Blog
> - [PyTorch NCCL 错误处理与 Flight Recorder 文档](https://docs.pytorch.org/docs/stable/distributed.html#nccl-debugging-tools) — PyTorch
