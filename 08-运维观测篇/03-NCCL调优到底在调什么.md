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

### 算法选择：Ring vs Tree vs CollNet

**Ring（环状算法）**：

```
GPU0 → GPU1 → GPU2 → GPU3 → ... → GPU0

特点：
  - 每个 GPU 只和邻居通信
  - 小规模（≤ 32 GPU）时最优
  - 带宽利用率高（接近 100%）
  - NCCL 默认算法
```

**Tree（树状算法）**：

```
         GPU0
       ↗    ↖
    GPU1    GPU2
   ↗   ↖   ↗   ↖
GPU3  GPU4 GPU5 GPU6

特点：
  - 大规模（> 32 GPU）时优于 Ring
  - 在跨机场景效率更高
  - 适合跨 leaf 交换机的通信
```

**CollNet（全局归约网络）**——NVSwitch 4 代新增：

```
利用 NVSwitch 的 SHARP（可扩展分级聚合归约）能力：
  - 通信在交换机内部完成
  - GPU 不需要参与聚合
  - 带宽最高，延迟最低
  - 需要 NVSwitch 4 + H100+
```

| 算法 | 适合规模 | 带宽效率 | 延迟 | 硬件要求 |
|------|---------|---------|------|---------|
| Ring | ≤ 32 GPU | 高（~100%） | 随 GPU 数线性增加 | 无 |
| Tree | > 32 GPU | 中（~70%） | 对数增长 | 无 |
| CollNet | 任意规模 | 最高 | 极低 | NVSwitch 4 |

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
export NCCL_IB_HCA=mlx5_0:1,mlx5_1:1       # 指定 IB HCA（不要用默认的 "=all"）
export NCCL_SOCKET_IFNAME=eth0              # TCP 引导接口
export NCCL_NET_GDR_LEVEL=5                 # GPU Direct RDMA（5=full）

# ─── 算法 ───
# NCCL_ALGO 一般不需要手动指定，让 NCCL 自动选
# 如果想强制 Ring：NCCL_ALGO=Ring
# 如果想强制 Tree：NCCL_ALGO=Tree

# ─── Buffer 和并行度 ───
export NCCL_BUFFSIZE=16777216               # 16MB（默认 4MB）
export NCCL_NTHREADS=512                    # 通信线程数（H100/NVLink5 建议 512）
export NCCL_MIN_NCHANNELS=4                 # 最小通信信道数

# ─── 超时和错误处理 ───
export NCCL_ASYNC_ERROR_HANDLING=1          # 异步错误处理（必设！）
export NCCL_TIMEOUT=1800                    # 超时时间（30 分钟）
```

### 调试变量（不出问题不要开）

```bash
export NCCL_DEBUG=WARN         # INFO 只有在诊断时才用（会降低性能 5-15%）
export NCCL_DEBUG_SUBSYS=INIT  # 只看初始化阶段
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
├─ NVLink 是否全 Active？ → nvidia-smi nvlink -s
├─ NCCL 是否用了 IB？ → NCCL_DEBUG=INFO 看 "NET/IB" 字样
├─ GDR 是否开启？ → NCCL_DEBUG=INFO 看 "GDR: Enabled"
└─ NCCL 选择了什么算法？ → 看 "Ring" 或 "Tree" 输出

第三步：如果基准测试正常但训练慢 → 应用层问题
├─ 计算/通信比是否失衡？
├─ 是否可以做通信和计算重叠？
└─ 是否可以用 FSDP/ZeRO 减少通信量？

第四步：如果训练不稳定，经常挂 → 错误处理
├─ NCCL_ASYNC_ERROR_HANDLING=1（必设）
├─ NCCL_TIMEOUT=1800（NCCL_TIMEOUT 设够大）
└─ 看 NCCL_DEBUG=WARN 的输出
```

### 🔧 自己动手验：完整的 NCCL 调优流程

```bash
# 1. 先跑基准测试，建立基线
all_reduce_perf -b 8 -e 8G -f 2 -g 8 | tee nccl_baseline.log
# 关键看 busbw 行
grep "busbw" nccl_baseline.log | tail -5

# 2. 检查 NCCL 的配置
NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT \
  python -c "import torch; torch.distributed.init_process_group('nccl')" 2>&1 | \
  grep -E "NCCL_VERSION|NCCL_ALGO|NCCL_PROTO|IB_HCA|GDR"

# 3. 确认 NCCL 用的是 IB 不是 TCP
# 如果看到 "NET/Socket" → 降级到 TCP 了（性能暴跌 10-50×）
# 应该看到 "NET/IB"

# 4. 逐个检查环境变量的影响
for algo in Ring Tree Auto; do
    for proto in Simple LL128 Auto; do
        NCCL_ALGO=$algo NCCL_PROTO=$proto \
          all_reduce_perf -b 128M -e 128M -f 2 -g 8 | \
          grep "busbw" | tail -1
    done
done

# 5. 检查是否被 NCCL Timeout 坑过
dmesg | grep -i nccl  # 查看内核日志中 NCCL 相关的错误
```

### ⚠️ 常见误区

- **误区**："NCCL 环境变量设置得越多越好"
  - **实际**：**建议只设置 5-10 个关键变量，其他的让 NCCL 自动选择**。乱设变量可能互相冲突，反而降低性能。NCCL 官方的建议是"从默认值开始，逐一修改，每次只改一个"。

- **误区**："NCCL_DEBUG=INFO 可以一直开"
  - **实际**：INFO 级别的日志会带来 **5-15% 的性能损失**。只在诊断问题时使用，正常运行设为 WARN。

- **误区**："NCCL_ALGO=Ring 总是最好的"
  - **实际**：Ring 在小规模（≤ 32 GPU）时最优。但大规模跨机场景（> 32 GPU），**Tree 算法可能更好**。不需要手动指定，让 NCCL 自动选择。

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
| NCCL_IB_HCA | 指定具体 HCA | 不要用 =all |
| NCCL_NET_GDR_LEVEL | 5 | 全开 GDR |
| NCCL_ASYNC_ERROR_HANDLING | 1 | 必设！ |
| NCCL_TIMEOUT | 1800 | 设大不设小 |
| NCCL_BUFFSIZE | 16777216 | 16MB |
| NCCL_NTHREADS | 512（H100）/ 256（A100）| 通信线程 |
| NCCL_DEBUG | WARN（生产）| INFO 只诊断用 |

### 💡 一句话记住

> **"NCCL 调优其实就三个步骤：先跑 nccl-tests 知道'理论上该多快'，然后检查 NCCL 有没有正确使用 IB 和 GDR（TCP 回退会让性能暴跌 50 倍），最后确认 ASYNC_ERROR_HANDLING 设置了（防止训练莫名其妙挂掉）。不要碰 NCCL_ALGO 和 NCCL_PROTO——让 NCCL 自动选，它比你聪明。"**

---

## 对后续的铺垫

NCCL 调优解决了"通信不高效"的问题。但还有一个更难发现、也更致命的问题：**慢节点（Straggler）。**

一个慢节点就能拖慢整个 All-Reduce——因为 NCCL 的同步通信要求所有 GPU 都完成才能继续。一个卡慢了，几百张卡一起等。

→ [[04-为什么说慢节点是大集群的头号杀手]]

---

> **参考资料**：
> - "NCCL Tuning for Multi-GPU LLM Training" — Spheron (2026)
> - "NCCL-tests in practice — mandatory pre-launch health check" — Alya (2026)
> - "NCCL Debugging & Tuning" — Rajat Pandit (2026)
> - NCCL 官方文档 (NVIDIA, 2026)
> - "NCCL Topology Dump and Tuning on Kubernetes" — K8s Recipes (2026)
