# Q: InfiniBand 和 RoCEv2 到底有什么区别？

## 为什么会有这个问题

AI 集群网络的核心技术是 RDMA（Remote Direct Memory Access，远程直接内存访问）。RDMA 允许一台机器直接读写另一台机器的内存（GPU 显存），完全绕过 CPU——这就避免了"CPU 做中转"的延迟。

RDMA 有两种技术路线：

| 方案 | 全称 | 底层硬件 | 一句话 |
|------|------|---------|--------|
| **InfiniBand** | 原生 RDMA 网络 | **专用**交换机和 HCA 卡 | 为 RDMA 而生，一切硬件都为这个目标设计 |
| **RoCEv2** | RDMA over Converged Ethernet | **标准以太网**交换机 + 网卡 | 在普通以太网上"伪装"RDMA |

> 顺带澄清一个常被忽略的点：**RoCE 有两个版本，今天说的 RoCE 默认都是 RoCEv2。** RoCEv1 是二层协议（EtherType 0x8915，不带 IP 头），只能在同一个二层广播域内工作，**不可跨 IP 路由**；RoCEv2 改用 UDP/IP 封装（**UDP 目的端口 4791**），可以跨 IP 网络路由，因而又叫 Routable RoCE。这也是今天大规模 AI 网络只用 RoCEv2 的原因——只有它能组出跨机架、跨三层的大网。

这个区别听起来像"专用工具 vs 通用改造"，但实际上它们的差距比这个名字要深刻得多。

---

## 核心差异：信用流控 vs PFC

### InfiniBand：先确认，再发送（信用流控）

InfiniBand 的流控原理类似于**你在电话里订外卖**：

```
用户（发送方）→ 商家（接收方）：
  "喂，我现在有 4 份菜（数据包）要送，你那里能接多少？"

商家（接收方）→ 用户（发送方）：
  "我现在可以接 3 份，你送吧。"

用户 → 商家：送出 3 份
商家 → 用户：吃完 1 份，释放 1 个位置 → "现在还可以再接 1 份"
[循环，直到所有数据送完]
```

**关键**：发送方从不去做"我猜你能收到"的假设。**每一次发送都基于接收方明确确认的容量。** 这就叫"信用流控"（Credit-Based Flow Control, CBFC）。

结果是：InfiniBand 网络**天生是无损的**。数据包永远不会因为"接收端缓冲区满了"而被丢弃。

### RoCEv2：先发出去，堵了再刹车（PFC）

RoCEv2 没有 InfiniBand 那种"先问后发"的机制。它就像**高速公路上的车流**：

```
正常情况：车流畅通

高峰期：某出口堵了
  出口 1 堵塞 → 车辆排到主路 → 影响出口 2 → 影响出口 3
  交警（PFC）发出 PAUSE 帧："入口暂停放行！"
  结果：所有入口都暂停了，包括不需要去出口 1 的车辆
  
PAUSE 帧传播效果：
  GPU 1 ──→ 交换机 A ──→ 交换机 B ──→ 交换机 C ──→ GPU 2
               ↓
          缓存满了
               ↓
         上游 PAUSE → 传播到 32 个端口
               ↓
         整个集群所有方向都减速
               ↓
         ┌─────────────────────────────────┐
         │      PFC 风暴！                  │
         │  不是只有堵了的那条路受影响       │
         │  整个集群的 RDMA 都降速了         │
         └─────────────────────────────────┘
```

这个机制叫 **PFC（Priority Flow Control，优先级流控）**。它是 RoCEv2 实现"无损"的方式——不是预防丢包，而是**丢包前紧急叫停**。问题是，PFC 的"叫停"会级联扩散，最终影响整个网络。

---

## 一个真实的 PFC 风暴案例

2024 年，国内某云计算公司为其内部大模型团队搭建了一个 2,048 卡 H100 集群，采用 RoCEv2 网络。

**症状**：训练程序每隔 2-3 天就会卡死一次。典型的日志不是 socket 报错，而是通信**整体停滞**——任务长时间无进展，最终触发 **NCCL watchdog 超时**（`Watchdog caught collective operation timeout`）。（顺带说明：`Socket creation failed` 这类报错通常来自 socket 传输层或系统资源问题——比如文件描述符耗尽、路由不可达，和 PFC 风暴没有直接因果关系，排查时别被它带偏。）

**排查过程**：
1. 第一反应是检查 GPU —— 两张卡有问题，换掉，问题依旧
2. 检查光纤 —— 发现 3 根光纤的 RX 光功率偏低（-12 dBm，正常应 > -8 dBm），更换，问题依旧
3. 终于想起来看网络计数器

```
# 通过 ethtool 查看 PFC 计数器（注意：这是累计计数，必须两次采样求差）
ethtool -S eth0 | grep pause
     rx_pause: 4,287,613,941     ← 开网以来的累计收帧数
     tx_pause: 3,196,542,178     ← 开网以来的累计发帧数

# 隔 1 分钟再采一次，两次相减得到增量：
#   增量 = 第 2 次采样 - 第 1 次采样
# 正常运行的集群：增量应在每小时百万级以下
# 这个集群：增量达到每小时数亿次，说明整个网络处于持续的 PFC 风暴中
```

**根因**：存储流量和 GPU 训练流量被分配到了同一个 PFC 优先级队列，存储的突发写入（Checkpoint 触发时）占满了交换机缓存，导致所有 GPU 间的 RDMA 流量都被 PFC 暂停。

**修复**：将存储流量和 GPU 流量分到不同的 PFC 优先级队列，并调整 ECN 标记阈值。修复后 PFC 计数器从每小时数亿次降到百万次级别，训练不再卡死。

但这个故事反映了一个现实：**生产环境中 RoCE 网络的问题往往不是"某个参数错了"，而是"几个参数的组合效应恰好踩到了坑"。**

---

## 关键规格对比（带实际含义）

| 特性 | InfiniBand (NDR) | RoCEv2 | 实际影响 |
|------|-----------------|--------|---------|
| 交换机延迟 | **~130 ns**（Quantum-2 QM9700 端口到端口，Quantum-X800 < 100 ns） | ~300-600 ns（Spectrum-4/SN5600、Tomahawk 5 等现代 DC 交换机，同为直通式） | 单跳差 ~200ns，**10 跳差 ~2μs**，累积在万卡通信中不容忽视 |
| 端到端延迟 | **~0.9 μs** | ~1.5-3 μs（调优后） | All-Reduce 的 barrier 等待时间，直接影响 MFU |
| 丢包恢复 | 硬件重传，微秒级 | go-back-N 重传，典型 RTT 下 **十微秒~毫秒级** | 重传本身不慢；真正的代价是超时/watchdog 被触发后的连锁反应 → 训练崩溃 |
| 拥塞检测 | FECN/BECN 拥塞标记 + 链路级 credit 流控 | 依赖 ECN + DCQCN 手动调参 | 调优周期：IB 即插即用 vs RoCE **2 周+** |
| 多路径 | 自适应路由（自动分到不同路径） | ECMP（按哈希值选路径） | ECMP 哈希冲突 → 多条流挤到同一路径 → 局部拥塞 |
| 单个子网规模 | 受 16-bit LID 限制，约 **48K–49K** 端点 | 受 PFC 风暴可达性限制 | 万卡 IB 需拆成多个子网（或引入 IB 路由器）；万卡 RoCE 也需拆成多个独立子网 |
| 成本（每端口） | 估算 ~$1,500-2,000（400G） | 估算 ~$500-800（400G） | 便宜 50-70%，但运维成本更高（以上为经验估算，非厂商报价） |

> **一个最实质的技术差距：网内归约（SHARP）。** InfiniBand 的 SHARP（Scalable Hierarchical Aggregation and Reduction Protocol）让交换机在转发数据的过程中就地完成 sum/max 等归约运算，All-Reduce 的通信量因此大幅减少；RoCEv2 阵营目前没有等价的标准能力。这是 IB 相对 RoCE 最实在的一条架构性优势。参见 [NVIDIA SHARP 官方介绍](https://developer.nvidia.com/blog/advancing-performance-with-nvidia-sharp-in-network-computing)。

---

## 性能差距怎么看

没有一份"放之四海皆准"的公开基准能同时覆盖 IB 与 RoCE 的各种规模——两者的差距高度依赖拓扑、拥塞控制调优程度和负载模式。可以参考两条真实的公开线索：

- AMD 在 2026 年发布的 *Comparative Analysis of Scale-Out RoCE Network Traffic Patterns and Loads in Training Large Language Models*（ROCm 博客）分析了大规模 RoCE 集群在 LLM 训练下的流量模式，指出**流量具有强突发性、并与集合通信阶段强相关**——这正是拥塞控制难调的根本原因。
- IBM Research 在 UCICS 2026 的 *Designing RoCE Networks for AI Workloads*（Alim Alim）则从设计角度讨论了 RoCE 在 AI 负载下的瓶颈与调优方向。

定性的结论是：**RoCE 调优后可以逼近 IB 的性能，但这份性能不是"开箱即得"，而是持续调优换来的；而调优本身不是免费的。**

---

## 实际调优 RoCE 需要的步骤

一个 512 端口 RoCE 集群的调优清单（下面这份是工程经验归纳，不是 IBM Research 那篇综述的原文；该综述见参考文献）：

```
Week 1: 基础搭建
  - 交换机固件升级（1 天）
  - PFC 队列分配：GPU 流量 → 优先级 3，存储 → 优先级 4，管理 → 0
  - DCQCN 初始参数配置：α、g、速率恢复计时器按网卡/固件型号的厂商默认值起步（各厂商差异很大，不要照搬别人的数字）
  - MTU 设置：以太网侧端口开启 jumbo frame（9000/9014/9216），同时把 RDMA MTU 配成 4096（RDMA MTU 必须小于端口 Jumbo MTU）

Week 2: 迭代调优
  - 跑 nccl-tests all_reduce_perf 观察 busbw
  - 如果 busbw 低于期望值 20%+：
    调整 ECN 标记阈值（在 400G 端口上按交换机缓存占用的百分比起调，而不是照搬千兆端口的 KB 数值）
    调整 DCQCN 的速率恢复速度
  - 在真实训练负载上跑 12h+：
    监控 PFC 计数器（ethtool -S）
    如果 rx_pause 的每小时增量 > 1M（间隔采样、两次求差），继续调
  
Week 3+: 稳定性验证
  - 引入存储流量，检查 PFC 风暴是否触发
  - 模拟链路故障，验证恢复时间
  - 持续监控，建立告警基线
```

一个经验丰富的网络工程师做完这一套大约需要 **2 周**。而 InfiniBand 集群上电后，网络工程师只需要验证子网管理器自动发现拓扑就完成了——**几个小时**。

---

## 到底选哪个？

这是 2026 年业界公认的决策逻辑：

```
你的集群规模：
  < 100 GPU  → RoCE 完全够用（运维简单，省钱）
  100-500 GPU → 混合架构可行（前 RoCE 后 IB）
  > 500 GPU  → 建议 IB（或国产 IB）
  > 10,000 GPU → IB 几乎唯一选择（架构性优势）
  
你的团队：
  有 3+ 资深网络工程师且熟悉 RDMA → 可以挑战 RoCE
  没有 → 买 IB 更省钱（省下的运维成本 > 硬件溢价）
```

---

## 🛠 实操指南

### 📋 决策指南：我怎么选 IB 还是 RoCE？

用这个简单的决策树：

```
你的团队有 3+ 名熟悉 RDMA 的网络工程师吗？
├─ 有 →
│  你的集群 > 500 GPU 吗？
│  ├─ 是 → 选 IB（大规模下 RoCE 的隐性成本 > 硬件差价）
│  └─ 否 → 两种都可以，RoCE 更省钱
└─ 没有 →
   │  你的集群 > 100 GPU 吗？
   ├─ 是 → 选 IB（你付不起 RoCE 的运维成本）
   └─ 否 → RoCE 够用
```

### 🔧 自己动手验：你的 RoCE 集群健康吗？

```bash
# 1. 检查 PFC 风暴风险（所有 GPU 节点执行）
#    注意：rx_pause / tx_pause 是累计计数，必须间隔采样两次、求差才是速率
ethtool -S eth0 | grep -E "rx_pause|tx_pause"
# 相隔 1 小时各采一次，两次相减得到每小时增量
# 期望：增量 < 100 万次/小时
# 如果增量 > 1,000 万次/小时 → PFC 风暴进行中

# 2. 检查是否有 RDMA 流量（确认真的在走 RDMA，不是 TCP 回退）
cat /sys/class/infiniband/*/ports/1/counters/port_rcv_data
# 这个计数器不动 → 你的 RDMA 链路可能没工作

# 3. 检查 NCCL 实际走的协议
NCCL_DEBUG=INFO mpirun -np 8 all_reduce_perf -b 256M -e 256M 2>&1 | grep "NCCL INFO"
# 看输出中的 "via NET/IB" = RDMA，"via NET/Socket" = TCP 回退（慢！）
```

### ⚠️ 常见误区

- **误区**："RoCE 比 IB 便宜 50%，选 RoCE 肯定是省钱的"
  - **实际**：硬件成本省 30-50%，但运维成本 + 训练中断损失可能在 6 个月内超过硬件差价。**总拥有成本（TCO）两者基本持平**。
- **误区**："ECMP 哈希可以把流量均匀分配到所有路径"
  - **实际**：ECMP 按流的五元组哈希分配，多条并行 All-Reduce 流可能哈希到同一条链路，造成局部拥塞。这就是为什么需要自适应路由——IB 原生支持，NVIDIA Spectrum-X（以太网）、Broadcom 动态负载均衡等也支持；**RoCEv2 协议本身并不禁止自适应路由**，只是传统以太网交换机默认走 ECMP。
- **误区**："我们的网络团队很厉害，RoCE 调优没问题"
  - **实际**：RoCE 调优不是"一次搞定"——每次换固件、换光模块、换交换机型号、增减节点，都要重新验证 PFC 行为。**IB 是"配好就不动"，RoCE 是"永远在调"。**

### 📊 底线数字速查

| 指标 | 正常 | 预警 | 故障 |
|------|------|------|------|
| PFC rx_pause 增量速率（两次采样求差） | < 100 万/小时 | 100 万-1,000 万/小时 | > 1,000 万/小时 |
| NCCL busbw（2节点 RoCE 400G） | > 300 GB/s | 200-300 GB/s | < 200 GB/s |
| NCCL busbw（2节点 IB NDR400） | > 350 GB/s | 300-350 GB/s | < 300 GB/s |
| 光模块 RX 光功率 | > -8 dBm | -8 ~ -12 dBm | < -12 dBm |
| 交换机温度 | < 65°C | 65-75°C | > 75°C |
| 端到端节点间 ping 延迟 | < 2 μs | 2-5 μs | > 5 μs |

### 💡 一句话记住

> **"IB 是打专线电话（先确认再发），RoCE 是发快递（先发了再说）。一万人同时发快递，堵一次的损失超过专线月费。"**

---

## 对后续的铺垫

那是不是说 RoCE 永远不行？不是。2025 年 6 月发布的 **UEC 1.0 规范** 试图在以太网上重建传输层（UET），从根本上解决 PFC 的问题。

→ [[04-UEC1.0是什么]]

---

> **参考资料**：
> - [IBM Research: "Designing RoCE Networks for AI Workloads" (Alim Alim, UCICS 2026)](https://research.ibm.com/publications/designing-roce-networks-for-ai-workloads)
> - [NVIDIA NCCL 文档：环境变量参考](https://docs.nvidia.com/deeplearning/nccl/)
> - [AMD ROCm Blog: "Comparative Analysis of Scale-Out RoCE Network Traffic Patterns and Loads in Training Large Language Models" (2026)](https://rocm.blogs.amd.com/artificial-intelligence/amd-comparative-analysis/README.html)
