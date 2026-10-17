# Q: 全光互联、CXL 内存池化意味着什么？

## 为什么会有这个问题

GPU 越来越快，HBM 越来越贵，显存越来越不够用。

你有一个 70B 模型、128K 的上下文。KV Cache 有多大？先算清楚：

```
KV Cache ≈ 2 × layers × kv_heads × head_dim × seq × batch × bytes

Llama 3 70B：80 层、8 个 KV head（GQA）、head_dim=128、
128K 上下文、FP16（2 bytes）、batch=1：

2 × 80 × 8 × 128 × 131,072 × 2 ≈ 40 GiB

batch=8 时就是 320 GiB——任何单卡显存都放不下。
```

**核心矛盾**：GPU 的算力每代翻倍，但 HBM 的容量和带宽增长慢得多，而且 HBM 焊死在 GPU 封装旁边，买不起也加不了。

**CXL（Compute Express Link）内存池化和光互联，是业界对"内存墙"的两个回答。但它们各自能干什么、不能干什么，经常被讲反。**

## 解决什么问题

### 先把最重要的事实说清楚

```
事实 1：CXL 池化的是"主机内存"（CPU 侧 DRAM），不是 HBM。

  CXL 建立在 PCIe 物理层之上，挂在 CPU 的 PCIe/内存子系统旁边。
  它能让你：
    · 给服务器加一块"内存条形态"的扩展内存（内存扩展设备）
    · 把多台服务器的主机 DRAM 池化共享
  它不能让你：
    · 给 GPU 增加 HBM——GPU 的显存扩展走的是 NVLink-C2C
      （Grace Hopper / Grace Blackwell 超级芯片：CPU DRAM 与 GPU HBM
      通过 NVLink-C2C 一致性互联，这才是 NVIDIA 的"显存扩展"路径）

事实 2：CXL 内存比本地内存慢，但比"放不下了只能换大卡"便宜。

  · 本地 HBM / DDR：延迟约几百 ns（HBM 经内存控制器约 300-500 ns 量级）
  · CXL 附加内存：在本地内存基础上再多出约 100-300 ns
    （链路 + 交换机转发）
  · 所以 CXL 适合"容量敏感、延迟容忍"的数据：
    KV Cache 温层、冷参数、数据集缓存
```

### CXL 的版本演进（按 CXL Consortium 官方口径）

```
CXL 1.1/2.0：基础能力——类型 1/2/3 设备、内存扩展、简单切换
CXL 3.x：引入 Fabric 能力——
         多级交换、内存设备可以跨主机被多个主设备访问、
         背板上的 switch 级联组成内存池
CXL 4.0：规范已由 CXL Consortium 发布
         （信号速率升至 128 GT/s，基于 PCIe 7.0 基础），
         但产品化时间待定——不要按某个"量产年份"做采购计划
```

### 光互联：OCS、光模块、光交换机是三件事

谈"全光互联"之前先分清三个词：

```
1. 可插拔光模块（最常见）
   电口交换机 + 光模块，用于"链路是光纤"的普通以太网/IB 网络。
   交换机内部仍然是电交换芯片。

2. OCS（Optical Circuit Switch，光路交换机）
   直接在光域做"开关矩阵"：把端口 A 连到端口 B，不光电、不解析包头。
   · 优点：转发不耗算力、功耗极低、与协议无关
   · 限制：只能做"电路"，不能做"包交换"，切换是毫秒-秒级
   · 可核验案例：Google 在 TPU Pod 内大规模使用 OCS 重构拓扑
     （训练中途把死掉的芯片绕开、动态改组 Pod 形状）
   · 注意：这是 Google 的实践，不要泛化成"所有集群都在用 OCS"

3. 光子互连/计算内光互连（Marvell Photonic Fabric 等厂商方向）
   用光做设备间的数据通路，面向内存/加速器互联，
   属于厂商路线图，不是行业标准品。

当前主流集群是"光电混合"：链路用光（光模块），
交换用电（交换芯片），OCS 是 Google TPU 这类特殊场景的补充。
```

---

## 三大互联的共存格局

2026-2027 年的 AI 集群中，几种互联各管一段：

| 互联 | 管什么 | 带宽口径 | 距离 | 说明 |
|------|--------|---------|------|------|
| **NVLink 5** | GPU-GPU 显存/通信 | 1.8 TB/s 双向/卡 | 机内-机柜 | 封闭生态，NVL72 为 72 卡域 |
| **UALink 1.0** | 开放 GPU-加速器互联（AMD/Intel 等联盟） | **800 Gb/s = 100 GB/s 单向/端口** | 机内-短距 | 规范已发布，产品在路上 |
| **CXL** | 主机内存扩展/池化 | 随 PCIe 通道数而定 | 机内-跨机（3.x fabric） | 池的是 **DRAM**，不是 HBM |
| **OCS** | 光路交换 | 与协议无关 | 机房内-园区 | Google TPU 是公开的成熟案例 |

两个容易写错的点：

```
1. UALink 的带宽是"每端口 800 Gb/s ≈ 100 GB/s 单向"，
   不是"1,600 GB/s"——后者是把方向或端口数搞混了。

2. NVLink Fusion（NVIDIA 对外开放 NVLink 生态的动作）与 UALink
   是竞争关系，但不要写成"某一方已死"——两者都处于早期，
   实际采用率随时在变，写结论之前看最新公告。
```

---

## 这对 AI 训练/推理意味着什么

### 场景 1：KV Cache 分层（最现实的用法）

```
推理集群的 KV Cache 分层：

  热层：GPU HBM（最快，放当前 batch 正在用的 KV）
  温层：主机 DRAM / CXL 内存池（放长上下文、多轮会话的历史 KV）
  冷层：NVMe/SSD（放不活跃会话）

  70B 模型 batch=8 的 128K 上下文 KV Cache ≈ 320 GiB，
  远超单卡 HBM——分层是必然的。

  注意：这里的"温层"是主机侧 DRAM（可以经 CXL 扩容），
  不是"给 GPU 加的 HBM"。GPU 访问主机 DRAM 走 PCIe（或 NVLink-C2C
  平台上的统一内存），带宽和延迟都不如 HBM——
  所以只放 decode 时低频访问的 KV。
```

### 场景 2：主机内存池化（内存型负载）

```
集群里有大量"内存不够、算力闲置"的负载：
  · 向量数据库、Embedding 缓存
  · 数据集预处理流水线
  · 小模型的 CPU 推理

CXL 内存池可以把多台服务器的 DRAM 池化，
让内存不够的节点借到池里的容量——提高整体利用率。
```

### 场景 3：不能做的事

```
❌ "CXL 给 GPU 加显存" —— GPU 显存扩展走 NVLink-C2C，不是 CXL
❌ "CXL 内存当 HBM 用跑权重更新" —— 延迟和带宽差一个量级
❌ "光互联让 300 米外内存像本地一样快" —— 光解决的是
   "距离上的带宽/功耗"，不改变内存访问的协议延迟
```

---

## CXL 的真实瓶颈（部署前必读）

CXL 内存池化目前落地面有限，卡在五个地方：

```
1. 延迟：CXL 附加内存比本地 DDR 慢 100-300 ns 量级。
   对延迟敏感的访问模式，这个差距直接决定能不能用。

2. 带宽共享：CXL 设备和 GPU 网卡、NVMe 共享 PCIe 通道。
   池化带宽是"所有访问者分"的，不是每设备独享。

3. 缓存一致性：跨主机的一致性（3.x fabric 的目标）协议开销大，
   实际产品多限于机箱内或近距场景。

4. 软件支持：内核（CXL 子栈）、内存分层管理器、
   运行时（如何告诉 PyTorch "这块内存在远端"）都还在演化。
   NUMA 感知、页面迁移策略要自己调。

5. 故障域：把内存池化后，一台内存设备故障影响的不再是一台
   服务器，而是整个池——故障域变大，隔离和排障更复杂。
```

结论：**CXL 在 2026 年的角色是"主机内存的容量扩展器"，而不是"GPU 显存的救星"**。规划集群时把它当作"更灵活的 DRAM"，预算和架构就不会跑偏。

---

## 🛠 实操指南

### 📋 决策指南：CXL 和光互联适合你吗？

```
你的场景？
├─ 单机训练（8 卡）→
│   └─ CXL 不是刚需；显存不够优先考虑模型并行/卸载
│
├─ 长上下文推理服务 →
│   └─ KV Cache 分层（HBM → 主机 DRAM → SSD），
│      主机 DRAM 可以用 CXL 扩容
│
├─ 显存不够但不想加 GPU →
│   └─ 先算算是不是 KV Cache 的问题（用上面的公式）；
│      CXL 帮不了 GPU HBM，能帮的是主机侧
│
└─ 设计下一代集群 →
    ├─ 基础设施预留 CXL 交换机/背板兼容性
    ├─ 跟踪 UALink / NVLink Fusion 的产品化进展
    └─ OCS 仅在超大规模场景认真评估（参考 Google 的用法）
```

### 🔧 自己动手验：检查你的系统是否支持 CXL

```bash
# 1. 检查 CPU 平台是否支持 CXL
#    Intel：Sapphire Rapids（CXL 1.1）/ Emerald Rapids / Granite Rapids
#    AMD：EPYC Genoa（9004）/ Turin（9005）系列支持 CXL
lscpu | grep -i "model name" | head -1

# 2. 检查是否检测到 CXL 设备/内存
#    Linux 5.15+ 有 cxl 子系统；有 CXL 内存时通常表现为新 NUMA 节点
ls /sys/bus/cxl/devices/ 2>/dev/null
numactl -H        # 看是否有额外的 NUMA 节点
cxl list 2>/dev/null   # 需要 cxl-utils

# 3. 观察 NUMA 距离，评估远端内存的代价
numactl -H | grep distances -A 10
# distance 越大（如 2 跳以上），访问该节点的代价越高
```

### ⚠️ 常见误区

- **误区**："CXL 内存可以当 HBM 用"
  - **实际**：CXL 池化的是主机 DRAM。它的延迟比 HBM 高（HBM 经内存控制器约 300-500 ns，CXL 还要再加 100-300 ns），带宽也低。它适合 KV Cache 温层、冷参数、数据缓存，不适合放频繁访问的权重。

- **误区**："CXL 能扩展 GPU 显存"
  - **实际**：GPU 显存的扩展路径是 **NVLink-C2C**（Grace Hopper / Grace Blackwell 把 CPU DRAM 和 GPU HBM 做成一致性域）。CXL 和 GPU HBM 之间没有这条通路。

- **误区**："光互联是未来的唯一方向"
  - **实际**：光互联在长距离和高带宽场景有优势，但短距离、机内场景电互联的成本和能效仍然更好。当前是"光电混合"：链路用光、交换用电，OCS 只在特定超大规模场景（如 Google TPU Pod）大规模使用。

- **误区**："CXL 4.0 发布了，明年就能部署"
  - **实际**：CXL 4.0 是规范层面的发布，产品化时间待定。今天能采购到的是 CXL 2.0/3.x 级别的设备。做集群规划时按"规范已定、产品待来"处理。

### 📊 底线数字速查

| 方案 | 角色 | 延迟量级 | 备注 |
|------|------|---------|------|
| 本地 HBM | GPU 显存 | 约 300-500 ns（经内存控制器） | 焊死，不可扩 |
| 本地 DDR5 | 主机内存 | 一两百 ns 起 | 每台机器容量有限 |
| CXL 附加内存 | 主机内存扩展 | 本地 + 100-300 ns | 池的是 DRAM |
| NVLink-C2C | CPU-GPU 一致性互联 | 远低于 PCIe | GPU"显存扩展"的正确路径 |
| OCS | 光路交换 | 毫秒-秒级切换 | 不增加带宽，重构拓扑 |

### 💡 一句话记住

> **"CXL 是'给主机内存扩容'的标准，不是'给 GPU 加显存'的标准——后者靠 NVLink-C2C。把 CXL 用在 KV Cache 温层和内存型负载上，它是好工具；把它当成 HBM 的平替，架构就从第一天就错了。光互联同理：OCS 重构拓扑很强（Google TPU 验证过），但它不是内存扩展方案。"**

---

## 对后续的铺垫

前沿科技看再多，不如动手搭一个。

前面我们聊了几千张 GPU 的集群——但**大部分团队不需要**。如果你只是想入门、做实验、跑小模型，2-4 台机器就够了。

下一节：自己动手搭建一个小集群。

→ [[04-自己动手搭建一个小集群怎么入门]]

---

> **参考资料**：
> - [CXL Consortium：About CXL](https://computeexpresslink.org/about-cxl/) — CXL 规范（1.1/2.0/3.x/4.0）的权威出处
> - [UALink Consortium](https://ualinkconsortium.org/) — UALink 1.0 规范（800 Gb/s/端口）
> - [Google Cloud：Ironwood TPU](https://cloud.google.com/blog/topics/systems-innovation/announcing-trillium-next-generation-tpu-ironwood) — Google Cloud 官方博客（ICI + OCS 的公开实践）
> - [Ultra Ethernet Consortium](https://ultraethernet.org/) — 以太网阵营对光/电网络的演进方向
> - [NVIDIA GB300 NVL72 官方规格页](https://www.nvidia.com/en-us/data-center/gb300-nvl72/) — NVIDIA（NVLink 5 与 GPU 显存路径的官方口径）
