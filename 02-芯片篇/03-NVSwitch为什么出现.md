# Q: NVLink 够用了吗？NVSwitch 为什么出现？

## 为什么会有这个问题

NVLink 很强大，但有个现实问题：**GPU 之间不是只做"点对点"通信的**。

在数据并行训练中，所有 GPU 需要做 All-Reduce 同步梯度——这意味着**每张卡都要和所有其他卡通信**（[[01-基础篇/02-为什么不用多卡直接跑]]）。

### 8 张卡需要多少条 NVLink 才能全互联？

如果每张卡直接连到其他 7 张卡：
- 总共需要的连接数：C(8,2) = **28 条**
- 每张卡需要：**7 个 NVLink 链路对**

H100 有 18 条 NVLink 4 链路。如果要 8 卡纯点对点全互联（不经过交换芯片），每对 GPU 之间最多分到 2 条链路——这是早期 DGX-1（P100/V100 时代）混合立方体拓扑的思路。

但 NVIDIA 在 DGX H100 上换了做法：**用交换芯片把 18 条链路集中管理**。这里先纠正一个常见说法——

> **DGX H100 不是"无 NVSwitch 直连"**。DGX H100 里有 **4 颗 NVSwitch 3 芯片**，每张 H100 的 18 条 NVLink 链路按 5+4+4+5 的方式分别接到这 4 颗 NVSwitch 上。**用 6 颗 NVSwitch 的是 DGX A100**（NVSwitch 2，每卡 12 条链路）。两代机器都不能算"点对点直连"。

### 跨机箱的场景

假设你有 16 张 H100（2 个 DGX 机箱），要让机箱 A 的 GPU 0 和机箱 B 的 GPU 15 通信：

```
机箱 A                                             机箱 B
┌──────┐                                         ┌──────┐
│ GPU0─┼──────NVLink──────┐                      │─GPU15│
│ GPU1─┼──────NVLink──────┤                      │─GPU14│
│ ...  │                 ❌ 没有直连               │ ...  │
│ GPU7─┼──────NVLink──────┤                      │─GPU8 │
└──────┘                  │                      └──────┘
                          │
                    必须先经过 CPU 或 PCIe → 变成慢速通道
```

NVLink 物理线缆长度有限，跨机箱时又回到了 PCIe 或网络。

---

## 解决什么问题：NVSwitch

**NVSwitch** 是一个专门为 GPU 间通信设计的**全互联交换芯片**，部署在机箱内部。

### 工作原理

```
NVSwitch 方案（DGX H100）：

            ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐
            │NVSwitch│  │NVSwitch│  │NVSwitch│  │NVSwitch│
            │   #1   │  │   #2   │  │   #3   │  │   #4   │
            └─┬─┬─┬──┘  └─┬─┬─┬──┘  └─┬─┬─┬──┘  └─┬─┬─┬──┘
              │ │ │       │ │ │       │ │ │       │ │ │
   每张 GPU 的 18 条 NVLink 链路按 5+4+4+5 分别接入这 4 颗 NVSwitch
              │ │ │       │ │ │       │ │ │       │ │ │
           GPU 0   GPU 1   ...        GPU 6   GPU 7

总带宽：任意 GPU ↔ 任意 GPU = 900 GB/s（双向，NVLink 4 聚合）
```

关键：**NVSwitch 不是一张"卡"，它是一个独立的、专用的交换芯片。**

| 规格 | NVSwitch 3（DGX H100） | NVSwitch 4（GB200 NVL72） |
|------|-------------------|-------------------|
| 端口数 | **64 端口** | **72 端口** |
| 单端口带宽 | NVLink 4，50 GB/s（双向） | NVLink 5，100 GB/s（双向） |
| 单芯片聚合带宽 | 64 × 50 = **约 3.2 TB/s** | 72 × 100 = **约 7.2 TB/s** |
| 每卡 NVLink 带宽 | 900 GB/s（18 link × 50） | 1,800 GB/s（18 link × 100） |
| 典型部署 | DGX H100：4 颗/机箱 | NVL72：数十颗/柜，整柜聚合带宽约 130 TB/s |

NVSwitch 让机箱内的 GPU 形成一个**全互联的超大"虚拟 GPU"**——任何两张卡之间的通信都是 NVLink 级带宽，没有"绕路"的惩罚。

### NVSwitch 能扩展到多大？——NVLink 域的边界

**这是本文最重要的一条边界线：**

- **NVLink 域（scale-up）**：所有 GPU 通过 NVLink/NVSwitch 全互联、共享显存语义的范围。上限由交换层级决定：
  - H100（NVSwitch 3）：**8 卡**（DGX H100）
  - B200/GB300（NVSwitch 4）：**72 卡**（GB200/GB300 NVL72）
  - Rubin（下一代）：**NVL144**
  - Rubin Ultra：**NVL576**
- **跨 NVLink 域（scale-out）**：域与域之间**走 InfiniBand 或以太网**，不是 NVLink。

> **"DGX SuperPOD 是 256 卡 NVLink 全互联"是误传。** SuperPOD 的核心思想是把多个机箱组成一个集群，但**机箱之间走的是 InfiniBand**（SuperPOD 的网络层），不是 NVLink。多个 SuperPOD 之间再由 IB 交换机互联成千卡、万卡集群。任何宣称"数百卡 NVLink 全互联"的说法，都要对照上面那张 NVLink 域上限表检查一遍。

---

## 优缺点

### NVSwitch 优点
| 优势 | 说明 |
|------|------|
| 任意 GPU 间全带宽 | 无"近远"GPU 之分，通信性能一致 |
| 简化拓扑管理 | NCCL 不需要操心路由，直接通过 NVSwitch |
| 支持 NVLS（NVLink SHARP） | All-Reduce 的聚合可以在 NVSwitch 内完成，进一步省带宽 |

### NVSwitch 缺点
| 劣势 | 说明 |
|------|------|
| 成本极高 | NVSwitch 芯片独立设计，DGX/NVL 机柜价格包含这部分成本 |
| **NVLink 域有上限** | 超出域（8 卡 / 72 卡）就要走 IB/以太网，带宽按端口口径骤降 |
| NVIDIA 独占 | 其他厂商无法使用 |

---

## 🛠 实操指南

### 📋 决策指南：你需要 NVSwitch 吗？

**判断依据很简单：你的 GPU 是否需要频繁做跨卡通信？**

| 场景 | 需要 NVSwitch？ | 理由 |
|------|---------------|------|
| 单卡推理 | ❌ 不需要 | 不需要跨卡通信 |
| 小规模微调（1-2 卡） | ❌ 不需要 | PCIe 或 NVLink 直连足够 |
| 单机 8 卡数据并行 | ✅ DGX 标配 | All-Reduce 频繁，NVSwitch 提升明显 |
| 单机 8 卡模型并行（TP） | ✅ **强烈推荐** | TP 要求每步通信，NVSwitch 延迟极低 |
| 多机多卡（32+ GPU） | ✅ 机箱内需要 | **机箱内走 NVSwitch（NVLink 域），机箱间走 IB/RoCE（scale-out）** |
| GB200/GB300 NVL72 | ✅ **NVSwitch 4 必须** | 72 GPU 全互联，全靠 NVSwitch |

**关键点**：如果你只在单机上跑 2-4 卡、且不做张量并行（TP），NVSwitch 的价值不明显。但一旦你开始做模型并行（8 卡 TP）或大模型跨机训练，**NVLink 域内 vs 域外的带宽差**直接决定了你的 MFU。

### 🔧 自己动手验：你的 NVSwitch 工作正常吗？

```bash
# 1. 检查 NVSwitch 健康状态（H100/B200 DGX 系统）
nvidia-smi nvswitch -i 0 -q
# 重要字段：
#   Version（固件版本）
#   Temperature（温度，正常 < 70°C）
#   Link Errors（链路错误，应该为 0）
#   Bandwidth Utilization（带宽利用率）

# 2. 检查所有 NVSwitch 的链路状态
nvidia-smi nvswitch -i 0 -q | grep -E "Link.*Status|Error|CRC"
# 所有 Link Status 应该是 Active
# CRC Error Count 应该为 0（CRC 错误说明信号质量有问题）

# 3. 更彻底的测试：全互联带宽基准（大消息，看 busbw）
mpirun -np 8 --allow-run-as-root \
  nccl-tests/build/all_reduce_perf -b 1G -e 1G -f 2
# 8×H100 DGX 预期结果：> 400 GB/s（busbw 口径）
# 如果 < 300 GB/s → 可能是 NVSwitch 链路降速或 NCCL 配置问题

# 4. 检查 NCCL 是否识别了 NVSwitch 拓扑
NCCL_DEBUG=INFO mpirun -np 8 \
  nccl-tests/build/all_reduce_perf -b 128M -e 128M
# 在日志中搜索：
#   "NVLS is supported" → NVLink SHARP 可用
#   "Tree"/"Ring"/"Auto" → 查看 NCCL 选择的算法
```

### ⚠️ 常见误区

- **误区**："NVSwitch = NVLink，有 NVLink 就不需要 NVSwitch"
  - **实际**：NVLink 是**链路协议**，NVSwitch 是**交换芯片**。少量卡可以靠 NVLink 点对点直连，但链路数不够分时（比如 8 卡全互联、或 8 卡以上的 NVLink 域），必须通过 NVSwitch 做交换。**没有交换层，多卡全互联要么牺牲每对卡之间的带宽，要么根本连不起来。**

- **误区**："NVSwitch 解决了跨机通信的问题"
  - **实际**：NVSwitch 只工作在 **NVLink 域内**（H100 域上限 8 卡，NVL72 为 72 卡）。跨机箱通信走 IB/以太网，每卡跨机带宽受单端口限制（IB NDR 400G 单端口 50 GB/s 单向，实测每卡约 40-50 GB/s），与整卡 NVLink 900 GB/s（双向）相比，整卡口径约 18:1。**这就是为什么大模型训练要把通信尽量限制在 NVLink 域内。**

- **误区**："NVSwitch 是 NVIDIA 独有技术，其他厂商没法做类似的东西"
  - **实际**：AMD 有 Infinity Fabric，华为有 HCCS，Google TPU 用 3D Torus——每个厂商都有自己的 scale-up 互联技术。但 NVSwitch 的**大规模商用部署经验**（从 2018 年 DGX-2 到 NVL72）是最成熟的。同时业界在推动开放标准：UALink 1.0 规范定义了 800G（100 GB/s 单向）/端口的 scale-up 互联；OCP 也成立了 ESUN（Ethernet for Scale-Up Networking）工作组。未来 NVSwitch 的专有地位可能被开放标准动摇。

### 📊 底线数字速查

| 指标 | NVSwitch 3（Hopper） | NVSwitch 4（Blackwell） | 下一代（Rubin） |
|------|-------------------|-------------------|------------------------|
| 每 GPU NVLink 带宽（双向） | 900 GB/s | **1,800 GB/s** | 以官方发布为准 |
| 每交换芯片端口数 | 64 端口 | **72 端口** | 待公布 |
| 单端口带宽 | 50 GB/s（双向） | 100 GB/s（双向） | 待公布 |
| 单芯片聚合带宽 | 约 3.2 TB/s | 约 7.2 TB/s | 待公布 |
| 典型部署 | DGX H100：4 颗 | NVL72：整柜聚合约 130 TB/s | NVL144 |
| NVLink 域上限 | **8 卡** | **72 卡** | NVL144（Rubin Ultra 为 NVL576） |
| 域外通信 | IB NDR 单端口 50 GB/s（单向） | IB/以太网（CX-8 800G 单端口 100 GB/s 单向） | 1.6T 级（下一代网卡） |

### 💡 一句话记住

> **"NVSwitch 是 GPU 之间的'高速公路立交桥'——它把每卡十几条 NVLink 集中交换，形成一个大'虚拟 GPU'。但这座桥只修到 NVLink 域的边界（8 卡 / 72 卡），出了域就是 IB/以太网的天下。"**

---

## 对后续的铺垫

NVSwitch 很好地解决了 **NVLink 域内**所有 GPU 的高速通信问题。但当你需要从 8 卡扩展到 1,000 卡时，**跨机通信**怎么解决？

NVLink 域上限目前是几十到几百卡（H100 8 卡、NVL72 72 卡、Rubin Ultra NVL576），1,000 卡以上怎么办？

→ **这就是 03-网络篇 要回答的问题**：机间通信靠 InfiniBand 或 RoCE。

[[03-网络篇/01-为什么GPU之间需要网络]]

---

> **参考资料**：
> - [NVIDIA DGX H100 页面](https://www.nvidia.com/en-us/data-center/dgx-h100/) — 4 颗 NVSwitch、每 GPU 18 条 NVLink 链路
> - [NVIDIA GB200 NVL72 官方页面](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) — 72 卡 NVLink 域、NVSwitch 4
> - [ServeTheHome: NVIDIA NVSwitch @ Hot Chips 34](https://www.servethehome.com/nvidia-nvlink4-nvswitch-at-hot-chips-34/) — NVSwitch 3 芯片规格细节
> - [NVIDIA NVLink 官方页面](https://www.nvidia.com/en-us/data-center/nvlink/) — 各代链路带宽与 NVLink 域概念
> - [NVIDIA DGX SuperPOD 页面](https://www.nvidia.com/en-us/data-center/dgx-superpod/) — 跨机箱走 IB 的集群架构
