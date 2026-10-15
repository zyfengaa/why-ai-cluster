# Q: PCIe 瓶颈是什么？NVLink 怎么解决？

## 为什么会有这个问题

GPU 计算能力极强，但 GPU 每做完一轮计算，就**必须和其他 GPU 交换数据**（梯度同步，见 [[01-基础篇/02-为什么不用多卡直接跑]]）。传统上，GPU 间的通信走 **PCIe（PCI Express）总线**。

但 PCIe 不是为 GPU 间点对点通信设计的。它的架构决定了它在这个场景下极度低效。

### 不匹配的带宽：先看数字

| 通道 | H100 HBM → 计算核心（单向） | PCIe 5.0 x16（GPU→CPU，单向） | NVLink 4.0（GPU→GPU，单向） |
|------|-------------------|----------------------|---------------------|
| 带宽 | **3,350 GB/s** | **64 GB/s** | **450 GB/s** |
| 差距 | — | 慢了 **52 倍** | 慢了 **7.4 倍** |

> 注：为统一口径，本表全部采用**单向**带宽。厂商宣传中常见的 128 GB/s（PCIe 5.0 x16）与 900 GB/s（NVLink 4.0）都是**双向**聚合值，正好是本表数字的 2 倍。

**但更糟的是**：3,350 GB/s 是显存到计算核心的内部带宽，64 GB/s（单向）是 PCIe 的物理上限。**一次 All-Reduce 需要把几百 MB 的梯度从每张 GPU 的显存搬出去、汇总、再搬回来——全要走 PCIe 这根管子。**

### 更隐蔽的问题：PCIe 的数据路径

PCIe 点对点（P2P）通信的真实数据流取决于机器拓扑。在**支持 P2P** 的机器上，GPU 0 直接把数据 DMA 到 GPU 1 的 BAR 空间，经 PCIe switch **直通**，**CPU 只负责发起、不参与搬运**——在 `nvidia-smi topo -m` 中表现为 `PIX`（同一 PCIe switch 下）或 `PXB`（跨 switch 但同属一个 CPU）。

只有在 **P2P 不可用**时（跨 NUMA / ACS 开启 / 部分虚拟化），才会退化成"绕 CPU 与 DDR"的老式 pinned host buffer 路径：

```
PCIe P2P 不可用时的退化路径：

GPU 0 显存 → GPU 0 PCIe → CPU Root Complex → DDR 内存
    ↓                                                   
CPU 从内存读数据 → 写回 DDR → CPU Root Complex → GPU 1 PCIe → GPU 1 显存

全程走了：
  2 次 PCIe 传输（GPU↔CPU 各一次）
  1 次 CPU 中转（CPU 要介入数据搬运）
  2 次 DDR 读写（CPU 写内存、读内存）
  
每跳延迟：~5-10 μs
总延迟（8 卡 All-Reduce）：200-500 μs ← 这就是 MFU 的杀手
```

在 `nvidia-smi topo -m` 里，这种退化路径会显示为 `SYS`（跨 NUMA 节点），而不是 `PIX`/`PXB`。

还有一点要特别注意：**H100 SXM 的 DGX 机器上，GPU 之间根本没有 PCIe 数据路径**——所有 GPU↔GPU 通信全部经 NVLink/NVSwitch，PCIe 只负责 GPU↔CPU（详见 [[03-NVSwitch为什么出现]]）。

打个比方：HBM 到计算核心是**16 车道高速路**（3,350 GB/s），PCIe 到其他 GPU 是**一条乡间小道**（64 GB/s，单向），而且一旦 P2P 不可用，这条小道还必须在 CPU 门口绕个大 U 字。

---

## 解决什么问题

**NVLink** 是 NVIDIA 在 2014 年推出的 GPU 间直连高速总线。核心思想极简单：**给 GPU 之间铺一条专属高速路，不经过 CPU。**

### 架构对比

```
PCIe 方案（P2P 不可用时绕经 CPU）：
  GPU 0 ──PCIe──→ CPU（中转处理）──PCIe──→ GPU 1
  
NVLink 方案（直连）：
  GPU 0 ══NVLink══→ GPU 1   ← GPU 直接访问另一张 GPU 的显存
```

**技术实现**：NVLink 在每张 GPU 芯片上集成了专用的 NVLink 接口控制器，直接连接到另一张 GPU 的 NVLink 控制器。数据从 GPU 0 的 HBM 出发，通过 NVLink 到达 GPU 1 的 HBM——**CPU 完全不知道这件事，也不需要知道。**

### 历代演进

| 版本 | 首款 GPU | **单卡双向带宽** | 相比同期 PCIe | 每卡链路数 |
|------|---------|------------|-------------|-----------|
| NVLink 1.0 | P100 (2016) | 160 GB/s | PCIe 3.0 × 5 | 4 |
| NVLink 2.0 | V100 (2017) | 300 GB/s | PCIe 3.0 × 9.4 | 6 |
| NVLink 3.0 | A100 (2020) | 600 GB/s | PCIe 4.0 × 9.4 | 12 |
| **NVLink 4.0** | **H100 (2022)** | **900 GB/s** | PCIe 5.0 × **7** | **18** |
| NVLink 5.0 | B200 (2025) | **1,800 GB/s** | PCIe 5.0 × **14** | 18 |

NVLink 4.0 的 900 GB/s 是什么概念？**一张 H100 在 1 秒内可以通过 NVLink 传输 11 部蓝光电影的数据（每部 80 GB）。**

### 真实场景：NVLink 带来的性能差异

看一个实际测试。8 张 H100 跑 All-Reduce（256 MB 消息）：

| 互联方式 | 实测带宽（busbw） | 延迟 | 适用 |
|---------|-------------------|------|------|
| PCIe 5.0 x16（P2P） | ~45-60 GB/s（单向理论上限 64 GB/s） | ~50 μs | 老集群，没 NVLink |
| NVLink 4.0（H100） | Ring ~360 GB/s / 启用 NVLS ~480 GB/s | ~10 μs | DGX H100 标准配置 |

**差距：带宽约 7 倍，延迟约 5 倍。** 这就是为什么所有现代的 AI 服务器都标配 NVLink——没有它，多卡训练的效率会惨不忍睹。

---

## 优缺点：NVLink 不是完美的

### 优点

1. **带宽碾压**：900-1,800 GB/s，PCIe 完全无法企及
2. **低延迟直连**：绕开 CPU，数据在 GPU 之间直接飞
3. **原子操作支持**：NCCL 的 All-Reduce 可以直接在 NVLink 层做 Reduce 操作（边传输边求和），不需要传完再算
4. **双向并行**：上行 450 GB/s + 下行 450 GB/s 同时跑

### 缺点

| 劣势 | 具体问题 | 真实影响 |
|------|---------|---------|
| **NVIDIA 私有** | AMD/Intel/华为 GPU 无法使用 | 生态锁定，采购别无选择 |
| **距离受限** | NVLink 域受限于机柜内的铜缆/背板 | 只能在同一机柜内，**跨机柜不行**；最大域规模从 Hopper 的 256 卡到 Blackwell 的 72 卡（NVL72） |
| **端口有限** | 每张 H100 有 18 个 NVLink 接口 | 8 卡全互联每卡只需 **7** 条，18 条绰绰有余；DGX H100 实际是经 **4 颗 NVSwitch** 全交叉，不是 GPU 直连 full mesh |
| **成本高** | NVLink Switch 是独立芯片 | DGX 机箱的溢价很大程度上来自 NVSwitch |
| **不支持 CXL** | NVLink 不是 CXL 标准，无法内存池化 | 无法做异构内存共享 |

### 最大的那个"但是"：跨机不行

NVLink 解决了单机内 8 张 GPU 的通信。但 8 张卡不够了怎么办？你想从一个 DGX 机箱（8 卡）扩展到两个（16 卡），NVLink 直连办不到了。

```
第 1 台 DGX             第 2 台 DGX
┌──────────────┐       ┌──────────────┐
│ GPU 0 ══ NVLink ══ GPU 0 │    ← 机内 OK
│ GPU 1 ══ NVLink ══ GPU 1 │
│ ...            │       │ ...          │
│ GPU 7 ══ NVLink ══ GPU 7 │
└──────────────┘       └──────────────┘
      ❌ GPU 0 (DGX1) 到 GPU 7 (DGX2) 没有 NVLink 直连
        只能通过 InfiniBand/RoCE 走网络
        单卡单向：NVLink 4.0 450 GB/s → IB NDR 50 GB/s
        （注：常见说法"900 → 50，18 倍"把 NVLink 双向值
          和单张网卡的单向值混比了，量纲不对）
        按节点看：一个训练节点通常配 8 张 NDR 网卡，聚合带宽
        ~400 GB/s，跨机通信的实际差距远小于"18 倍"这个数字
```

这就是为什么超过 8 卡后，**网络设计就替代 NVLink 成为通信瓶颈的决定性因素**（[[03-网络篇/01-为什么GPU之间需要网络]]）。

---

## 现实中的 NVLink 问题：一次典型的排错流程

假设一台 8 卡 H100 训练节点上，某一张卡的 NCCL All-Reduce 速度明显低于其他卡——这是"疑似 NVLink 掉链路"的典型场景。排查流程如下（以下为通用步骤示意，非某一具体案例）：

```
Step 1: nvidia-smi topo -m
检查 GPU 拓扑连接
正常：任意两张 GPU 之间都是 NV2/NV3 标签
异常：某张 GPU 对应的行里出现 PCI / PHB / SYS
     —— 说明这张卡的 NVLink 链路没有全部建立

Step 2: nvidia-smi nvlink -s（或 -g <id> -s 只看某张卡）
逐条检查每条链路是 Enabled 还是 Disabled，
以及每条链路的速率是否符合预期

Step 3: 定位与恢复
  先用 NCCL_DEBUG=INFO 确认 NCCL 实际选了哪条路径
  → 排除软件/拓扑问题
  再尝试 nvidia-smi -r 做 GPU 级热重置
  仍不行 → 物理断电重启
  反复出现 → 记录该节点，联系硬件维保
            （通常指向连接器接触不良或背板问题）
```

教训：**NVLink 也会出问题，但排查相对简单，工具链成熟。** 对比网络问题的排查，NVLink 问题通常直接锁定到特定 GPU。

---

## 🛠 实操指南

### 📋 决策指南：你的 GPU 通信链路正常吗？

新上线的 GPU 节点，第一步就是确认 NVLink 连通性和性能：

```
检查步骤：
  1. nvidia-smi topo -m    → 看所有 GPU 间 NVLink 连接数
  2. nvidia-smi nvlink -s  → 看每条链路状态
  3. 跑 nccl-tests         → 看实际带宽
  
正常期望：
  8×H100 DGX：任何两张 GPU 的 NVLink 连接数 = 18 条
  8×A100 DGX：任何两张 GPU 的 NVLink 连接数 = 12 条
```

### 🔧 自己动手验：NVLink 状态全检查

```bash
# 1. 查看 NVLink 拓扑（是否有链路掉线）
nvidia-smi topo -m

# 期望输出（H100 DGX）：
#       GPU0 GPU1 GPU2 GPU3 GPU4 GPU5 GPU6 GPU7
# GPU0   X   NV2  NV2  NV2  NV2  NV2  NV2  NV2
# GPU1  NV2   X   NV2  NV2  NV2  NV2  NV2  NV2
# ...（所有格子都是 NV2 / NV3，没有 PCI 或 PHB）

# 2. 逐条检查 NVLink 链路状态  
nvidia-smi nvlink -s

# 3. 跑实际带宽验证
mpirun -np 8 all_reduce_perf -b 128M -e 256M -f 2
# 期望 busbw：350-450 GB/s（Ring）/ 450-480 GB/s（启用 NCCL_NVLS_ENABLE 时）
```

**如果发现链路 Disabled：**

```bash
# 注意：nvidia-smi nvlink -r 只是清空错误计数器（且需要 root），
# 并不会重置/恢复链路。H100 的 NVLink 没有可通过 nvidia-smi
# 执行的"软重启"——恢复手段是 GPU 级重置或物理断电。
nvidia-smi -r     # GPU 级热重置（会一并复位该卡上的 NVLink 链路）

# 如果不行 → 需要物理断电重启
# 如果反复出现 → 记录该节点，联系硬件维保
```

### ⚠️ 常见误区

- **误区**："既然有 NVLink，那多卡训练就不会有通信瓶颈"
  - **实际**：NVLink 只解决**单机内**的通信瓶颈。一旦涉及跨机（超过 8 卡），机间走的是网络（IB/RoCE）：单卡单向带宽从 NVLink 4.0 的 450 GB/s 掉到一张 NDR 网卡的 50 GB/s；但按节点看，一个训练节点通常配 8 张 NDR、聚合约 400 GB/s，实际差距远小于"18 倍"。**你的训练瓶颈很可能不在 NVLink 上，而在网络上。**
- **误区**："nvidia-smi 显示所有 GPU 利用率 100%，通信肯定没问题"
  - **实际**：NVLink 带宽利用率不等于 GPU 利用率。一个常见的场景是：多卡在做同步时 GPU 是 idle 的，但 nvidia-smi 仍可能显示 100%（因为有计算任务存在但不一定在跑 Tensor Core）。要**真的看 NVLink 流量**，用 `nvidia-smi nvlink -g 0 -t`（`--get-throughput`）看吞吐。

### 📊 底线数字速查

| 指标 | 正常 | 预警 | 故障 |
|------|------|------|------|
| 每卡 NVLink 活动链路数（H100） | 18 | 12-17 | < 12 |
| ALL-Reduce 带宽（8×H100 单机，Ring） | 350-450 GB/s | 250-349 GB/s | < 250 GB/s |
| ALL-Reduce 带宽（8×H100 单机，启用 `NCCL_NVLS_ENABLE`） | 450-480 GB/s | 350-449 GB/s | < 350 GB/s |
| NVLink 链路带宽 | 50 GB/s 每链路 | 25-49 GB/s | < 25 GB/s |
| NVLink 重启次数/月 | 0 | 1-2 | > 2 |

> 注：Ring 与 NVLS（NVLink SHARP）是两套基线，先确认是否设置了 `NCCL_NVLS_ENABLE`，再对照相应一行——未开 NVLS 的健康机器不该被判成"预警"。

### 💡 一句话记住

> **"NVLink 是 GPU 之间的专属高速路——18 车道，900 GB/s（双向）。但它只修到机箱门口，出门就得走网络那条乡间小道（单卡单向带宽掉到约 1/9）。"**

---

## 对后续的铺垫

NVLink 解决了 8 卡机内的直连，**但 8 张卡之间怎么做到"任意两张都全带宽通信"？** 如果每张卡只有 18 条 NVLink，怎么连 8 张卡形成一个全互联的网络？

这就需要 **NVSwitch**——一个专门为 GPU 通信设计的交换芯片。

→ [[03-NVSwitch为什么出现]]

---

> **参考资料**：
> - [NVIDIA 第三代 NVSwitch 技术博客：多 GPU 互联升级](https://developer.nvidia.com/blog/upgrading-multi-gpu-interconnectivity-with-the-third-generation-nvidia-nvswitch/)
> - [PCI Express 6.0 规范 (PCI-SIG)](https://pcisig.com/pci-express-6.0-specification)
> - [NVIDIA DGX H100 用户指南](https://docs.nvidia.com/dgx/dgxh100-user-guide/introduction-to-dgxh100.html)
> - [ServeTheHome: NVIDIA NVLink4 / NVSwitch at Hot Chips 34](https://www.servethehome.com/nvidia-nvlink4-nvswitch-at-hot-chips-34/)
