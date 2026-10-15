# Q: NVLink 够用了吗？NVSwitch 为什么出现？

## 为什么会有这个问题

NVLink 很强大，但有个现实问题：**GPU 之间不是只做"点对点"通信的**。

在数据并行训练中，所有 GPU 需要做 All-Reduce 同步梯度——这意味着**每张卡都要和所有其他卡通信**（[[01-基础篇/02-为什么不用多卡直接跑]]）。

### 8 张卡需要多少条 NVLink 才能全互联？

如果每张卡直接连到其他 7 张卡：
- 总共需要的连接数：C(8,2) = **28 条**
- 每张卡需要：**7 个 NVLink 端口**

H100 有 18 个 NVLink 端口，理论上足以点对点连接 7 张卡。但**真实的 DGX H100 并不是这么做的**——它使用 **4 颗第三代 NVSwitch（NVSwitch 3.0）**，每颗提供 64 个 NVLink 4.0 端口，**GPU 之间没有任何直连**。历史上不用 NVSwitch、靠 NVLink 直连的是 **DGX-1**；从 **DGX-2** 起 NVIDIA 就引入了 NVSwitch。

但问题是：**GPU 数量继续增加会怎样？**

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

NVLink 只能在同一机箱内的 GPU 间直连。跨机箱时，又回到了 PCIe 或网络。

---

## 解决什么问题：NVSwitch

**NVSwitch** 是一个专门为 GPU 间通信设计的**全互联交换机**，部署在机箱内部。

### 工作原理

```
NVSwitch方案（DGX H100，4 颗 NVSwitch 全交叉）：
                         ┌──────────┐
                    ┌────┤ NVSwitch ├────┐
                    │    │   × 4    │    │
            ┌───────┼────┤         ├────┼───────┐
            │       │    └──────────┘    │       │
          GPU 0   GPU 1   ...        GPU 6   GPU 7
      (18 NVLink) (18)              (18)    (18)

每张 GPU 用 18 条 NVLink 上行；4 颗 NVSwitch 共 256 个端口，
承载 8 GPU × 18 = 144 条链路，任意 GPU 之间都是全交叉
总带宽：任意 GPU ↔ 任意 GPU = 900 GB/s（双向）
```

关键：**NVSwitch 不是一张"卡"，它是一个独立的、专用的交换芯片。**

| 规格 | NVSwitch 3 (H100) | NVSwitch 4 (B200) |
|------|-------------------|-------------------|
| 端口数 | 每芯片 64 个 NVLink 4.0 | 每芯片 72 个 NVLink 5.0（一个 switch tray 内两颗芯片合计 144） |
| 单芯片交换吞吐 | 13.6 Tb/s（上一代 7.2 Tb/s） | **57.6 Tb/s ≈ 7.2 TB/s** |
| 延迟 | ~100 ns | <100 ns |
| 功耗 | ~75W | ~100W |

> 注：57.6 Tb/s 是**系统级**数字（DGX H100 内 8 张 GPU × 18 链路 × 50 GB/s 的全交叉聚合），不是单颗 NVSwitch 的规格；单颗 NVSwitch 3 的交换吞吐是 13.6 Tb/s。另外，交换吞吐的单位是 **Tb/s**（比特），不是 TB/s（字节）。

NVSwitch 让 DGX 机箱内的 8 张 GPU 形成一个**全互联的超大"虚拟 GPU"**——任何两张卡之间的通信都是 NVLink 级带宽，没有"绕路"的惩罚。

### NVSwitch 如何扩展到更大规模？

NVIDIA 可以连接多个 NVSwitch 组成更大的交换平面。DGX SuperPOD 就是通过二级 NVSwitch 互联将 32 个 DGX（256 张 GPU）整合为一个超大计算域：

```
DGX × 32 (256 GPU)
  │           │
8×NVSwitch   8×NVSwitch  ...
  │           │
  └─────┬─────┘
   二级 NVSwitch fabric
    → 256 GPU 全互联
```

---

## 优缺点

### NVSwitch 优点
| 优势 | 说明 |
|------|------|
| 任意 GPU 间全带宽 | 无"近远"GPU 之分，通信性能一致 |
| 简化拓扑管理 | NCCL 不需要操心路由，直接通过 NVSwitch |
| 可扩展 | 多级 NVSwitch 可以构建超大交换平面 |

### NVSwitch 缺点
| 劣势 | 说明 |
|------|------|
| 成本极高 | NVSwitch 芯片独立设计，DGX 机箱价格包含这部分成本 |
| **仅限机箱内** | 跨机箱还是要走 InfiniBand/以太网 |
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
| 多机多卡（32+ GPU） | ✅ 每机箱内需要 | 机箱内用 NVSwitch，机箱间用 IB/RoCE |
| GB200 NVL72 / GB300 NVL72 | ✅ **NVSwitch 4 必须** | 72 GPU 全互联，全靠 NVSwitch |

**关键点**：如果你只在单机上跑 2-4 卡、且不做张量并行（TP），NVSwitch 的价值不明显。但一旦你开始做模型并行（8 卡 TP 或 32+ 卡 DP），NVSwitch 的**任意 GPU 间全带宽**能力直接决定了你的 MFU。

### 🔧 自己动手验：你的 NVSwitch 工作正常吗？

```bash
# 1. 检查 NVLink 链路状态（GPU 侧视角，通用）
nvidia-smi nvlink -s
# 所有链路应为 Active；Disabled 说明该链路未起来
# 看错误计数：nvidia-smi nvlink -e（CRC/replay 错误应为 0）

# 2. 检查链路吞吐与拓扑
nvidia-smi nvlink -g 0 -t        # 每张卡的 NVLink 吞吐
nvidia-smi topo -m               # GPU 间是 NV# 还是 SYS/PIX

# 3. NVSwitch 本身的健康状态需要平台工具（标准 nvidia-smi 没有 nvswitch 子命令）：
#    · DGX 系统：nvsm show switch / nvsm show health（DGX OS 自带）
#    · 非 DGX：NVIDIA Fabric Manager（NVSwitch 管理服务，随机型提供）
#    · 或经 DCGM：dcgmi diag -r 3（含 NVLink/NVSwitch 诊断）
#    字段关注：Temperature（< 70°C）、Link Errors / CRC 计数（应为 0）
```

# 3. 更彻底的测试：全互联带宽基准
# 在 DGX 机箱内跑
mpirun -np 8 --allow-run-as-root \
  nccl-tests/build/all_reduce_perf -b 1G -e 1G -f 2
# H100 DGX 预期结果：350-450 GB/s（Ring）/ 450-480 GB/s（启用 NCCL_NVLS_ENABLE）
# 如果 < 350 GB/s（Ring）→ 可能是 NVSwitch 链路降速或 NCCL 配置问题

# 4. 检查 NCCL 是否识别了 NVSwitch 拓扑
NCCL_DEBUG=INFO mpirun -np 8 \
  nccl-tests/build/all_reduce_perf -b 128M -e 128M
# 在日志中搜索：
#   "NVLS is supported" → NVLink SHARP 可用
#   "Tree" → NCCL 选择了正确的算法
```

### ⚠️ 常见误区

- **误区**："NVSwitch = NVLink，有 NVLink 就不需要 NVSwitch"
  - **实际**：NVLink 是**连接线**的协议，NVSwitch 是**交换芯片**。NVLink 可以点对点直连，但超过 8 卡必须通过 NVSwitch 做全互联。**没有 NVSwitch，8 卡以上只能通过 PCIe 或网络绕路，单向带宽从 NVLink 的 450 GB/s 掉到 PCIe 5.0 x16 的 64 GB/s。**

- **误区**："NVSwitch 解决了跨机通信的问题"
  - **实际**：NVSwitch 只工作在**单机柜内**（8 卡 DGX 或 NVL72 的 72 卡）。跨机柜通信（DGX → DGX）还是要走 InfiniBand 或 RoCE。NVLink 5 每 GPU 单向带宽 900 GB/s，而跨机 IB NDR 只有 50 GB/s（单向，400 Gb/s）——**差了 18 倍**。这就是为什么大模型训练要把通信尽量限制在 NVSwitch 域内。

- **误区**："NVSwitch 是 NVIDIA 独有技术，其他厂商没法做类似的东西"
  - **实际**：AMD 有 Infinity Fabric，华为有 HCCS，Google TPU 用 3D Torus——每个厂商都有自己的互联技术。但 NVSwitch 的**大规模商用部署经验**（从 2018 年 DGX-2 到 2026 年 NVL72）是最成熟的。2025 年 OCP 成立了 ESUN（Ethernet for Scale-Up Networking）工作组，**12 家创始成员包括 NVIDIA、AMD、Arista、Arm、Broadcom、Cisco、HPE Networking、Marvell、Meta、Microsoft、OpenAI、Oracle**，目标是开放标准化 scale-up 网络；ESUN 1.0 规范已于 2026 年发布，未来可能动摇 NVSwitch 的专有地位。

### 📊 底线数字速查

| 指标 | NVSwitch 3（H100） | NVSwitch 4（B200） | NVSwitch 下一待（Rubin） |
|------|-------------------|-------------------|------------------------|
| 每 GPU NVLink 带宽 | 900 GB/s（双向） | **1,800 GB/s（双向）** | 待公布 |
| 每交换芯片端口数 | 64 端口 | **72 端口**（一个 switch tray 两颗芯片合计 144） | 待公布 |
| 单芯片交换吞吐 | 13.6 Tb/s | **57.6 Tb/s ≈ 7.2 TB/s** | 待公布 |
| 延迟 | ~100 ns | **<100 ns** | 待公布 |
| 最大单域 GPU 数 | 256（二级交换） | **72（NVL72）** | 待公布 |
| 功耗 | ~75W | ~100W | 待公布 |
| 跨域瓶颈 | IB NDR ~50 GB/s（单向，单卡） | IB NDR ~50 GB/s（单向，单卡） | CX-9 1.6T |
| 价格 | DGX 机箱包含 | DGX 机箱包含 | 待公布 |

### 💡 一句话记住

> **"NVSwitch 是 GPU 之间的'高速公路立交桥'——8 卡以下靠直连，8 卡以上没有立交桥就得绕 PCIe 小路。但记住，这座桥只在机箱内免费通行，出机箱还是要交 InfiniBand 过路费。"**

---

## 对后续的铺垫

NVSwitch 很好地解决了**单机内**所有 GPU 的高速通信问题。但当你需要从 8 卡扩展到 1,000 卡时，**跨机通信**怎么解决？

NVLink/NVSwitch 的单域规模是有上限的——Hopper 时代（NVLink 4 + NVSwitch 3）通过二级交换最高做到 256 卡全互联，Blackwell 的 NVLink 5 单域则是 72 卡（NVL72）。再往上怎么办？

→ **这就是 03-网络篇 要回答的问题**：机间通信靠 InfiniBand 或 RoCE。

[[03-网络篇/01-为什么GPU之间需要网络]]

---

> **参考资料**：
> - [NVIDIA GB200 NVL72 官方页面](https://www.nvidia.com/en-us/data-center/gb200-nvl72/)
> - [NVIDIA DGX SuperPOD 用户指南](https://docs.nvidia.com/dgx-superpod/)
> - [ServeTheHome: NVIDIA NVLink4 / NVSwitch at Hot Chips 34](https://www.servethehome.com/nvidia-nvlink4-nvswitch-at-hot-chips-34/)
