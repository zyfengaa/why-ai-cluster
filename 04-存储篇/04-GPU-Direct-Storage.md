# Q: GPU Direct Storage 是什么？

## 为什么会有这个问题

GPU 要读取数据时，传统流程是这样的：

```
传统 I/O 路径：
硬盘 ──→ 系统内存（DDR，bounce buffer）──→ 应用缓冲 ──→ GPU 显存（HBM）
   ↑              ↑                            ↑              ↑
  NVMe          PCIe 传输                    CPU 搬运        PCIe 传输
(~7 GB/s 单盘)  (PCIe 5.0 x16 双向 128 GB/s；               (同上；单向 64 GB/s)
                 单向 64 GB/s)
```

数据从盘到显存要经过**系统内存中转（bounce buffer）和 CPU 多次搬运拷贝**。CPU 在这个过程中只是"快递员"——但每次中转都有拷贝开销和 CPU 占用，而且大块顺序传输时，中转本身就是吞吐天花板。

## 解决什么问题

**GPU Direct Storage（GDS）** 让存储（本地 NVMe 或支持 GDS 的并行文件系统）可以**直接向 GPU 显存传输数据**，跳过系统内存中的 bounce buffer。

```
GDS 直接路径：
NVMe SSD ──→ GPU HBM（直通 DMA！）
     ↑           ↑
   PCIe 传输    单次 PCIe 传输
                CPU 不做数据搬运
```

### GDS 为什么快

1. **DMA 直通**：存储设备通过 PCIe 直接 DMA 到 GPU 显存，省掉系统内存中转；
2. **减少拷贝**：传统路径中"盘 → 内核缓冲 → 用户缓冲 → GPU"的多次拷贝被砍掉。注意：**绕过 CPU ≠ 字面意义的零拷贝**——GPU 端缓冲仍需注册（`cuFileBufRegister`），数据传输本身仍是一次 DMA；"零拷贝"指的是省掉 DRAM 中转，不是无任何内存操作；
3. **CPU 释放**：CPU 不再做 I/O 搬运工，可以专心做数据预处理和调度。

### 统一的预期收益（一组口径，全文只用这一组）

| 指标 | 传统路径 | GDS | 说明 |
|------|---------|-----|------|
| **单 GPU 顺序读吞吐**（本地 NVMe，1 MB 顺序块） | ~8 GB/s | **~20 GB/s** | 2.5×；单客户端口径 |
| 10 GB 顺序读耗时 | ~1.25 s | **~0.5 s** | 与上行同口径（10/8 与 10/20） |
| CPU 占用（I/O 期间） | 30-50% | 5-10% | 释放给数据预处理 |

**适用范围要划清楚**：GDS 的优势在**顺序大块 I/O**。小文件、随机读通常没有稳定收益——缓冲注册、对齐要求的开销甚至会让它比传统路径更慢。如果你的瓶颈是小文件元数据，先去做分片打包（06 篇），GDS 救不了你。

### 2026 年的 GDS 生态与 PyTorch 现状

| 组件 | 现状 |
|------|------|
| cuFile API（C 层） | `cuFileRead/Write`、异步 `cuFileReadAsync/WriteAsync`、batch API、缓冲注册 API，配套 `nvidia-fs` 内核驱动 |
| 官方工具 | `gdscheck.py`（检查环境是否支持）、`gdsio`（测 GDS 吞吐），随 GDS 包安装（通常在 `/usr/local/cuda/gds/tools/`） |
| 本地 NVMe | 支持（ext4/xfs 等常见文件系统） |
| WEKA | 原生支持 |
| Lustre | 较新社区版本支持（需确认版本与配置） |
| GPFS / Storage Scale、BeeGFS | 有支持（BeeGFS 7.3+），均需确认客户端/内核/驱动组合 |
| **PyTorch** | 截至本教程核对时点（PyTorch 2.7），GDS 仅以**原型接口** `torch.cuda.gds` 提供（`GdsFile`、`gds_register_buffer` / `gds_deregister_buffer`）；`torch.distributed.checkpoint` 的 GDS 集成**不是稳定公开 API**——网上流传的 `torch.distributed.checkpoint.GDS`、`gds.mmap` 都不存在，照抄会直接 ImportError |

### 部署前提

GDS 不是"装个驱动就能用的"：

```
需要满足的条件：
1. GPU：NVIDIA GPU（Ampere 及之后架构支持较完整）
2. 存储：本地 NVMe，或支持 GDS 的文件系统（见上表）
3. nvidia-fs 内核驱动 + cuFile 运行库（随 GDS 包/CUDA Toolkit 安装）
4. 主板：Above 4G Decoding（部分场景需要 Resizable BAR）
5. PCIe 拓扑：GPU 与 NVMe/存储网卡尽量在同一 root complex，
   跨 NUMA/跨 PCIe switch 会吃掉直通收益
6. 网络：本地 NVMe 场景不涉及网络；远程并行文件系统场景
   用的是该文件系统自己的网络（IB/RoCE），GDS 本身不规定网络协议
```

## 🛠 实操指南

### 📋 决策指南：你的场景需要 GDS 吗？

**GDS 不是免费的提速——先确认瓶颈确实是"大块顺序 I/O 经过 CPU 中转"：**

```
硬性前提条件：
□ GPU：NVIDIA Ampere 及之后架构
□ 存储：本地 NVMe 或支持 GDS 的文件系统（版本/组合逐项确认）
□ nvidia-fs 驱动 + cuFile 已安装
□ 用 gdscheck.py 验证环境 → PASSED 再谈收益

收益判断：
□ 你的负载是顺序大块 I/O（checkpoint 分片、tar 数据分片、模型加载）？
  → GDS 大概率有效
□ 你的瓶颈是小文件/随机读/元数据？
  → 先去打包分片，GDS 无效甚至更差
□ 用 06 篇的方法测过 data wait 占比 > 10%，且磁盘顺序带宽未跑满？
  → 才轮到 GDS 上场
```

| 场景 | 预期收益 | 是否值得 | 理由 |
|------|---------|---------|------|
| 小规模微调（1-2 卡） | 不明显 | ❌ 不值得 | 数据加载不是瓶颈 |
| 单机 8 卡训练 | 适度 | ⚠️ 看情况 | 数据加载已够快则边际收益有限 |
| 多机训练、顺序大块读为瓶颈 | 明显（约 2.5× 该环节） | ✅ 推荐 | 走 06 篇实测流程确认后上 |
| 大模型 checkpoint 写 | 明显 | ✅ 推荐 | 配合异步 checkpoint 效果最好 |
| 长上下文推理 KV Cache offload | 场景性显著 | ✅ 值得评估 | KV cache 超出显存/DRAM 后 NVMe 直送 HBM 是标准做法 |

### 🔧 自己动手验：你的系统支持 GDS 吗？

```bash
# 1. 检查 PCIe 拓扑：GPU 与 NVMe 是否在同一 root complex
nvidia-smi topo -m
# GPU 与 NVMe/网卡跨 socket 时，直通收益会打折

# 2. 检查 nvidia-fs 驱动是否加载
lsmod | grep nvidia_fs
# 输出为空 → 需要安装 GDS 包（含 nvidia-fs 内核模块）

# 3. 官方环境自检工具
python /usr/local/cuda/gds/tools/gdscheck.py
# 输出各环节（驱动、cuFile、GPU、存储）的支持情况
# 全部 PASSED → 环境就绪；具体路径以你安装的 GDS 包为准

# 4. 测 GDS 顺序吞吐，并与传统路径对比
# 传统路径基线（单客户端）：
dd if=/mnt/nvme/data.bin of=/dev/null bs=1M count=10000
# GDS 路径：用 gdsio 跑顺序读/写并对比
# （用法见 NVIDIA GDS Troubleshooting Guide，链接见参考资料）
# 合理预期：1 MB 顺序块下约 2.5×（8 → 20 GB/s 量级）

# 5. PyTorch 侧确认原型接口存在（2.7+）
python -c "import torch; print(hasattr(torch.cuda, 'gds'))"
# True → torch.cuda.gds 原型接口可用
# 注意：这只是原型，DCP 的 GDS 集成不是稳定公开 API，不要照抄网上的
# torch.distributed.checkpoint.GDS 代码——它不存在
```

### ⚠️ 常见误区

- **误区**："GDS 是个软件功能，装个驱动就能用"
  - **实际**：GDS 需要环境配合：nvidia-fs 驱动、cuFile、BIOS 的 Above 4G Decoding、合理的 PCIe 拓扑。先用 `gdscheck.py` 自检，任何一环缺失都会静默回退到兼容模式（bounce buffer），你以为在走直通，其实在原地踏步。
- **误区**："GDS 提升一切 I/O"
  - **实际**：GDS 的收益集中在**顺序大块 I/O**。小文件/随机读没有稳定提升，缓冲注册与对齐开销甚至让它更慢。小文件问题先做分片打包，不是 GDS 的靶子。
- **误区**："CPU 绕过了，就是零拷贝"
  - **实际**："绕过 CPU" 指数据不再经系统内存中转，但 GPU 缓冲仍需注册与对齐，兼容模式下还会回退到 bounce buffer。零拷贝是路径特征，不是性能保证——收益要实测。
- **误区**："GDS 让我的 1.12 TB checkpoint 秒存"
  - **实际**：算清口径。单卡视角：32 卡分片保存 1.12 TB，每卡分片 35 GB，单卡顺序写从 ~8 GB/s（4.4 秒）提到 ~20 GB/s（1.75 秒）——收益真实但只有 2.5×。聚合视角：1.12 TB 总量不变，**整体保存时间取决于存储后端的聚合写带宽**（38-44 GB/s 量级 → 约 25-29 秒），单卡 GDS 带宽再高也变不了一秒。GDS 与 05 篇的异步 checkpoint 配合才是完整方案。

### 📊 底线数字速查

| 性能指标 | 传统路径（无 GDS） | GDS 开启 | 条件 |
|---------|------------------|---------|------|
| 单 GPU 顺序读（本地 NVMe，1 MB 块） | ~8 GB/s | **~20 GB/s** | 单客户端口径 |
| 10 GB 顺序读 | ~1.25 s | **~0.5 s** | 同上 |
| CPU 占用（I/O 期间） | 30-50% | 5-10% | 释放给预处理 |
| 32 卡保存 1.12 TB（每卡 35 GB 分片） | 每卡 ~4.4 s | 每卡 ~1.75 s | 聚合耗时仍受后端聚合带宽约束（~25-29 s） |
| 小文件/随机读 | 基线 | 无稳定收益，可能更差 | 先做分片打包 |

### 💡 一句话记住

> **"GDS 砍掉的是'盘 → DRAM → GPU'的中间人：顺序大块 I/O 约 2.5×，CPU 解放出来做预处理。但它不是万能加速器——小文件、随机读、单点聚合瓶颈都不归它管。先用 gdscheck 验环境，再用 data wait 占比确认瓶颈，最后记住：单卡带宽翻倍 ≠ 1.12 TB 的 checkpoint 秒存。"**

---

## 对后续的铺垫

GDS 让数据加载变得更快，但还有一个更大的挑战：**训练过程中，Checkpoint 的保存正在悄悄地浪费 GPU 时间**，而且规模越大越严重。

→ [[05-Checkpoint隐形杀手]]

---

> **参考资料**：
> - [NVIDIA GPUDirect Storage 文档](https://docs.nvidia.com/gpudirect-storage/)
> - [GDS cuFile API Reference](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html)
> - [GDS Troubleshooting Guide（含 gdscheck/gdsio 用法）](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html)
> - [PyTorch 2.7 CUDA 文档（torch.cuda.gds 原型接口）](https://pytorch.org/docs/2.7/cuda.html)
> - [PyTorch Distributed Checkpoint (DCP) 官方文档](https://pytorch.org/docs/stable/distributed.checkpoint.html)
