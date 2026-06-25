# Q: GPU Direct Storage 是什么？

## 为什么会有这个问题

GPU 要读取数据时，传统流程是这样的：

```
传统 I/O 路径：
硬盘 ──→ 系统内存（DDR）──→ CPU 处理 ──→ GPU 显存（HBM）
    ↑            ↑              ↑              ↑
  NVMe      PCIe 传输       CPU 不可少      PCIe 传输
  (~7 GB/s)  (~32 GB/s)                    (~64 GB/s)
```

走了**两个 PCIe 往返**、**一次 CPU 中转**。CPU 在这个过程中只是"快递员"——但每次中转都有延迟。

## 解决什么问题

**GPU Direct Storage（GDS）** 让 NVMe 存储可以**直接向 GPU 显存传输数据**，完全跳过 CPU 内存。

```
GDS 直接路径：
NVMe SSD ──→ GPU HBM（直通！）
     ↑           ↑
  PCIe 传输    单次 PCIe 传输
               CPU 完全绕开
```

### 性能提升

| 指标 | 传统路径 | GDS | 提升 |
|------|---------|-----|------|
| 8 GPU 读吞吐 | 8 GB/s | 20 GB/s | **2.5×** |
| CPU 利用率 | 60-80% | 10-15% | **降低 4-6×** |
| Checkpoint 写时间（10GB） | ~1.25s | ~0.3s | **4× 更快** |
| 端到端数据加载延迟 | 高 | 低 30-50% | — |

### GDS 为什么快

1. **DMA 直通**：NVMe 控制器通过 PCIe 直接 DMA（直接内存访问）到 GPU HBM
2. **零拷贝**：数据不经过 CPU 缓存，不存在"从用户态缓冲区拷贝到内核态缓冲区"的开销
3. **CPU 释放**：CPU 不用做 I/O 搬运工，可以专心做数据预处理和模型调度

---

## 2026 年的 GDS 生态

| 状态 | 说明 |
|------|------|
| NVIDIA 支持 | H100 系列起全面支持 |
| WEKA | **官宣 GDS 原生认证**，200+ 文件级并发 |
| Lustre | 社区版有限支持，企业版（DDN EXAScaler）支持 |
| GPFS | 有 GDS 认证但测试覆盖不完整 |
| 框架支持 | PyTorch 2.4+ 自带 `torch.distributed.checkpoint.GDS` 支持 |

---

## 部署前提

GDS 不是"装个驱动就能用的"：

```
需要满足的条件：
1. GPU：NVIDIA H100 或更高（Ampere 支持有限）
2. 存储：WEKA 或 Lustre（企业版）
3. 网络：InfiniBand NDR 或 RoCEv2（最低 100 Gb/s）
4. CPU：需要支持 Resizable BAR 和 Above 4G Decoding
5. 系统：Linux 5.15+ 内核，nvidia-fs 驱动
```

---

## 🛠 实操指南

### 📋 决策指南：你的场景需要 GDS 吗？

**GDS 不是免费的提速——它有硬件门槛。先检查你是否满足条件：**

```
硬性前提条件：
□ GPU：H100 或更高（Ampere A100 支持有限，B200/GB300 最佳）
□ 存储系统：WEKA（原生支持）或 Lustre 企业版
□ 网络：InfiniBand NDR 或 RoCEv2（最低 100 Gb/s）
□ CPU 主板：支持 Resizable BAR 和 Above 4G Decoding
□ 内核：Linux 5.15+，nvidia-fs 驱动
□ CUDA 版本：12.3+（GDS 2.0）

软件框架支持：
□ PyTorch 2.4+（内置 torch.distributed.checkpoint.GDS）
□ DeepSpeed 0.17.1+（内置 FastPersist）
□ SGLang / vLLM（KV Cache offload 支持）
```

**收益 vs 成本判断：**

| 场景 | 预期提速 | 是否值得部署 GDS | 理由 |
|------|---------|----------------|------|
| 小规模微调（1-2 卡） | 不明显 | ❌ 不值得 | 数据加载不是瓶颈 |
| 单机 8 卡训练 | 适度（~20%） | ⚠️ 看情况 | 如果数据加载已经够快，GDS 边际收益有限 |
| 多机 32+ 卡分布式训练 | **显著（2-4×）** | ✅ **强烈推荐** | 数据加载成为瓶颈，GDS 释放 CPU 做计算 |
| Checkpoint 频繁的大模型训练 | **巨大（4-20×）** | ✅ **必须** | Checkpoint 写入速度是 MFU 的关键瓶颈 |
| 超长上下文推理（KV Cache offload） | **40×（延迟）** | ✅ **强烈推荐** | CoreWeave 实测 Llama 3 70B 延迟从 24 秒降到 0.58 秒 |
| 万亿参数模型预训练 | **20× + 降低 CPU 占用** | ✅ **必须** | 少 1% GPU 空闲=节省数百万美元 |

### 🔧 自己动手验：你的系统支持 GDS 吗？

```bash
# 1. 检查 GPU 是否支持 GDS
nvidia-smi topo -m
# 查看 NVLink 和 PCIe 拓扑，确认 GPU 之间有 NVLink 连接
# 确认 GPU 和存储网卡在同一 PCIe root complex（避免跨 socket）

# 2. 检查 GDS 驱动是否安装
lsmod | grep nvidia_fs
# 如果输出为空 → 需要安装 nvidia-fs 驱动
# 安装：sudo apt install nvidia-fs 或从 NVIDIA 官网下载

# 3. 检查 Resizable BAR 是否启用
# 在 dmesg 中搜索
dmesg | grep -i "BAR"
# 如果有 "BAR0: 256G" 或类似大 BAR 地址 → Resizable BAR 已启用
# 如果只有 "BAR0: 16M" → Resizable BAR 未启用，需要在 BIOS 中开启

# 4. 运行 GDS 官方测试工具
# 安装 GDS 后，运行：
/usr/local/cuda/samples/5_Domains/GDS_serialize/
./GDS_serialize
# 如果输出 "PASSED" → GDS 配置正确
# 如果 "FAILED" → 检查上述步骤

# 5. 对比 GDS 开/关的性能差异
# 关 GDS（默认路径）：
dd if=/path/to/data.bin of=/dev/null bs=1M count=10000

# 开 GDS（通过 GDS 直接读取）：
export CUDA_VISIBLE_DEVICES=0
python -c "
from torch.cuda import gds
# GDS 读取大文件
data = gds.mmap('/path/to/data.bin')
print(f'GDS 读取: {len(data)/1024**3:.1f} GB')
"
```

### ⚠️ 常见误区

- **误区**："GDS 是个软件功能，装个驱动就能用"
  - **实际**：GDS 需要**硬件级支持**——要求 GPU（H100+）、NVMe SSD、网卡、CPU 主板四者全部支持 PCIe Resizable BAR 和 Above 4G Decoding。很多老主板在 BIOS 里默认关闭了这些功能。**GDS 不是"装个软件"就能用的，可能需要更新 BIOS、重新安装驱动、甚至换主板。**

- **误区**："GDS 提升了 Checkpoint 写入速度，但我的 Checkpoint 本来就够快了"
  - **实际**：不，你的 Checkpoint 大概率不够快。一个 70B 模型的 Checkpoint 是 420 GB。用传统路径写入 10 GB 需要约 1.25 秒看起来很快，但 GDS 只需 0.3 秒——**每写一次省 1 秒**。如果每天写 32 次 Checkpoint，省下 32 秒看起来不多。但如果你用的是 40 GB 的 Checkpoint？8 次？GDS 的收益随模型规模和集群规模放大。**在万亿参数模型上，一次 Checkpoint 就能省几分钟。**

- **误区**："GDS 只提升训练速度，推理用不上"
  - **实际**：**推理场景中 GDS 的价值可能更大**。CoreWeave 2025 年的测试：在 Llama 3 70B（128K context）推理时，用 GDS + NVMe 做 KV Cache offload，首字延迟（TTFT）从 **23.97 秒降到 0.58 秒**——降低了 40 倍。对于长上下文推理，GDS 是让模型"记住"更大上下文的关键技术，**比在训练场景的收益更显著**。

### 📊 底线数字速查

| 性能指标 | 传统路径（无 GDS） | GDS 开启 | 提升 |
|---------|-----------------|---------|------|
| 单 GPU 顺序读吞吐 | 12 GB/s | **25 GB/s** | **2.1×** |
| 单 GPU Checkpoint 写入 | 8 GB/s | **20 GB/s** | **2.5×** |
| CPU 利用率（I/O 期间） | 30-50% | **5-10%** | 降低 **4-6×** |
| 小文件随机读 IOPS | 500K IOPS/GPU | **1.2M IOPS/GPU** | **2.4×** |
| 模型加载（405B，AWS FSx Lustre） | 18 分钟 | **6.4 秒** | **169×** |
| KV Cache offload（70B，128K context） | 24 秒 | **0.58 秒** | **40×** |
| DeepNVMe FastPersist Checkpoint | Baseline | **20× 更快** | **20×** |

### 💡 一句话记住

> **"GDS 让 NVMe 到 HBM 之间的数据走'高速公路'而不是'乡间小路'——直接跳过了 CPU 这个收费站。数据加载延迟降 40-60%，CPU 占用从 50% 降到 5%，Checkpoint 从分钟级压缩到秒级。2026 年的万卡集群，没有 GDS 就等于在烧钱。"**

---

## 对后续的铺垫

GDS 让数据加载变得更快，但还有一个更大的挑战：**训练过程中，Checkpoint 的保存正在悄悄地浪费你 20% 以上的 GPU 算力**，而且规模越大越严重。

→ [[05-Checkpoint隐形杀手]]

---

> **参考资料**：
> - [NVIDIA GPUDirect Storage 官方文档](https://docs.nvidia.com/gpudirect-storage/)
> - [WEKA GDS 基准测试](https://www.weka.io/)
> - [PyTorch Distributed Checkpoint + GDS](https://pytorch.org/docs/stable/distributed.checkpoint.html)
