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
| 单 GPU 顺序读吞吐 | 12 GB/s | 25 GB/s | **2.1×** |
| CPU 利用率 | 30-50% | 5-10% | **降低 4-6×** |
| Checkpoint 写时间（10GB） | ~1.25s | ~0.5s | **2.5× 更快** |
| 端到端数据加载延迟 | 高 | 低 30-50% | — |

### GDS 为什么快

1. **DMA 直通**：NVMe 控制器通过 PCIe 直接 DMA（直接内存访问）到 GPU HBM
2. **零拷贝**：数据不经过 CPU 缓存，不存在"从用户态缓冲区拷贝到内核态缓冲区"的开销
3. **CPU 释放**：CPU 不用做 I/O 搬运工，可以专心做数据预处理和模型调度

---

## 2026 年的 GDS 生态

| 状态 | 说明 |
|------|------|
| NVIDIA 支持 | GDS 自 CUDA 11.4 GA 起支持；compute capability > 6 的 GPU（T4/V100/A100 起）均可用，H100 起只是**新增原生支持** |
| WEKA | **官宣 GDS 原生认证** |
| Lustre | 社区版可经 `nvidia-fs` 支持；**DDN EXAScaler**、**AWS FSx for Lustre** 为官方支持项 |
| IBM Storage Scale（原 GPFS） | 支持（`nvidia-fs`） |
| 框架支持 | PyTorch **2.7** 起提供 `torch.cuda.gds.GdsFile`（实验性/prototype），配合 `load_storage()` / `save_storage()` / `is_available()` / `gds_register_buffer()` |

---

## 部署前提

GDS 不是"装个驱动就能用的"：

```
需要满足的条件：
1. GPU：compute capability > 6 的 NVIDIA GPU 均可（T4/V100/A100 起）；H100 起新增原生支持
2. 存储：本地 NVMe 上的 ext4 / XFS 即可；RDMA 分布式文件系统（DDN EXAScaler、WekaFS、VAST、IBM Storage Scale、BeeGFS）或 NFS over RDMA 也可
3. 网络：本地 NVMe → GPU HBM 路径**不需要网络**；仅远端/分布式路径才需要 RDMA 网卡（ConnectX-6 Dx 或更新）+ MOFED
4. CPU：需要支持 Resizable BAR 和 Above 4G Decoding
5. 系统：Linux 4.15.0+ 内核（覆盖 RHEL 8 / Ubuntu 20.04 等），nvidia-fs 驱动
```

---

## 🛠 实操指南

### 📋 决策指南：你的场景需要 GDS 吗？

**GDS 不是免费的提速——它有硬件门槛。先检查你是否满足条件：**

```
硬性前提条件：
□ GPU：compute capability > 6（T4/V100/A100 起均支持；H100 起原生支持，B200/GB300 最佳）
□ 存储系统：本地 NVMe（ext4/XFS）即可，或 RDMA 分布式文件系统（WekaFS / DDN EXAScaler / VAST / IBM Storage Scale / BeeGFS）；GDS 不要求并行文件系统
□ 网络：仅远端/分布式路径需要 RDMA 网卡 + MOFED（本地 NVMe → HBM 不需要网络）
□ CPU 主板：支持 Resizable BAR 和 Above 4G Decoding
□ 内核：Linux 4.15.0+，nvidia-fs 驱动
□ CUDA 版本：11.4+（GDS 2.0 需 12.3+）

软件框架支持：
□ PyTorch 2.7+（内置 torch.cuda.gds.GdsFile，实验性）
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
| 超长上下文推理（KV Cache offload） | **40×（延迟）** | ✅ **强烈推荐** | CoreWeave 实测 Llama 3 70B 延迟从 24 秒降到 0.58 秒（案例来源待补，未经核实） |
| 万亿参数模型预训练 | **20× + 降低 CPU 占用** | ✅ **必须** | 少 1% GPU 空闲=节省数百万美元 |

### 🔧 自己动手验：你的系统支持 GDS 吗？

```bash
# 1. 检查 GPU 与存储/NIC 的 PCIe 亲和性（GDS 与 NVLink 无关）
nvidia-smi topo -m
# 确认 GPU 与存储/NIC 在同一 PCIe root complex、同一 PCIe switch（避免跨 socket）

# 2. 检查 GDS 驱动是否安装
lsmod | grep nvidia_fs
# 如果输出为空 → 需要安装 GDS 用户态工具与内核模块
# 安装：通过 CUDA 网络仓库安装元包 nvidia-gds（含 nvidia-fs-dkms、gds-tools）
#   sudo apt install nvidia-gds

# 3. 检查 Resizable BAR 是否启用（看 BAR1 大小）
nvidia-smi -q | grep -A2 "BAR1"
# 或：lspci -vv -s <GPU_BDF> | grep "Region 1"
# BAR1 为几十 GB 量级 → Resizable BAR 已启用
# 若 BAR1 只有 256M → Resizable BAR 未启用，需要在 BIOS 中开启

# 4. 运行 GDS 官方工具（来自 nvidia-gds / gds-tools 包，不在 CUDA samples 下）
gdscheck -p          # 环境自检：列出 nvidia-fs 与各文件系统的支持状态
gdsio -f /path/to/data.bin -d 0 -w 8 -s 1G -i 1M -x 0 -I 1   # 性能测试
# gdscheck 报告全部支持、gdsio 吞吐明显高于裸 IO → GDS 配置正确
# 若 gdscheck 提示 fallback / not supported → 检查上述步骤

# 5. 对比 GDS 开/关的性能差异
# 关 GDS（默认路径）：
fio --name=read --rw=read --bs=1M --direct=1 --numjobs=8 --iodepth=32 \
    --filename=/path/to/data.bin --runtime=30 --time_based --group_reporting

# 开 GDS（通过 GDS 直接读取）：
export CUDA_VISIBLE_DEVICES=0
python -c "
import os, torch
from torch.cuda import gds
assert gds.is_available()
t = torch.empty(1024**3, dtype=torch.uint8, device='cuda')   # 1 GiB on GPU
f = gds.GdsFile('/path/to/data.bin', os.O_RDWR)
f.load_storage(t.untyped_storage(), 0)                       # 直通读入显存
print(f'GDS 读取: {t.numel()/1024**3:.1f} GiB')
"
# 注：缓冲区需 4KB 对齐（torch.utils.serialization.config.save.storage_alignment = 4096）
```

### ⚠️ 常见误区

- **误区**："GDS 是个软件功能，装个驱动就能用"
  - **实际**：GDS 需要**硬件级支持**——其中 PCIe Resizable BAR / Above 4G Decoding 是 **GPU + 主板/BIOS** 侧的问题（决定 BAR1 能否暴露足够大的窗口）；NVMe SSD 与网卡本身不涉及 RBAR。很多老主板在 BIOS 里默认关闭了这些功能。**GDS 不是"装个软件"就能用的，可能需要更新 BIOS、重新安装驱动、甚至换主板。**

- **误区**："`nvidia-smi` 正常、`nvidia-fs` 也装了，GDS 就一定生效了"
  - **实际**：不一定。当文件系统或传输路径不被支持时，cuFile 会**静默回退到兼容模式（compatibility mode / POSIX bounce buffer）**——数据仍经 CPU 内存中转，性能与不开 GDS 几乎无差别。需在 `/etc/cufile.json` 关闭 `allow_compat_mode`，并用 `gdscheck` 验证确实走了直通路径。

- **误区**："GDS 提升了 Checkpoint 写入速度，但我的 Checkpoint 本来就够快了"
  - **实际**：不，你的 Checkpoint 大概率不够快。一个 70B 模型的完整 Checkpoint ≈ 1.12 TB。用传统路径写入 10 GB 需要约 1.25 秒看起来很快，但 GDS 只需 0.5 秒——**每写一次省约 0.75 秒**。如果每天写 32 次 Checkpoint，省下约 24 秒看起来不多。但如果你用的是 40 GB 的 Checkpoint？8 次？GDS 的收益随模型规模和集群规模放大。**在万亿参数模型上，一次 Checkpoint 就能省几分钟。**

- **误区**："GDS 只提升训练速度，推理用不上"
  - **实际**：**推理场景中 GDS 的价值可能更大**。CoreWeave 2025 年的测试：在 Llama 3 70B（128K context）推理时，用 GDS + NVMe 做 KV Cache offload，首字延迟（TTFT）从 **23.97 秒降到 0.58 秒**——降低了 40 倍。对于长上下文推理，GDS 是让模型"记住"更大上下文的关键技术，**比在训练场景的收益更显著**。（注：上述 CoreWeave 数据尚未经独立核实，来源待补。）

### 📊 底线数字速查

| 性能指标 | 传统路径（无 GDS） | GDS 开启 | 提升 |
|---------|-----------------|---------|------|
| 单 GPU 顺序读吞吐 | 12 GB/s | **25 GB/s** | **2.1×** |
| 单 GPU Checkpoint 写入 | 8 GB/s | **20 GB/s** | **2.5×** |
| CPU 利用率（I/O 期间） | 30-50% | **5-10%** | 降低 **4-6×** |
| 小文件随机读 IOPS（单节点） | ~500K IOPS | **~3M IOPS**（企业级 NVMe 单盘水平） | **~6×** |
| 模型加载（405B，AWS FSx Lustre） | 18 分钟 | **6.4 秒** | **169×** |
| KV Cache offload（70B，128K context） | 24 秒 | **0.58 秒** | **40×**（案例来源待补） |
| DeepNVMe FastPersist Checkpoint | Baseline | **20× 更快** | **20×** |

### 💡 一句话记住

> **"GDS 让 NVMe 到 HBM 之间的数据走'高速公路'而不是'乡间小路'——直接跳过了 CPU 这个收费站。数据加载延迟降 40-60%，CPU 占用从 50% 降到 5%，Checkpoint 从分钟级压缩到秒级。2026 年的万卡集群，没有 GDS 就等于在烧钱。"**

---

## 对后续的铺垫

GDS 让数据加载变得更快，但还有一个更大的挑战：**训练过程中，Checkpoint 的保存正在悄悄吃掉大量 GPU 时间**，而且规模越大越严重。

→ [[08-运维观测篇/05-训练断了怎么办]]

---

> **参考资料**：
> - [NVIDIA GPUDirect Storage 官方文档](https://docs.nvidia.com/gpudirect-storage/)
> - [nvidia-fs 支持矩阵](https://docs.nvidia.com/gpudirect-storage/nvidia_fs/index.html)
> - [GPUDirect Storage 排障手册](https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html)
> - [WEKA 官方文档](https://docs.weka.io/)
> - [PyTorch: GPU Direct Storage（torch.cuda.gds）](https://docs.pytorch.org/tutorials/unstable/gpu_direct_storage.html)
> - [gds-nvidia-fs 源码](https://github.com/NVIDIA/gds-nvidia-fs)
> - [DeepSpeed DeepNVMe: "Affordable I/O Scaling for Deep Learning Applications"](https://pytorch.org/blog/deepnvme-affordable-i-o-scaling-for-deep-learning-applications/)
> - [AWS: "Accelerate LLM model loading and increase context windows with GPUDirect on Amazon FSx for Lustre and TurboQuant"](https://aws.amazon.com/blogs/machine-learning/accelerate-llm-model-loading-and-increase-context-windows-with-gpudirect-on-amazon-fsx-for-lustre-and-turboquant/)
