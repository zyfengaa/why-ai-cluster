# Q: 为什么 AI 训练用的是 GPU 而不是 CPU？

## 为什么会有这个问题

CPU（中央处理器）是计算机的大脑，什么都能做。但到了 AI 训练领域，GPU 却成了主角。为什么 "全能" 的 CPU 反而输给了 "专精" 的 GPU？

## 解决什么问题

先来看 AI 训练的核心计算是什么。以一次矩阵乘法为例：

```python
# 一个简单的神经网络层
y = W × x + b  # y: [4096], W: [4096×4096], x: [4096]
```

这就是 **4096 × 4096 = 1,677 万次乘加运算**。一个训练迭代中，这种操作要重复几亿次。

### CPU 的设计哲学 — 低延迟

```
CPU 核心结构：
┌──────────────────────────────┐
│  Control  │  ALU  │  Cache  │
│  (复杂)    │  (少)  │  (大)   │
└──────────────────────────────┘
```

- **大缓存**：减少访存延迟，快速响应单个任务
- **复杂控制单元**：分支预测、乱序执行、超线程
- **少量但强大的 ALU**：追求单线程性能极致

**CPU 适合**：逻辑判断、分支跳转、串行计算。就像一位全能数学家——什么都会，但一次只能算一道题。

### GPU 的设计哲学 — 高吞吐

```
GPU 核心结构（简化）：
┌──────────────────────────────┐
│  Control │ ALU │ ALU │ ALU  │
│  (简单)   │ ALU │ ALU │ ALU  │
│          │ ALU │ ALU │ ALU  │
└──────────────────────────────┘
```

- **大量 ALU**：一个 GPU 有数千个计算核心（H100 SXM 有 16,896 个 CUDA Cores，PCIe 版为 14,592 个）
- **简单控制单元**：不做复杂的指令调度，一次给所有 ALU 发相同指令
- **小缓存**：靠高并行度隐藏访存延迟

**GPU 适合**：大规模并行计算、矩阵运算、SIMD（单指令多数据）。就像一个万名小学生组成的方阵——每个人只能算简单加减法，但一秒钟能算出一亿道。

### 关键补充：算力表上的 TFLOPS 来自 Tensor Core

前面说的“数千个 CUDA Core”是标量/向量单元。AI 算力表里的 **989 TFLOPS（H100 BF16 稠密）不是 CUDA Core 贡献的，而是 Tensor Core**：

```
CUDA Core 做 FP32 FMA：H100 约 67 TFLOPS
Tensor Core 做 BF16 矩阵乘：H100 约 989 TFLOPS（稠密）
两者相差约 15 倍——AI 训练吃的是后者。
```

一次 Tensor Core 指令完成一个小矩阵块的乘加（如 m16n8k16），多个 Tensor Core 在一个 SM 内并行，再由上百个 SM 并行——这才是深度学习算力爆发的来源。

这也解释了三件事：

- **精度越低，算力越高**：BF16 989 → FP8 1,979 → FP4（Blackwell）逐级翻倍，因为 Tensor Core 一次能吞吐更多低精度元素；
- **矩阵形状重要**：Tensor Core 要求维度对齐（如 16 的倍数），小矩阵/不规则形状会退化成 CUDA Core 路径；
- **“峰值”是稠密值**：稀疏（2:4）可以到 2 倍，但需要模型和 kernel 配合，不能直接拿来算 MFU。

> 更细的指令形状与流水线见 NVIDIA Hopper / Blackwell 架构白皮书；这里只需记住：**算力表上的 TFLOPS 是 Tensor Core 的矩阵吞吐，不是 CUDA Core 的标量吞吐。**

### 一个端到端算例：8B 模型的一次前向

粗算 Llama 3 8B 在 batch=1、seq=2048 时的一次前向：

```
FLOPs ≈ 2 × N × tokens = 2 × 8×10⁹ × 2,048 ≈ 3.3×10¹³

H100 BF16 稠密 989 TFLOPS、按 40% MFU → 约 4.0×10¹⁴ FLOP/s
→ 前向约 0.08 s

双路 64 核 Xeon、AVX-512 BF16 峰值约 5 TFLOPS、按 50% → 约 2.5×10¹² FLOP/s
→ 前向约 13 s
```

**单次前向差约 150 倍**；训练时反向约为前向的 2 倍，差距同量级。这个算例说明的不是“GPU 什么都快”，而是“矩阵负载下，GPU 的 Tensor Core 路径和 CPU 的向量路径不在一个量级”。

---

## 具体数字对比

| 任务类型 | CPU（单核） | CPU（全核） | GPU（全核） | 结论 |
|---------|-----------|-----------|-----------|--------|
| 大矩阵乘法（GEMM） | 基线 | 多核可摊平 | 快一到几个数量级 | GPU 胜 |
| 标量加法 | 快 | 快 | 有 kernel 启动开销（μs 级） | CPU 胜 |
| 分支预测密集代码 | 快 | 快 | 慢 | CPU 胜 |

> 注意：上表只给方向性结论。具体的加速倍数**没有统一数字**——它取决于矩阵规模、精度、库实现和硬件代际。想拿到你环境里的真实倍数，用下方脚本实测，不要相信"快 N 倍"的笼统说法。

AI 训练的工作负载就是 **大量矩阵运算** + **少量标量操作** + **几乎没有分支**。完全是 GPU 的主场。

---

## 优缺点

| 方面 | CPU | GPU |
|------|-----|-----|
| 单线程性能 | ✅ 极强 | ❌ 弱 |
| 并行计算 | ❌ 弱（几十核） | ✅ 极强（数千核） |
| 灵活性 | ✅ 什么都能做 | ❌ 适合特定模式 |
| 功耗效率（FLOPS/W） | ❌ 矩阵负载下低 | ✅ 矩阵负载下显著更高（倍数随负载/代际变化，无统一值） |
| 编程难度 | ✅ 标准化 | ⚠️ 需要 CUDA/OpenCL |

---

## 🛠 实操指南

### 📋 决策指南：什么时候用 CPU，什么时候用 GPU？

| 工作负载类型 | 用 CPU | 用 GPU | 为什么 |
|------------|--------|--------|-------|
| 大模型训练（7B+） | ❌ | ✅ | 纯矩阵运算，GPU 占绝对优势（倍数实测为准） |
| 小模型推理（<1B） | ⚠️ 可用 | ✅ 更优 | CPU 延迟低但吞吐不足 |
| 数据预处理 | ✅ | ❌ | 逻辑判断多，GPU 不擅长 |
| 模型 serving 路由 | ✅ | ❌ | 少量请求，延迟敏感 |
| 批处理推理（高吞吐） | ❌ | ✅ | GPU 并行优势明显 |
| HPC 科学计算（FP64） | ⚠️ 视场景 | ⚠️ 视场景 | AMD MI300X FP64 强，NVIDIA 相对弱 |
| 训练 405B Llama 3 | ❌ | ✅ 必须 GPU | 仅 BF16 权重就约 810 GB 显存 |

**一条经验：如果你的代码里 `for` 循环多、分支多、逻辑判断多 → 用 CPU。如果你的代码里矩阵乘法多 → 用 GPU。**

### 🔧 自己动手验：你的任务 CPU 和 GPU 差多少？

```python
# 用 PyTorch 简单对比 CPU vs GPU 的矩阵运算速度
import torch
import time

size = 5000
# 随机生成两个大矩阵
A = torch.randn(size, size)
B = torch.randn(size, size)

# CPU 上跑
t0 = time.time()
C = torch.mm(A, B)
cpu_time = time.time() - t0
print(f"CPU 矩阵乘法: {cpu_time*1000:.1f} ms")

# GPU 上跑（如果有）
if torch.cuda.is_available():
    A_gpu = A.cuda()
    B_gpu = B.cuda()
    torch.cuda.synchronize()
    t0 = time.time()
    C_gpu = torch.mm(A_gpu, B_gpu)
    torch.cuda.synchronize()
    gpu_time = time.time() - t0
    print(f"GPU 矩阵乘法: {gpu_time*1000:.1f} ms")
    print(f"GPU 加速比: {cpu_time/gpu_time:.0f}×")

    # 还可以测更大的矩阵
    size2 = 20000
    A2 = torch.randn(size2, size2).cuda()
    B2 = torch.randn(size2, size2).cuda()
    torch.cuda.synchronize()
    t0 = time.time()
    C2 = torch.mm(A2, B2)
    torch.cuda.synchronize()
    print(f"GPU 20K×20K: {(time.time()-t0)*1000:.1f} ms")
```

```bash
# 命令行查看 GPU 算力状态
nvidia-smi --query-gpu=name,compute_cap,memory.total,temperature.gpu \
  --format=csv
# 输出示例：NVIDIA H100, 9.0, 81259 MiB, 35
# Compute Capability 7.0+（Volta）= 支持混合精度训练（Tensor Core）
# Compute Capability 8.0 = A100（Ampere），只到 BF16/TF32，不支持 FP8
# Compute Capability 8.9（Ada）/ 9.0（Hopper）起 = 原生支持 FP8
```

### ⚠️ 常见误区

- **误区**："GPU 核心多，所以什么都比 CPU 快"
  - **实际**：GPU 的核心是简单 ALU，适合大规模并行计算。但**处理单个任务时，CPU 快得多**——GPU 上哪怕一次小计算也有微秒级的 kernel 启动开销，CPU 只要纳秒级。**GPU 的强项不是快，而是多——不是跑得快，而是能同时跑很多。**

- **误区**："买 GPU 就是买算力，TFLOPS 越高越好"
  - **实际**：TFLOPS 是峰值算力（引用时务必区分稠密/稀疏——稀疏是稠密的 2 倍），实际能用到多少取决于**内存带宽**和**数据流水线**。H100 的稠密 FP16 峰值约 989 TFLOPS，但实际训练 LLM 时 MFU（模型算力利用率）通常在 30-50%。瓶颈往往在**显存带宽**（H100 3.35 TB/s）而不是算力本身。这就是为什么 HBM 带宽比算力更关键。

- **误区**："AMD MI300X 的 FP16 峰值算力比 H100 高，所以训练更快"
  - **实际**：微基准测试可以接近或超过 H100，但**端到端训练性能不等于微基准**——取决于模型、框架、算子覆盖和互联。MI300X 与 H100 的真实训练差距常见经验区间约 20-40%（不同模型/框架差异很大，**需用自有负载实测**，不构成精确承诺），主要差在 ROCm 软件栈和集群互联。**硬件峰值只是纸面数字，软件生态和工程实现同样重要。**

### 📊 底线数字速查

| 指标 | CPU（第5代 Intel Xeon 典型值） | GPU（H100 SXM） | GPU（B200） |
|------|----------------------|----------------|------------|
| 典型核心数 | 32-64 核 | 16,896 CUDA 核心（PCIe 版 14,592） | 未公开（以 Tensor Core 为准） |
| FP16/BF16 算力 | ~5 TFLOPS（AVX，理论值） | **989 TFLOPS 稠密**（稀疏 1,979） | **2.25 PFLOPS 稠密**（稀疏 4.5） |
| FP8 / FP4 | — | FP8 稠密 1,979 TFLOPS（稀疏 3,958） | FP8 稠密 4.5 PFLOPS；FP4 稠密 9 PFLOPS（稀疏 18） |
| FP64 算力 | ~2.5 TFLOPS | 34 TFLOPS | 低（Blackwell 不侧重 FP64） |
| 内存带宽 | ~0.5 TB/s（DDR5，8 通道） | **3.35 TB/s（HBM3）** | **8 TB/s（HBM3e）** |
| 功耗（TDP） | ~350 W（CPU 本身） | 700 W | ~1,000 W |
| 适合场景 | 逻辑控制、数据预处理 | **大规模矩阵运算** | **大规模训练 + FP4/FP8 推理** |

### 💡 一句话记住

> **"CPU 是米其林大厨——什么都会做，但一次只能做一道菜。GPU 是一万名小学生——每个人只会做加减法，但一秒钟能算一亿道。AI 训练全是加减法矩阵运算——你猜谁更快？"**

---

## 对后续的铺垫

理解了 CPU 和 GPU 的分工后，下一个问题就自然浮现了：

**GPU 的计算速度极快，那数据和指令怎么才能及时"喂"到它嘴里？**

这就引出了 GPU 内部的数据通道问题（PCIe、NVLink、HBM）。但我们先把显存的层次搞清楚——

→ [[02-PCIe瓶颈与NVLink诞生]]

---

> **参考资料**：
> - [NVIDIA H100 官方规格页](https://www.nvidia.com/en-us/data-center/h100/) — CUDA Core 数、TFLOPS（稠密/稀疏）、HBM 带宽
> - [NVIDIA Hopper 架构深度解析（NVIDIA Developer Blog）](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
> - [NVIDIA Hopper Architecture Technical Brief（Tensor Core 指令形状）](https://resources.nvidia.com/en-us-tensor-core/gtc22-whitepaper-hopper)
> - [NVIDIA Blackwell Architecture Technical Brief（FP4/FP8 Tensor Core）](https://resources.nvidia.com/en-us-blackwell-architecture)
> - [NVIDIA CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) — Compute Capability 与各代硬件特性对照
> - [NVIDIA DGX B200 页面（Blackwell）](https://www.nvidia.com/en-us/data-center/dgx-b200/)
