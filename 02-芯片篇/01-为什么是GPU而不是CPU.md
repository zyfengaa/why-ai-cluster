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

- **大量 ALU**：一个 GPU 有数千个计算核心（H100 有 18,432 个 CUDA Cores）
- **简单控制单元**：不做复杂的指令调度，一次给所有 ALU 发相同指令
- **小缓存**：靠高并行度隐藏访存延迟

**GPU 适合**：大规模并行计算、矩阵运算、SIMD（单指令多数据）。就像一个万名小学生组成的方阵——每个人只能算简单加减法，但一秒钟能算出一亿道。

---

## 具体数字对比

| 任务类型 | CPU（单核） | CPU（全核） | GPU（全核） | 加速比 |
|---------|-----------|-----------|-----------|--------|
| 10² × 10³ 矩阵乘法 | 0.5 ms | 8 ms（16核） | 0.02 ms | **400×** |
| 标量加法 | 0.3 ns | 0.3 ns | 5 ns | 慢16× |
| 分支预测密集代码 | 快 | 快 | 慢 | CPU 胜 |

AI 训练的工作负载就是 **大量矩阵运算** + **少量标量操作** + **几乎没有分支**。完全是 GPU 的主场。

---

## 优缺点

| 方面 | CPU | GPU |
|------|-----|-----|
| 单线程性能 | ✅ 极强 | ❌ 弱 |
| 并行计算 | ❌ 弱（几十核） | ✅ 极强（数千核） |
| 灵活性 | ✅ 什么都能做 | ❌ 适合特定模式 |
| 功耗效率（FLOPS/W） | ❌ 低 | ✅ 高 5-10 倍 |
| 编程难度 | ✅ 标准化 | ⚠️ 需要 CUDA/OpenCL |

---

## 🛠 实操指南

### 📋 决策指南：什么时候用 CPU，什么时候用 GPU？

| 工作负载类型 | 用 CPU | 用 GPU | 为什么 |
|------------|--------|--------|-------|
| 大模型训练（7B+） | ❌ | ✅ | 纯矩阵运算，GPU 快 100-400× |
| 小模型推理（<1B） | ⚠️ 可用 | ✅ 更优 | CPU 延迟低但吞吐不足 |
| 数据预处理 | ✅ | ❌ | 逻辑判断多，GPU 不擅长 |
| 模型 serving 路由 | ✅ | ❌ | 少量请求，延迟敏感 |
| 批处理推理（高吞吐） | ❌ | ✅ | GPU 并行优势明显 |
| HPC 科学计算（FP64） | ⚠️ 视场景 | ⚠️ 视场景 | AMD MI300X FP64 强，NVIDIA 弱 |
| 训练 405B Llama 3 | ❌ | ✅ 必须 GPU | 单卡 ~810 GB 显存需求 |

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
# Compute Capability 8.0+ = 支持 FP8 训练
# Compute Capability 7.0+ = 支持混合精度训练
```

### ⚠️ 常见误区

- **误区**："GPU 核心多，所以什么都比 CPU 快"
  - **实际**：GPU 的核心是简单 ALU，适合大规模并行计算。但**处理单个任务时，CPU 比 GPU 快得多**。比如你的程序只有一个 `if-else` 判断，CPU 可能 1 ns 完成，GPU 需要 5-10 μs（慢 5,000-10,000 倍）。**GPU 的强项不是快，而是多——不是跑得快，而是能同时跑很多。**

- **误区**："买 GPU 就是买算力，TFLOPS 越高越好"
  - **实际**：TFLOPS 是峰值算力，实际能用到多少取决于**内存带宽**和**数据流水线**。H100 的理论 FP16 算力 2000 TFLOPS，但实际训练 LLM 时 MFU（模型算力利用率）通常在 30-50%。瓶颈往往在**显存带宽**（H100 3.35 TB/s）而不是算力本身。这就是为什么 HBM 带宽比算力更关键。

- **误区**："AMD MI300X 的 FP16 TFLOPS 比 H100 高 32%，所以训练更快"
  - **实际**：2025-2026 年实测数据：MI300X 在实际 AI 训练中只能达到峰值的约 45%，而 H100 能到 93%。8×GPU 场景下 H100 的吞吐比 MI300X 高 46%。**硬件峰值只是纸面数字，软件生态和实际工程实现才是决定因素。**

### 📊 底线数字速查

| 指标 | CPU（Intel Xeon 第5代） | GPU（H100 SXM） | GPU（B200） |
|------|----------------------|----------------|------------|
| 典型核心数 | 32-64 核 | 18,432 CUDA 核心 | 未公开 |
| FP16 算力 | ~5 TFLOPS（AVX） | **~2,000 TFLOPS** | **~9,000 TFLOPS** |
| FP64 算力 | ~2.5 TFLOPS | 34 TFLOPS | 低（Blackwell 不侧重 FP64）|
| 内存带宽 | ~0.5 TB/s（DDR5） | **3.35 TB/s（HBM3）** | **8 TB/s（HBM3e）** |
| 矩阵乘法加速比（vs 单 CPU 核） | 1× | **~400×** | **~1,000+×** |
| 功耗 | 350W（CPU 本身） | 700W | 1,000W |
| 适合场景 | 逻辑控制、数据预处理 | **大规模矩阵运算** | **大规模训练 + FP4 推理** |

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
> - NVIDIA CUDA C++ Programming Guide (docs.nvidia.com)
> - Hennessy & Patterson, "Computer Architecture: A Quantitative Approach" (第6版)
> - GPU 与 CPU 架构对比（NVIDIA Developer Blog）
