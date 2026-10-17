# Q: Profiling 工具怎么用？

## 为什么会有这个问题

你的训练速度比预期慢了 30%。你检查了 GPU 利用率——98%。你检查了 NCCL——配置正确。你检查了 DataLoader——没有瓶颈。

**但训练就是慢。**

这时候你要的不是"猜"哪里慢了——你需要一个 X 光机，把 GPU 上每毫秒在干什么都拍下来。这就是 Profiling 工具。

## 解决什么问题

Profiling 工具的核心功能：**把 GPU 的时间线可视化，告诉你每个 kernel、每次通信、每步计算花了多少时间。**

### 你需要的三种 Profiling 工具

| 工具 | 层级 | 能告诉你 | 适合场景 |
|------|------|---------|---------|
| **Nsight Systems** | 系统级 | 时间线视图：CPU、GPU、NCCL 活动、空窗 | 宏观定位瓶颈（第一站） |
| **Nsight Compute** | 内核级 | 单个 CUDA kernel 的吞吐、带宽、指令 | 微观优化计算（第二站） |
| **PyTorch Profiler** | 框架级 | PyTorch 算子的耗时、内存分配、调用栈 | 算子归因（需要把时间对应回代码行时） |

### 统一的工作流：先时间线，再单 kernel，最后算子归因

```
第一步：Nsight Systems —— 回答"时间花在哪一层"
  看整条时间线：GPU 空窗、NCCL 段、kernel 密度
  输出：瓶颈是通信 / 数据 / kernel launch / 计算本身

第二步：Nsight Compute —— 回答"这个 kernel 为什么慢"
  只针对第一步挑出的慢 kernel 做深度分析（不是全程序！）
  输出：compute-bound 还是 memory-bound，瓶颈在哪条流水线

第三步：PyTorch Profiler —— 回答"这段时间对应哪行代码"
  算子级耗时表 + 调用栈，把 GPU 时间归因到模型代码
  输出：哪个 nn.Module / 算子最值得优化

为什么是这个顺序：Nsight Systems 便宜、全局、告诉你去哪看；
Nsight Compute 昂贵、精确、但视野只有单个 kernel；
PyTorch Profiler 牺牲底层细节换框架级归因。
顺序反了，你会花大量时间分析一个根本不重要的 kernel。
```

### Nsight Systems 的时间线能告诉你什么

```
┌─────────────────────────────────────────────────────┐
│            Nsight Systems 时间线                       │
├─────────────────────────────────────────────────────┤
│ CPU: [dataloader][forward][backward][optimizer]     │
│ GPU0: [matmul][flash_attn][matmul][  gemm   ]       │
│ GPU1: [matmul][flash_attn][matmul][  gemm   ]       │
│ NCCL:                       [AllReduce(梯度同步)]      │
│                                                      │
│ 关键问题：                                             │
│ 1. GPU 空窗段 → 在等 NCCL？在等 DataLoader？          │
│ 2. NCCL 段过长 → 通信是瓶颈                            │
│ 3. Small kernel 过多 → kernel launch 开销             │
└─────────────────────────────────────────────────────┘
```

### 实际案例：Nsight Systems 发现的问题

```
问题：70B 模型训练 MFU 只有 32%，远低于同类的 40%+

Nsight Systems 时间线：
  GPU 计算：              ████████░░░░░░████████░░░░░░
  NCCL 通信：             ░░░░░░░░████░░░░░░░░░░████░░
  DataLoader：           ░░░░░░░░░░░░░░░███████░░░░░░

发现 1：NCCL 通信时间占总时间的 35%
  → 通信是瓶颈

发现 2：DataLoader 在 GPU 计算期间不是"完全异步"的
  → 有时 GPU 在等数据

修复：
  1. 增大 batch size → 计算/通信比从 1.2:1 提升到 2.5:1
     （All-Reduce 通信量由参数量决定，与 batch 无关；
      batch 变大 = 计算变多、通信不变，比值自然上升）
  2. 增加 DataLoader num_workers → GPU 等待数据时间降到 2%
  3. 开启 NCCL 通信与计算重叠 → 通信时间被部分隐藏

结果：MFU 从 32% 提升到 44%
```

---

## 🛠 实操指南

### 📋 决策指南：什么时候用什么工具？

```
你的问题是什么？
├─ "训练整体慢，不知道瓶颈在哪" →
│   └─ Nsight Systems（系统级时间线，第一站）
│
├─ "时间线里有个 kernel 特别慢，想优化它" →
│   └─ Nsight Compute（内核级深度分析，第二站）
│
├─ "训练代码改了以后变慢了，想对比/归因到算子" →
│   └─ PyTorch Profiler（算子表 + 调用栈）
│
└─ "训练过程中 OOM 了" →
    └─ 显存工具，不是 nsys：
       torch.cuda.memory_summary() 看当前分配
       torch.cuda.memory._record_memory_history() + 
       torch.cuda.memory._dump_snapshot() 生成显存快照，
       用 pytorch.org/memory_viz 可视化谁占了显存
```

### 🔧 第一步：Nsight Systems 看时间线

```bash
# 1. 安装 Nsight Systems
# 从 NVIDIA 官网下载安装 https://developer.nvidia.com/nsight-systems

# 2. 快速 Profiling（CUDA + NVTX + NCCL + OS 运行时）
nsys profile -o my_profile -t cuda,nvtx,nccl,osrt python train.py
#   -t 指定追踪的 API 类别：
#     cuda  = CUDA API + kernel + memcpy（跟踪，非采样）
#     nvtx  = NVTX 标注（强烈建议在训练代码里加 range 标注）
#     nccl  = NCCL 通信 API
#     osrt  = OS 运行时线程（配合查 CPU 侧问题）

# 3. 多 GPU/多机 Profiling（每个 rank 一个报告）
torchrun --nproc_per_node=8 \
  -m nsys profile -o profile_%q{RANK} -t cuda,nvtx,nccl train.py

# 4. 缩小范围：只抓关键段（用 NVTX/cudaProfilerApi 圈范围）
nsys profile \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  -o minimal_profile \
  python -c "
import torch
# 在你想 profiling 的代码段加开关
torch.cuda.cudart().cudaProfilerStart()
# ... 你的训练代码 ...
torch.cuda.cudart().cudaProfilerStop()
"

# 5. 分析输出
# 在 Nsight Systems GUI 中打开 .nsys-rep 文件，或者用命令行：
nsys stats my_profile.nsys-rep
```

> **关于开销**：CUDA 活动是**跟踪**（每笔 API/kernel 都记录），有开销，且开销与 kernel 数量成正比——所以生产建议是：**先缩小捕获范围（`--capture-range`）、用 NVTX 标注只分析代表性 step**，而不是"全程挂着跑三天"。不要相信"<3% 开销"这类万能数字，你的负载要自己量。

### 🔧 第二步：Nsight Compute 看单 kernel

```bash
# 1. 只分析目标 kernel（别全程序跑！ncu 会串行化重放 kernel）
ncu --kernel-name regex:flash_attn -o kernel_report python train.py

# 2. 标准做法：SpeedOfLight section 一眼看出 compute/memory bound
ncu --section SpeedOfLight --kernel-name regex:flash_attn python train.py
# 输出里最关键的两行：
#   SM %（计算吞吐占比） vs Memory %（显存带宽占比）
#   双低 → latency bound；一个顶格 → 对应 bound

# 3. 全量指标（开销大，只对单个 kernel 用）
ncu --set full -o full_report python train.py

# 4. 常用控制
ncu --replay-mode kernel python train.py   # kernel 重放（默认）；应用级问题换 application 模式
ncu --launch-count 5 python train.py       # 每个 kernel 只分析前 5 次启动
```

三个必知的"坑"：

- **`--query-metrics` 不运行你的程序**：`ncu --query-metrics python train.py` 只是列出硬件支持的指标然后退出。要分析就用 `--set full` 或 `--section SpeedOfLight`。
- **权限**：GPU 性能计数器默认受限（多用户/容器环境常见 `ERR_NVGPUCTRPERM`）。需要驱动侧放开计数器访问（容器里加 `CAP_SYS_ADMIN` / 相应 NVIDIA 驱动能力配置），并确认没有其他进程同时在用计数器——**同一时刻一个 GPU 只允许一个 profiler 独占计数器**。
- **重放**：ncu 为了收集全部指标会**重放 kernel 多次**，被测 kernel 的执行时间和真实运行完全不同——用它看瓶颈类型，别用它推测真实耗时。

### 🔧 第三步：PyTorch Profiler 做算子归因

```python
# 框架级 Profiling——把 GPU 时间归因到代码
import torch
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[
        ProfilerActivity.CPU,
        ProfilerActivity.CUDA,
    ],
    record_shapes=True,      # 记录 tensor shape
    with_stack=True,          # 记录 Python 调用栈
    profile_memory=True,      # 记录内存分配
) as prof:
    for step in range(100):
        with record_function("step_forward"):
            outputs = model(inputs)
        with record_function("step_backward"):
            loss.backward()
        with record_function("step_optimizer"):
            optimizer.step()
            optimizer.zero_grad()

# 输出最重要的统计
print(prof.key_averages().table(
    sort_by="cuda_time_total",  # 按 CUDA 时间排序
    row_limit=20                 # 只看 Top-20
))

# 输出：
# ----------------------  -----------  -----------  -----------
# Name                  CPU total    CUDA total    Calls
# ----------------------  -----------  -----------  -----------
# aten::matmul          12.4s         45.2s         1280
# flash_attention       8.2s          28.1s         256
# nccl:all_reduce       2.1s          15.3s         64
# ...
# ----------------------  -----------  -----------  -----------
# 结论：把 Nsight Systems 发现的"慢"对应到具体算子，
#       决定优化哪一层；prof.export_chrome_trace() 可导出给
#       Perfetto/浏览器时间线查看
```

### ⚠️ 常见误区

- **误区**："Profiling 是锦上添花，不是必须的"
  - **实际**：**Profiling 是性能优化的前提**。没有数据就没有方向。很多团队凭感觉优化，改了一堆东西吞吐没变——Profiling 一次就发现瓶颈在 NCCL 上，半小时搞定。

- **误区**："Nsight Systems 开销很小，可以全程开着"
  - **实际**：CUDA 是跟踪模式，开销和 kernel 密度成正比，没有统一的"<3%"承诺。正确姿势：`--capture-range` 圈定代表性 step + NVTX 标注，短窗口拿高质量数据。

- **误区**："先跑 PyTorch Profiler，有需要再上 Nsight"
  - **实际**：**反过来才是对的工作流**——先 Nsight Systems 看时间线确定瓶颈在哪一层，再 Nsight Compute 深挖慢 kernel，需要把时间归因到代码行时才用 PyTorch Profiler。先看算子表容易把精力花在"总耗时大但不是瓶颈"的算子上。

- **误区**："OOM 了就用 Nsight Systems 看看"
  - **实际**：nsys 的时间线不是显存分析工具。**OOM 用 `torch.cuda.memory_summary()` 和显存快照（`memory._record_memory_history` + memory_viz）**，它们能告诉你每一块显存是谁分配的、峰值时刻的分配栈。

### 📊 底线数字速查

**Profiling 工具速查**：

| 工具 | 用途 | 数据方式 | 推荐时长 |
|------|------|---------|---------|
| Nsight Systems | 宏观瓶颈定位 | 跟踪（开销随 kernel 密度增长） | 几十~几百个代表性 step |
| Nsight Compute | 单 kernel 深度分析 | 计数器重放（高开销） | 目标 kernel，launch-count 限次 |
| PyTorch Profiler | 算子归因 | 框架级跟踪 | 50-200 step |
| memory_summary / 快照 | 显存问题 | 即时 / 带历史记录 | OOM 排查专用 |

**顺序口诀**：

| 步骤 | 工具 | 回答的问题 |
|------|------|-----------|
| 1 | Nsight Systems | 时间花在哪一层？（通信/数据/计算） |
| 2 | Nsight Compute | 这个 kernel 为什么慢？（compute/memory/latency bound） |
| 3 | PyTorch Profiler | 这段时间对应哪行代码？（算子归因） |

### 💡 一句话记住

> **"Profiling 就是训练的 X 光机——没有它，你只是在猜哪里慢。固定工作流：Nsight Systems 看时间线定层 → Nsight Compute 深挖慢 kernel（记得限 kernel、开权限、别信重放后的耗时）→ PyTorch Profiler 归因到代码。OOM 不归 nsys 管，归显存快照管。"**

---

## 对后续的铺垫

到这你已经掌握了 AI 集群从芯片到网络的全部基础知识。

最后一章，我们看看最前沿的实践：

- Meta/Google/Microsoft 的超大集群长什么样？
- 液冷会成为标配吗？
- 全光互联和 CXL 内存池化意味着什么？
- 自己动手搭小集群怎么入门？

→ 跳到 [09-前沿与实践篇/01-超大集群长什么样](../09-前沿与实践篇/01-超大集群长什么样.md)

---

> **参考资料**：
> - [Nsight Systems 用户指南](https://developer.nvidia.com/nsight-systems) — NVIDIA
> - [Nsight Compute 用户指南](https://developer.nvidia.com/nsight-compute) — NVIDIA
> - [PyTorch Profiler 教程](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) — PyTorch
> - [PyTorch CUDA 显存管理与快照工具](https://docs.pytorch.org/docs/stable/torch_cuda_memory.html) — PyTorch
