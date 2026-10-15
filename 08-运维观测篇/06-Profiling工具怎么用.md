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
| **Nsight Systems** | 系统级 | 时间线视图：CPU、GPU、NCCL 活动 | 宏观定位瓶颈 |
| **Nsight Compute** | 内核级 | 单个 CUDA kernel 的吞吐、带宽、指令 | 微观优化计算 |
| **PyTorch Profiler** | 框架级 | PyTorch 操作的耗时、内存分配 | 快速定位框架问题 |

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
│ 1. GPU 空闲段 → 在等 NCCL？在等 DataLoader？          │
│ 2. NCCL 阶段过长 → 通信是瓶颈                          │
│ 3. Small kernel 过多 → kernel launch 开销             │
└─────────────────────────────────────────────────────┘
```

### 实际案例：Nsight Systems 发现的问题

```
问题：70B 模型训练 MFU 只有 32%，远低于预期的 45%

Nsight Systems 时间线：
  GPU 活动：              ████████░░░░░░████████░░░░░░
  NCCL 通信：             ░░░░░░░░████░░░░░░░░░░████░░
  DataLoader：           ░░░░░░░░░░░░░░████████░░░░░░

发现 1：NCCL 通信时间占总时间的 35%
  → 通信是瓶颈

发现 2：DataLoader 在 GPU 计算期间不是"完全异步"的
  → 有时 GPU 在等数据

修复：
  1. 增大 batch size → 计算/通信比从 1.2:1 提升到 2.5:1
  2. 增加 DataLoader num_workers → GPU 等待数据时间降到 2%
  3. 开启 NCCL 通信与计算重叠 → 通信时间隐藏 40%
  
结果：MFU 从 32% 提升到 44%
```

---

## 🛠 实操指南

### 📋 决策指南：什么时候用什么工具？

```
你的问题是什么？
├─ "训练整体慢，不知道瓶颈在哪" →
│   └─ Nsight Systems（系统级视图）
│
├─ "有个 kernel 特别慢，想优化它" →
│   └─ Nsight Compute（内核级视图）
│
├─ "训练代码改了以后变慢了，想对比" →
│   └─ PyTorch Profiler（快速对比）
│
└─ "训练过程中 OOM 了" →
    └─ Nsight Systems + 内存跟踪
```

### 🔧 自己动手验：用 PyTorch Profiler 快速诊断

```python
# 最简单的 Profiling——PyTorch Profiler
import torch
from torch.profiler import profile, record_function, ProfilerActivity, schedule

# PyTorch Profiler 的底层是 Kineto：它同时挂钩 CPU、CUDA runtime（含 cuDNN/NCCL）
# 与 GPU 计数器，所以能在一张表里同时看到 CPU 时间、设备(GPU)时间与 NCCL 通信。
#
# 注意：不要用 with profile(...) 直接包住 100 个 step——训练早期的
# warmup、显存分配、kernel 编译都会严重失真。生产环境用 schedule(wait/warmup/active/repeat)，
# 只采"已经稳定"的那几个 step。
sched = schedule(wait=5, warmup=3, active=5, repeat=2)

with profile(
    activities=[
        ProfilerActivity.CPU,
        ProfilerActivity.CUDA,
    ],
    schedule=sched,
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./log/prof"),  # 导出 timeline 供 TensorBoard 查看
    record_shapes=True,      # 记录 tensor shape
    with_stack=True,          # 记录 Python 调用栈
    profile_memory=True,      # 记录内存分配
) as prof:
    for step in range(40):
        with record_function("step_forward"):
            outputs = model(inputs)
        with record_function("step_backward"):
            loss.backward()
        with record_function("step_optimizer"):
            optimizer.step()
            optimizer.zero_grad()
        prof.step()               # 每步推进一次 schedule

# 1) 表格：快速看谁最耗时
print(prof.key_averages().table(
    sort_by="device_time_total",  # 按设备(GPU)时间排序（PyTorch 2.0 起；原 cuda_time_total 已废弃）
    row_limit=20                  # 只看 Top-20
))

# 2) 时间线：真正的"X 光片"——用 TensorBoard 打开 ./log/prof
#    tensorboard --logdir ./log/prof
#    也可以导出 chrome://tracing 能打开的 JSON：
prof.export_chrome_trace("trace.json")

# 输出：
# ----------------------  -----------  -----------  -----------
# Name                  CPU total    Device total  Calls
# ----------------------  -----------  -----------  -----------
# aten::matmul          12.4s         45.2s         1280
# flash_attention       8.2s          28.1s         256
# nccl:all_reduce       2.1s          15.3s         64
# ...                   
# ----------------------  -----------  -----------  -----------
# 结论：matmul 占了最多设备时间 → 可能计算密集型的层是瓶颈
#       all_reduce 占 15% → 通信开销中等
```

### Nsight Systems 使用示例

```bash
# 1. 安装 Nsight Systems
# 从 NVIDIA 官网下载安装 https://developer.nvidia.com/nsight-systems

# 2. 快速 Profiling（默认配置）
nsys profile -o my_profile -t cuda,nvtx python train.py

# 3. 多 GPU Profiling（需要在每个 rank 上跑）
# 使用 MPI
mpirun -np 8 nsys profile -o profile_%q{OMPI_COMM_WORLD_RANK} \
  -t cuda,nvtx,nccl python train.py

# 4. 最小化 profiling 开销（只抓关键部分）
nsys profile \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --export=sqlite \
  -o minimal_profile \
  python -c "
import torch
# 在你想 profiling 的代码段加开关
torch.cuda.cudart().cudaProfilerStart()
# ... 你的训练代码 ...
torch.cuda.cudart().cudaProfilerStop()
"

# 5. 分析输出
# 在 Nsight Systems GUI 中打开 .nsys-rep 文件
# 或者用命令行导出
nsys stats my_profile.nsys-rep
```

### NVTX 埋点：让时间线"看得懂"

上面时间线里的 `[AllReduce]`、`step_forward` 这类彩色区块，本质是 **NVTX 区间**。两种常见写法：

```python
# 1) PyTorch 内置的 record_function（就是上面示例里的用法，等价于一次 NVTX range）
with record_function("step_forward"):
    ...

# 2) 手动 NVTX（可包裹任意代码，包括自写的通信 / 数据加载逻辑）
import torch
torch.cuda.nvtx.range_push("AllReduce")
dist.all_reduce(grad)
torch.cuda.nvtx.range_pop()
```

配合 `nsys profile -t cuda,nvtx,nccl` 采集后，这些区间会直接变成时间线上的命名区块——这才把"能看"变成了"看得懂 NCCL 到底卡在哪一步"。

### Nsight Compute 使用示例

```bash
# 1. 分析单个 CUDA kernel（正则匹配要加 regex: 前缀，否则会被当成字面 kernel 名）
ncu --kernel-name regex:matmul python train.py

# 2. 采集 kernel 的完整指标
#    注意：--import-source 在新版 Nsight Compute 中已弃用/移除（被 --source-folders 系列取代），
#    而且对 python -c 这种一次性程序抓源码也没有意义，这里只用 --set full -o
ncu --set full -o my_kernel_report python \
  -c "import torch; \
      x=torch.randn(4096,4096,device='cuda'); \
      w=torch.randn(4096,4096,device='cuda'); \
      for _ in range(100): torch.mm(x,w); torch.cuda.synchronize()"

# 3. 判断 kernel 是 compute / memory / latency bound
#    正确做法：用 --set full 采集后，看报告里的 "Speed of Light" / Roofline 小节。
#    注意 --query-metrics 只是"列出可用指标名后退出"的子命令，它不会分析 kernel，
#    也不会输出瓶颈分类——拿它 grep "Memory|Compute|Latency" 必然是空的。
```

### ⚠️ 常见误区

- **误区**："Profiling 是锦上添花，不是必须的"
  - **实际**：**Profiling 是性能优化的前提**。没有数据就没有方向。很多团队凭感觉优化，改了一堆东西吞吐没变——Profiling 一次就发现瓶颈在 NCCL 上，半小时搞定。

- **误区**："Nsight Systems 开销太大，不能在训练中开"
  - **实际**：nsys 对 CUDA 的采集本质是**跟踪（tracing）**，并不存在官方的"采样/跟踪"二分法。默认配置开销不高，但**开销与 trace 的事件量和 buffer 大小有关**——长时间开启会让 buffer 膨胀、主机侧开销上升。正确做法是用 `--duration` / `--capture-range` 限定采集窗口，只在需要的区间抓。

- **误区**："PyTorch Profiler 和 Nsight Systems 是替代关系"
  - **实际**：**它们是互补的。** PyTorch Profiler 给你框架层面的快照，Nsight Systems 给你系统级的时间线。先跑 Nsight Systems 找到瓶颈在哪一层（计算/通信/数据），再针对性深入，最后用 PyTorch Profiler 验证优化效果。

### 📊 底线数字速查

**Profiling 配置速查**：

| 工具 | 用途 | 开销 | 推荐时长 |
|------|------|------|---------|
| PyTorch Profiler | 快速排查 | 与 trace 事件量有关 | 用 schedule，采 50-200 step |
| Nsight Systems | 宏观瓶颈定位 | 与 trace 事件量、buffer 有关 | 用 `--capture-range` 限流 |
| Nsight Compute | 内核级优化 | 高（逐 kernel 重放）| 单个 / 少量 kernel |

### 💡 一句话记住

> **"Profiling 就是训练的 X 光机——没有它，你只是在猜哪里慢。先跑 Nsight Systems 找到瓶颈在哪层（计算/通信/数据），再用 Nsight Compute 优化具体 kernel，最后用 PyTorch Profiler 验证优化效果。三步走，永远不要让团队'凭感觉'调优。"**

---

## 对后续的铺垫

到这你已经把 AI 集群"跑起来之后"的运维观测主线走了一遍——监控、利用率、NCCL 调优、慢节点、Checkpoint 恢复与 Profiling。

最后一章，我们看看最前沿的实践：

- Meta/Google/Microsoft 的超大集群长什么样？
- 液冷会成为标配吗？
- 全光互联和 CXL 内存池化意味着什么？
- 自己动手搭小集群怎么入门？

→ 跳到 [09-前沿与实践篇/01-超大集群长什么样](../09-前沿与实践篇/01-超大集群长什么样.md)

---

> **参考资料**：
> - [Nsight Systems 用户指南](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) (NVIDIA, 2026)
> - [Nsight Compute CLI 文档](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html) (NVIDIA, 2026)
> - [PyTorch Profiler（torch.profiler）文档](https://docs.pytorch.org/docs/stable/profiler.html) (PyTorch, 2026)
> - [PyTorch Profiler 教程](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html) (2026)
