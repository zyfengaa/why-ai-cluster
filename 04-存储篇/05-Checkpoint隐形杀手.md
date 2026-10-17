# Q: Checkpoint 为什么是存储的"隐形杀手"？

## 为什么会有这个问题

训练一定会中断——硬件故障、网络抖动、代码 bug、驱动崩溃。Meta 的 Llama 3 论文记录得非常直白：在 16,384 块 GPU 上训练 54 天，发生了 **419 次意外中断**，约 78% 是硬件原因（数据来自论文的故障恢复章节）。

中断不可怕，可怕的是**回滚**：训练状态多久保存一次、保存得多快，直接决定了每次中断浪费多少 GPU 时间。而保存这个动作本身又要打断训练、占用存储带宽——两头都是存储的活。

它"隐形"有三个原因：

1. **单次代价小**：一次 checkpoint 只停几十秒，单看日志无感；
2. **乘数效应大**：停顿 × 频率 × 卡数，一天下来是天文数字；
3. **抢带宽**：checkpoint 写入经常和 dataloader 读取撞车，让两个瓶颈互相放大。

## 解决什么问题

### 一个 checkpoint 到底存了什么？

很多人以为 checkpoint 就是"存模型权重"。完整训练状态远不止：

| 内容 | 大小（每参数） | 说明 |
|------|--------------|------|
| BF16 权重 | 2 B | 模型本体 |
| BF16 梯度 | 2 B | 反向传播产物 |
| FP32 master 权重 | 4 B | 混合精度训练的"真身" |
| Adam m / v | 4 B + 4 B | 一阶、二阶动量 |
| **合计** | **16 B/参数** | 完整训练状态 |

按 16 B/参数（完整训练状态）计算：70B 模型 = **1.12 TB**。若只存权重 + 优化器（14 B/参数）是 980 GB，仅 BF16 权重（2 B/参数）是 140 GB。除了这些张量，还得保存 **RNG 状态、数据加载位置（epoch/step）、学习率调度器状态**——漏了它们，续训的 loss 曲线就无法精确衔接。

### 为什么大：写入带宽与训练停顿

写入时间 = checkpoint 大小 ÷ 聚合写入带宽。以 70B 完整状态 1.12 TB、每 45 分钟保存一次为例：

| 存储 | 聚合写入带宽 | 写 1.12 TB | 停顿占比（写入时间 ÷ 45 分钟） |
|------|------------|-----------|------------------------------|
| NFS | 1.1 GB/s | 约 17 分钟 | **约 38%** |
| Lustre（32 OST） | 38 GB/s | 约 29 秒 | 约 1.1% |
| WEKA + GDS | 44 GB/s | 约 25 秒 | 约 0.9% |

带宽口径全部为**多节点聚合写入**。NFS 那一行不是夸张：单点 1.1 GB/s 写 1.12 TB 就是十几分钟，等于每 45 分钟停课 17 分钟。

### 同步 vs 异步 checkpoint

- **同步**：停训练 → 写存储 → 继续。简单、状态一致，但训练停顿 = 完整写入时间。
- **异步**：先把状态快照到 CPU 内存或本地 NVMe（GPU 只停几百毫秒做 device→host 拷贝），后台再慢慢写远端存储。代价是额外一份内存/本地盘开销，且快照期间的参数需要双缓冲保护。

生产大集群普遍走异步路线，把训练停顿从"分钟"压到"秒级"。

### DCP 与分片保存

`torch.distributed.checkpoint`（DCP）的核心做法：**每个 rank 只写自己那一份分片**，直接写存储上的最终布局，不需要先 gather 到单点。

32 卡保存 1.12 TB：每卡写入 35 GB。注意关键陷阱——

> **每卡写入量下降 N 倍 ≠ 聚合写入量下降 N 倍。** 32 卡每卡 35 GB，聚合仍然是 1.12 TB。存储后端承受的聚合压力只取决于模型大小 ÷ 带宽，与切多少卡无关。分片保存解决的是"单点写 1.12 TB"的排队问题，不是总量问题。

### 间隔与 MTBF：一笔两边都要算的账

checkpoint 间隔的选择是两个成本之和的优化：

- **保存开销** = 保存耗时 ÷ 间隔（生产经验值 1–5%）；
- **期望损失** = 中断发生时平均回滚 间隔/2 的工作。

用 Llama 3 的频率感受一下：419 次中断 ÷ 54 天 ≈ 每 3.1 小时一次。间隔 45 分钟时，每次中断平均损失约 22 分钟；若间隔放宽到 4 小时，每次中断平均损失 2 小时——乘以 16,384 卡，就是数千 GPU 小时。间隔太密则保存开销吃掉训练时间，太疏则一次故障白跑半天。

### 原子性：临时目录 + rename

直接往目标路径写，进程崩溃时会留下半截文件——恢复时才发现整个 checkpoint 不可用。正确做法：

```python
# 1. 先写临时目录 / 文件
tmp_path = ckpt_path + ".tmp"
save_state_dict(state, tmp_path)
# 2. 强制落盘（fsync），保证数据真的到了持久介质
os.sync()  # 或对每个文件 fd 调 fsync
# 3. 原子 rename，目标路径要么是完整旧版，要么是完整新版
os.replace(tmp_path, ckpt_path)
# 4. 只保留最近 N 个 checkpoint，清理更早的
```

`os.replace`（rename）在同一文件系统内是原子操作，这是"checkpoint 永远可恢复"的底线。

### 恢复时的 reshard

换卡数续训（比如从 256 卡缩到 128 卡）、换并行策略（TP=8 改 TP=4），checkpoint 的分片布局就和当前进程对不上了。DCP 支持读时 reshard：加载时按当前布局重新切分。更进一步，[ByteCheckpoint](https://arxiv.org/abs/2407.20143)（NSDI '25）和 DeepSpeed 的 [Universal Checkpointing](https://arxiv.org/abs/2406.18820) 把"保存时布局无关、加载时任意 reshard"做成了系统能力，训练中途改并行策略不再需要昂贵的离线转换。

## 🛠 实操指南

### 📋 决策指南：checkpoint 间隔怎么定？

```
Q1：你的集群实际 MTBF 是多少？
    从训练日志统计两次意外中断的平均间隔（不是拍脑袋）。
    Llama 3 量级：~3 小时/次；健康集群可能几天/次。

Q2：写一次完整状态要多久（实测，含 fsync）？
    保存开销占比 = 保存耗时 ÷ 间隔，目标 1–5%。

Q3：业务能接受损失多少？
    平均损失 ≈ 间隔/2。推理预算：中断频率 × 间隔/2。

经验起点：MTBF 数小时 → 间隔 30–60 分钟 + 异步保存；
         MTBF 数天 → 间隔 1–4 小时也能接受。
```

### 🔧 自己动手验

```python
import time, torch, os

def timed_save(model, optimizer, path):
    torch.cuda.synchronize()
    t0 = time.time()
    tmp = path + ".tmp"
    state = {"model": model.state_dict(), "optim": optimizer.state_dict(),
             "step": step, "rng": torch.get_rng_state()}   # RNG 也要存
    torch.save(state, tmp)
    os.replace(tmp, path)          # 原子替换
    torch.cuda.synchronize()
    print(f"save {os.path.getsize(path)/1e12:.2f} TB in {time.time()-t0:.1f}s")
    # 把这个时间除以 checkpoint 间隔，占比 > 5% 就该做异步/分片优化了
```

再做一个灾难演练：训练跑 20 分钟后 `kill -9` 主进程，从最近 checkpoint 恢复，确认 loss 曲线与中断前衔接。**没做过 kill -9 演练的 checkpoint 方案，默认不可信。**

### ⚠️ 常见误区

- **误区**："现在每卡只写 35 GB，存储一点压力都没有"
  - **实际**：每卡写入量小了 N 倍 ≠ 聚合写入量小了 N 倍。1.12 TB 还是要从存储系统整体过一遍，聚合带宽才是硬约束。卡数翻倍只是把排队变成并发，总量没变。
- **误区**："checkpoint 存权重就够了"
  - **实际**：漏掉优化器状态/RNG/数据加载位置，续训不是失败就是 loss 跳变。70B 完整状态是 1.12 TB（16 B/参数），不是 140 GB。
- **误区**："写到 NFS 上，反正不赶时间"
  - **实际**：1.1 GB/s 写 1.12 TB ≈ 17 分钟，45 分钟间隔下停顿占比 38%。而且同步写 NFS 的这段时间 GPU 完全空转。
- **误区**："checkpoint 间隔越小越安全"
  - **实际**：保存本身有开销（1–5%），间隔太密是给存储上刑、给训练放血。间隔要按实测 MTBF 算，不是越小越好。

### 📊 底线数字速查

| 指标 | 优秀 | 及格 | 报警 |
|------|------|------|------|
| 保存开销占比（保存耗时 ÷ 间隔） | < 1% | 1–5% | > 5% |
| 70B 完整状态（1.12 TB）保存耗时（聚合） | < 30 秒 | 30 秒–2 分钟 | > 5 分钟 |
| checkpoint 间隔 vs 实测 MTBF | 间隔 ≤ MTBF/4 | 间隔 ≤ MTBF/2 | 间隔 ≈ MTBF |
| 恢复演练 | 每周自动执行 | 手工执行过 | 从未做过 |
| checkpoint 保留策略 | 最近 N 个 + 定期长期档 | 最近 N 个 | 只留 1 个 |

### 💡 一句话记住

> **"Checkpoint 是存储的隐形杀手：它把'训练会断'和'保存要停'两笔账都算在存储头上。用完整状态（16 B/参数）算容量，用聚合带宽算停顿，用 MTBF 算间隔——三笔账都对了，它就从杀手变成了保险。"**

---

## 对后续的铺垫

Checkpoint 解决的是**写路径**的容错问题。而训练的**读路径**——每个 step 从磁盘取数据、解码、送到 GPU——是每天每秒都在发生的常态 I/O。它才是决定 GPU 是在算还是在等的第一道关卡。

→ [[06-数据加载流水线优化]]

---

> **参考资料**：
> - [PyTorch Distributed Checkpoint (DCP) 官方文档](https://pytorch.org/docs/stable/distributed.checkpoint.html)
> - [The Llama 3 Herd of Models（arXiv:2407.21783，故障恢复章节）](https://arxiv.org/abs/2407.21783)
> - [ByteCheckpoint: A Unified Checkpointing System for Large Foundation Model Development（arXiv:2407.20143，NSDI '25）](https://arxiv.org/abs/2407.20143)
> - [Universal Checkpointing: Efficient and Flexible Checkpointing for Large Scale Distributed Training（arXiv:2406.18820）](https://arxiv.org/abs/2406.18820)
> - [DeepSpeed Universal Checkpointing 博客](https://github.com/microsoft/DeepSpeed/blob/master/blogs/deepspeed-ucp/README.md)
> - [NVIDIA GPUDirect Storage 文档](https://docs.nvidia.com/gpudirect-storage/)
