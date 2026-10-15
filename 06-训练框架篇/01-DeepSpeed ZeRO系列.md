# Q: DeepSpeed ZeRO 系列在解决什么问题？

## 为什么会有这个问题

数据并行（DDP）有一个巨大的浪费：**每张 GPU 都存了完整的模型参数、梯度和优化器状态。** 如果 64 张 GPU 训练一个 70B 模型，每张卡都存了 140 GB 的 FP16 权重、140 GB 的 FP16 梯度、280 GB 的 FP32 master 权重和 560 GB 的 Adam 状态（动量 m + 方差 v）……合计 1120 GB（16 bytes/参数）。这些数据在每张卡上都是**一模一样**的。

**核心矛盾**：数据并行为了独立计算，每张卡都要有完整副本。但这些副本是冗余的——算梯度时并不需要每张卡都有完整的优化器状态。

## 解决什么问题：ZeRO（Zero Redundancy Optimizer）

ZeRO 的核心思想：**冗余数据不要存 N 份，而是每张卡只存一份，需要的时候再通信获取。**

### ZeRO 的三个阶段

| 阶段 | 分片内容 | 显存节省（相对 DDP） | 通信开销 |
|------|---------|-------------------|---------|
| **ZeRO-1** | 仅优化器状态（Adam：参数×2 的动量+方差） | ~4× | 不变（只减少存储） |
| **ZeRO-2** | 优化器状态 + 梯度 | ~8× | 不变 |
| **ZeRO-3** | 优化器状态 + 梯度 + 模型参数 | **~N×**（N=GPU 数） | **增加 50%**（需要 AllGather 参数） |

**三个容易混淆的变体**（它们不是"第四个阶段"，而是不同方向的补充）：
- **ZeRO-Offload**：把优化器状态和优化器计算**放到 CPU**（配合 `DeepSpeedCPUAdam`），单卡即可训练 > 13B 参数的模型
- **ZeRO-Infinity**：在 ZeRO-Offload 基础上再加 **NVMe**，形成 CPU + NVMe 两级卸载
- **ZeRO++**：用量化通信（qgZ/qwZ/hpZ）**降低 ZeRO-3 的通信量**，把"增加 50% 通信"的代价压下来

### ZeRO 的各阶段显存节省

```
70B 模型（混合精度，16 bytes/参数），64 卡训练：

账本（每卡完整副本）：
  权重（FP16/BF16） 140 GB + 梯度（FP16/BF16） 140 GB
  + FP32 master 权重 280 GB + Adam 动量 m 280 GB + Adam 方差 v 280 GB
  = 1120 GB = 16 bytes/参数

DDP：每卡需要
  权重 140 GB + 梯度 140 GB + FP32 master 280 GB + Adam m/v 560 GB + 激活值 ~50 GB = ~1170 GB ❌ 放不下

ZeRO-1：每卡需要（只分片优化器状态：840 GB → 840/64 ≈ 13.1 GB）
  权重 140 GB + 梯度 140 GB + 优化器状态 13.1 GB + 激活值 ~50 GB = ~343 GB ❌ 还是放不下

ZeRO-2：每卡需要（再分片梯度：140/64 ≈ 2.2 GB）
  权重 140 GB + 梯度 2.2 GB + 优化器状态 13.1 GB + 激活值 ~50 GB = ~205 GB ❌

ZeRO-3：每卡需要（连权重也分片：140/64 ≈ 2.2 GB）
  权重 2.2 GB + 梯度 2.2 GB + 优化器状态 13.1 GB + 激活值 ~50 GB = ~68 GB ✅ 可以装下！
```

**关键洞察**：ZeRO-3 让 70B 模型能在 64 张 H100 上训练——按 16 bytes/参数的账本，DDP 单卡要 ~1170 GB、ZeRO-2 也需要 ~205 GB/卡，都远超 80 GB 的 H100；只有 ZeRO-3 能把每卡压进 ~68 GB。

### ZeRO 的通信模式

ZeRO-3 的通信比 DDP 多了一步：

```
DDP（传统数据并行）：
  1. 每张卡独立前向+反向 → 算好全量梯度
  2. 一次 All-Reduce 同步梯度
  3. 每张卡独立更新参数

ZeRO-3：
  1. 前向传播前：AllGather 获取分片在其他卡的参数（①）
  2. 前向传播中：用完整参数计算
  3. 反向传播中：AllGather 获取梯度分片（②）→ ReduceScatter 分片梯度（③）
  4. 每张卡只更新自己分片的那部分参数
```

通信从 DDP 的 "1 次 All-Reduce" 变成了 ZeRO-3 的 "3 次集体通信"。**这就是 ZeRO-3 的性能代价。**

**注意**：ZeRO-1/2 的"通信开销不变"，指的是**通信量相同**——ZeRO-2 用 **ReduceScatter + AllGather** 替代了 DDP 的单次 AllReduce，搬的总字节数一样（这正是它被称作"甜点"、几乎无代价的原因）；ZeRO-3 才因为每步要多一轮参数 AllGather 而增加约 50% 通信量。

---

## 优缺点

| 优点 | 缺点 |
|------|------|
| ✅ **消除冗余**：每张 GPU 只存 1/N 的模型 | ❌ **增加 50% 通信**：ZeRO-3 每步多 2 次通信 |
| ✅ **无需修改模型代码**：DeepSpeed 封装对用户透明 | ❌ **CPU Offload 性能下降明显**：NVMe 卸载更慢 |
| ✅ **与 TP/PP 兼容**：可以组合使用 | ❌ **小模型上优势不大**：7B 模型用 ZeRO 收益有限 |
| ✅ **ZeRO-Infinity**：支持 CPU + NVMe 二级卸载（论文口径：单台 DGX-2 可微调 10T+ 参数模型） | ❌ **配置项繁多**：ZeRO / Offload / 通信各有独立配置块，调优门槛高 |

**关于维护状态**：DeepSpeed 并未放缓——仓库已迁至 `deepspeedai/DeepSpeed`，2025 年发到 0.17/0.18，2026 年有 0.19 与 SuperOffload（ASPLOS 2026），仍是活跃的高频迭代项目。

---

## 🛠 实操指南

### 📋 决策指南：你的模型用哪个 ZeRO 阶段？

```
模型能否放进单卡？
├─ 能 → 不需要 ZeRO，DDP 就够
└─ 不能 →
   ├─ 显存差一点（< 2×）→ ZeRO-1（优化器状态分片）
   ├─ 显存差很多（< 4×）→ ZeRO-2（优化器+梯度分片）
   └─ 完全放不下（> 8×）→ ZeRO-3（全部三部分分片）
       └─ 依然放不下？→ 加 ZeRO-Infinity Offload
```

**一条经验**：2026 年的实践中，**ZeRO-2 是"甜点"**——显存节省 8×，通信开销不增加。ZeRO-3 虽然能省更多显存，但通信开销增加 50%，仅在必要时使用。

### 🔧 自己动手验：ZeRO 配置

```python
# DeepSpeed ZeRO 配置示例（ds_config.json）
{
  "zero_optimization": {
    "stage": 2,  # 先试 ZeRO-2，不够再升到 3
    "reduce_bucket_size": 5e8,
    "overlap_comm": true  # 通信与计算重叠
    # 注意：allgather_bucket_size 只对 stage 3 有意义（控制参数 AllGather 的桶大小），
    #       stage 2 不做参数 AllGather，写在这里不生效
  },
  "gradient_accumulation_steps": 4,
  "train_batch_size": 128,
  "fp16": {
    "enabled": true
  }
}

# ZeRO-3 + CPU Offload（极端情况）
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    # offload_optimizer 时务必使用 DeepSpeedCPUAdam（zero_force_ds_cpu_optimizer 默认为 true）；
    # 若关掉，CPU Adam 会回退到 PyTorch 版实现，慢很多
    "zero_force_ds_cpu_optimizer": true,
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    }
  }
}
```

```python
# 查看 ZeRO 的显存收益——注意 DeepSpeed 并不会打印
# "ZeRO: reducing memory from X GB to Y GB" 这种固定字样。
# 最可靠的方式是自己在训练循环里打点，对比开/关 ZeRO 前后的峰值显存：
import torch
print("peak allocated (GB):", torch.cuda.max_memory_allocated() / 1024**3)

# 或在任意时刻手动打印：
#   import deepspeed
#   deepspeed.utils.see_memory_usage("after zero init", force=True)
```

### ⚠️ 常见误区

- **误区**："ZeRO-3 总是最好的，因为显存最省"
  - **实际**：ZeRO-3 增加的通信开销可能导致**吞吐下降 20-30%**。但要注意限定条件——**70B 全参预训练/微调在 80 GB 卡上必须用 ZeRO-3（或 FSDP2 全分片，`reshard_after_forward=True`）**：按 16 bytes/参数的账本，ZeRO-2 仍有 ~205 GB/卡，根本放不下。只有当**只训练少量参数**时（如 LoRA，底座冻结、优化器状态极小），ZeRO-2 才够用——这一点常被笼统说成"70B 用 ZeRO-2 就行"，是误导。

- **误区**："ZeRO 和模型并行是互相替代的"
  - **实际**：ZeRO 解决的是**数据并行中的内存冗余**，TP/PP 解决的是**模型放不进单卡**。它们是互补的。实践中，ZeRO-3 + TP 是 70B+ 模型的常见组合。

- **误区**："ZeRO-Infinity 可以让你在任何 GPU 上跑任何模型"
  - **实际**：Offload 到 CPU 或 NVMe 会带来**巨大的性能损失**。从 GPU HBM 到 CPU DDR 的带宽差 10-50 倍，到 NVMe 差 50-500 倍。ZeRO-Infinity 是"能跑"的方案，不是"能高效跑"的方案。

### 📊 底线数字速查

| ZeRO 阶段 | 显存节省（vs DDP） | 通信开销增加 | 推荐场景 |
|-----------|-------------------|-------------|---------|
| 无（DDP） | 1× | 0% | 模型能装下单卡 |
| **ZeRO-1** | **~4×** | **0%** | 优化器状态太大 |
| **ZeRO-2** | **~8×** | **0%** | 显存略紧张，推荐甜点 |
| **ZeRO-3** | **~N×** | **~50%** | 模型放不进单卡时必选 |
| ZeRO-Infinity | 极大 | 100%+ (offload) | 单机跑 10T+ 参数模型（论文口径） |

### 💡 一句话记住

> **"ZeRO 不是说'大家一起存同样的数据太浪费了'——把优化器状态、梯度、参数也分片到各卡，需要的时候再通信获取。ZeRO-2 是最佳甜点：省 8× 显存、不增通信；ZeRO-3 是终极方案：省 N× 显存、多 50% 通信。"**

---

## 对后续的铺垫

ZeRO 解决了数据并行中的内存冗余问题。但有些场景 ZeRO 帮不了你——比如模型的**单层太大**，一张 GPU 连一层都放不下。这是模型并行（TP/PP）的领域。

ZeRO 和模型并行是什么关系？是竞争、互补、还是结合使用？

→ [[02-ZeRO和模型并行是什么关系]]

---

> **参考资料**：
> - [Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" (2020)](https://arxiv.org/abs/1910.02054)
> - [Ren et al., "ZeRO-Offload: Democratizing Billion-Scale Model Training" (2021)](https://arxiv.org/abs/2101.06840)
> - [Rajbhandari et al., "ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning" (2021)](https://arxiv.org/abs/2104.07857)
> - [Wang et al., "ZeRO++: Extremely Efficient Collective Communication for Giant Model Training" (2023)](https://arxiv.org/abs/2306.10209)
> - [DeepSpeed ZeRO 官方教程](https://www.deepspeed.ai/tutorials/zero/)
> - [DeepSpeed 仓库（deepspeedai/DeepSpeed）](https://github.com/deepspeedai/DeepSpeed)
