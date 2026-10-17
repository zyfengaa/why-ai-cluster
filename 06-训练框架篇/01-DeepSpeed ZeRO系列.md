# Q: DeepSpeed ZeRO 系列在解决什么问题？

## 为什么会有这个问题

数据并行（DDP）有一个巨大的浪费：**每张 GPU 都存了完整的模型参数、梯度和优化器状态。** 用 Adam 做混合精度训练时，完整训练状态是 **16 B/参数**：

```
BF16 权重         2 B/参数
BF16 梯度         2 B/参数
FP32 master 权重  4 B/参数
Adam 动量 m (FP32) 4 B/参数
Adam 方差 v (FP32) 4 B/参数
─────────────────────────
合计              16 B/参数
```

一个 70B 模型的完整训练状态 = 70×10⁹ × 16 B ≈ **1.12 TB**（其中仅 Adam 的 m+v 就是 560 GB，FP32 master 再加 280 GB）。如果 64 张 GPU 做数据并行，这 1.12 TB 在每张卡上都是**一模一样**的副本。

**核心矛盾**：数据并行为了独立计算，每张卡都要有完整副本。但这些副本是冗余的——算梯度时并不需要每张卡都有完整的优化器状态。

## 解决什么问题：ZeRO（Zero Redundancy Optimizer）

ZeRO 的核心思想：**冗余的模型状态不要存 N 份，而是每张卡只存 1/N，需要的时候再通信获取。**

### ZeRO 的三个阶段

下表中的"节省倍数"**只针对模型状态（权重+梯度+优化器状态），不包含激活值**——激活值在三个阶段下都不分片，只能靠激活重计算/序列并行来省。

| 阶段 | 分片内容 | 模型状态节省（相对 DDP） | 通信量 |
|------|---------|-------------------|---------|
| **ZeRO-1** | 仅优化器状态（FP32 master + m + v，12 B/参数） | 16/4 = **4×** | 与 DDP 相同 |
| **ZeRO-2** | 优化器状态 + 梯度（14 B/参数） | 16/2 = **8×** | 与 DDP 相同（reduce-scatter 替代 all-reduce，总量不变） |
| **ZeRO-3** | 优化器状态 + 梯度 + 权重（全部 16 B/参数） | **N×**（N=数据并行度） | 最多约 **1.5×**（前向/反向各多一次参数 All-Gather） |

### ZeRO 的各阶段显存账本（70B 模型，64 卡）

```
70B 模型完整训练状态 = 70e9 × 16 B = 1,120 GB，激活值另计（设约 50 GB）：

DDP：每卡
  权重 140 + 梯度 140 + master 280 + m 280 + v 280 + 激活 ~50 ≈ 1,170 GB ❌

ZeRO-1：优化器状态分片到 64 卡
  权重 140 + 梯度 140 + (master+m+v) 840/64 ≈ 13 + 激活 ~50 ≈ 343 GB ❌

ZeRO-2：优化器状态 + 梯度分片
  权重 140 + 梯度 140/64 ≈ 2.2 + 13 + 激活 ~50 ≈ 205 GB ❌

ZeRO-3：模型状态全分片
  权重 140/64 ≈ 2.2 + 梯度 2.2 + 13 + 激活 ~50 ≈ 68 GB ✅ 放得下
```

注意三点：

1. **模型状态从 1,120 GB 降到约 18 GB，是 64× 的下降；但端到端显存不是 64×**——激活值（约 50 GB）始终在，实际每卡占用约 68 GB 而不是 1,170/64 ≈ 18 GB。
2. 端到端收益 = 模型状态下降幅度 × 激活在显存中的占比。激活占比越高（长序列、大 batch），ZeRO 的相对收益越小，越需要配合激活重计算。
3. 上表的 4×/8×/N× 是**理论模型状态分片倍数**；ZeRO 论文里报告的端到端加速（对同规模基线约 1.8×～3× 量级）是在特定模型规模/卡数/激活占比下测得的，换一个设置不能直接套用。

**关键洞察**：ZeRO-3 让 70B 模型能在几十张 80 GB 卡上训练——没有它，每卡 1.12 TB 的训练状态根本无处安放。

### ZeRO 的通信模式

ZeRO-3 的通信比 DDP 多了一步：

```
DDP（传统数据并行）：
  1. 每张卡独立前向+反向 → 算好全量梯度
  2. 一次 All-Reduce 同步梯度（等价于 reduce-scatter + all-gather）
  3. 每张卡独立更新参数

ZeRO-3：
  1. 前向每一层前：All-Gather 取回该层的完整参数（①），用完即可释放
  2. 反向每一层前：再次 All-Gather 取回该层参数（②）
  3. 反向算完该层梯度后：Reduce-Scatter 把梯度归约成分片（③）
  4. 每张卡只更新自己持有分片的参数
```

即 DDP 的 "reduce-scatter + all-gather（约 2S 通信量）" 变成了 ZeRO-3 的 "两次参数 All-Gather + 一次梯度 Reduce-Scatter（最多约 1.5× DDP 的通信量）"。**这就是 ZeRO-3 的性能代价。**

---

## 优缺点

| 优点 | 缺点 |
|------|------|
| ✅ **消除冗余**：每张 GPU 只存 1/N 的模型状态 | ❌ **ZeRO-3 通信更多**：前向/反向各多一次参数 All-Gather，通信量最多 1.5× |
| ✅ **无需修改模型代码**：DeepSpeed 封装对用户透明 | ❌ **CPU/NVMe Offload 有明确带宽代价**：见下文，不是"免费显存" |
| ✅ **与 TP/PP 兼容**：可以组合使用 | ❌ **激活不分片**：长上下文/大 batch 场景还需配合激活重计算或序列并行 |
| ✅ **ZeRO-Infinity**：CPU + NVMe 二级卸载，把"放不下"变成"能跑" | ❌ **卸载路径慢一个数量级以上**：只适合低吞吐可接受的场景 |

---

## 🛠 实操指南

### 📋 决策指南：你的模型用哪个 ZeRO 阶段？

先算账再选阶段——每卡需要容纳的模型状态 = 16 B/参数 × 参数量 ÷ 阶段分片度：

```
每卡完整训练状态（16 B/参数）能放进单卡显存？
├─ 能 → DDP 就够
└─ 不能 →
   ├─ 优化器状态分片后能放下（÷N 后 < 显存的 60% 左右）→ ZeRO-1
   ├─ 再分梯度后能放下（÷N 后 < 显存的 60% 左右）→ ZeRO-2
   └─ 权重也要分片才能放下 → ZeRO-3
       └─ 依然放不下？→ ZeRO-3 + CPU/NVMe Offload（接受吞吐损失）
```

例子：70B 在 64×80 GB 上，ZeRO-2 后每卡仍需约 205 GB（含激活），放不下 → 选 ZeRO-3。

### 🔧 自己动手验：ZeRO 配置

```python
# DeepSpeed ZeRO 配置示例（ds_config.json），字段以官方 config-json 文档为准
{
  "bf16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 2,  # 先试 ZeRO-2，不够再升到 3
    "allgather_bucket_size": 5e8,
    "reduce_bucket_size": 5e8,
    "overlap_comm": true  # 通信与计算重叠
  },
  "gradient_accumulation_steps": 4,
  "train_batch_size": 128
}

# ZeRO-3 + CPU/NVMe Offload（放不下时的兜底方案）
{
  "zero_optimization": {
    "stage": 3,
    "contiguous_gradients": true,
    "overlap_comm": true,
    "offload_optimizer": {
      "device": "cpu",      # 也可设为 "nvme"，需配 "nvme_path"
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    }
  }
}
```

字段核对：`zero_optimization.stage`（0–3）、`offload_optimizer`、`offload_param`、`allgather_bucket_size`、`reduce_bucket_size`、`overlap_comm` 均为 DeepSpeed 官方配置项（见参考资料中的 config-json 文档）。

```bash
# 查看 ZeRO 的显存节省效果（DeepSpeed 启动日志）
grep -E "ZeRO|memory|Leaked|memory allocated" training.log | tail -10
```

### ⚠️ 常见误区

- **误区**："ZeRO-3 总是最好的，因为显存最省"
  - **实际**：ZeRO-3 多出两次参数 All-Gather，在跨机带宽有限时吞吐损失明显。模型状态分片后就能放下的场景，ZeRO-2（不增加通信量）通常吞吐更高。

- **误区**："ZeRO 和模型并行是互相替代的"
  - **实际**：ZeRO 解决的是**数据并行中的模型状态冗余**，TP/PP 解决的是**单层/单段计算放不进单卡**。它们是互补的，实践中常组合使用（详见下一篇）。

- **误区**："ZeRO-Infinity 可以让你在任何 GPU 上免费扩展显存"
  - **实际**：Offload 的带宽账很清楚——GPU HBM 带宽 3 TB/s 量级，PCIe 5.0 x16 到 CPU 内存双向只有 128 GB/s（差 20 倍以上），NVMe 顺序读普遍只有几 GB/s（差数百倍）。优化器卸载意味着每步 optimizer step 的数据要走一遍 PCIe 甚至 NVMe，吞吐下降是数量级的。ZeRO-Infinity 是"能跑"的方案，不是"能高效跑"的方案。

- **误区**："选阶段看模型参数量阈值就行"
  - **实际**：应该按**每卡模型状态预算**（16 B/参数 × N 卡分片）+ 激活占用来算，同一模型在不同卡数下的正确阶段不同。

### 📊 底线数字速查

| ZeRO 阶段 | 模型状态节省（vs DDP） | 通信量 vs DDP | 推荐场景 |
|-----------|-------------------|-------------|---------|
| 无（DDP） | 1× | 1× | 模型状态能装下单卡 |
| **ZeRO-1** | **4×** | ≈1× | 优化器状态是瓶颈 |
| **ZeRO-2** | **8×** | ≈1× | 分片后能放下时的甜点 |
| **ZeRO-3** | **N×** | 最多约 1.5× | 模型状态分到最后也放不下 |
| ZeRO-3 + Offload | 超出 GPU 显存 | + PCIe/NVMe 带宽代价 | 兜底：能跑优先 |

### 💡 一句话记住

> **"ZeRO 把 16 B/参数的模型状态（权重+梯度+FP32 master+Adam m/v）从每卡一份分片成 1/N 份，需要时再通信取回：ZeRO-1/2 省 4×/8× 且不增加通信量，ZeRO-3 省 N× 但最多多 50% 通信。激活值不分片，账要分开算；CPU/NVMe 卸载是拿带宽换显存，不是免费午餐。"**

---

## 对后续的铺垫

ZeRO 解决了数据并行中的模型状态冗余问题。但有些场景 ZeRO 帮不了你——比如模型的**单层太大**，一张 GPU 连一层的计算都放不下。这是模型并行（TP/PP）的领域。

ZeRO 和模型并行是什么关系？是竞争、互补、还是结合使用？

→ [[02-ZeRO和模型并行是什么关系]]

---

> **参考资料**：
> - [Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" (2020)](https://arxiv.org/abs/1910.02054)
> - [Rajbhandari et al., "ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning" (2021)](https://arxiv.org/abs/2104.07857)
> - [DeepSpeed 官方文档：config-json 配置项](https://www.deepspeed.ai/docs/config-json/)
> - [DeepSpeed 官方教程：ZeRO-Offload](https://www.deepspeed.ai/tutorials/zero-offload/)
> - [Wang et al., "ZeRO++: Extremely Efficient Collective Communication for Giant Model Training" (2023)](https://arxiv.org/abs/2306.10209)
> - [microsoft/DeepSpeed GitHub](https://github.com/microsoft/DeepSpeed)
