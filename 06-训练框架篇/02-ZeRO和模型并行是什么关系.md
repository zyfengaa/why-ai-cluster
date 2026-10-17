# Q: ZeRO 和模型并行（TP/PP）是什么关系？

## 为什么会有这个问题

ZeRO-3 通过分片参数解决了"单卡放不下模型"的问题。TP/PP 也解决了"单卡放不下模型"的问题。那它们有什么区别？是竞争、互补、还是可以组合？

## 解决什么问题

### 两者切的对象不同

**ZeRO 切"存储"，TP/PP 切"计算"**——这是理解它们关系的核心：

- **ZeRO 分片的是模型状态**（权重、梯度、优化器状态），计算图本身不变：每张卡仍然要算每一层的完整前向/反向，只是在需要参数时通过 All-Gather 临时取回完整权重。**不修改模型代码**。
- **TP（张量并行）切的是单层内的矩阵乘法**：把一个大 GEMM 沿行或列切开，每张卡只算自己那一片。模型里单层的激活和权重都变小了，所以**单层放不下时只有 TP 能救**。需要按并行规范重写模型。
- **PP（流水线并行）切的是层与层之间**：把模型按层分段，不同段放不同卡，用 1F1B 等调度流水线填充气泡。

通信模式也不同：ZeRO-3 的 All-Gather/Reduce-Scatter 发生在**数据并行组**（每步一次，量是整个模型的参数量）；TP 的 All-Reduce 发生在**每层内部**（前向 2 次 + 反向 2 次，每步共 4×层数次），因此 TP 通常被限制在 NVLink 域内的高带宽组里。

### ZeRO 与 TP/PP 是互补关系，可以叠加

|- | ZeRO-3 能解决 | ZeRO-3 不能解决 |
|---|--------------|----------------|
| 模型状态（权重+梯度+优化器）太大 | ✅ 分片到多卡 | — |
| 单层激活/权重放不下、必须切计算 | — | ❌ **只有 TP 能切单层** |
| 层数太多、单卡存不下所有层 | ✅（计算仍是全模型） | ❌ 若单层就放不下，仍需 TP |
| 通信位置 | DP 组，每步一次 | TP 在层内每层 4 次，宜限 NVLink 域内 |

叠加时，模型状态被两级分片：TP 把每一层的状态切开，ZeRO 再把"各 TP 组持有的完整模型状态副本"沿 DP 维切开。

### 典型联合配置（算清每卡账）

```
70B 模型，128 GPU，TP=8，DP=16（DP 组内再上 ZeRO-3）

完整训练状态 = 70e9 × 16 B ≈ 1.12 TB
参数（BF16）  = 70e9 × 2 B  = 140 GB

TP=8：把每层权重/激活切成 8 份（模型状态也随之 ÷8）
ZeRO-3：把每个 TP 组的模型状态再 ÷16

每卡模型状态 ≈ 1.12 TB ÷ (8 × 16) ≈ 8.75 GB
每卡参数     ≈ 140 GB ÷ (8 × 16) ≈ 1.1 GB
```

关键对比：**只用 TP=8、不上 ZeRO 时，每卡仍要放 1.12 TB ÷ 8 = 140 GB 的完整训练状态副本**——8×80 GB 根本放不下。这就是"TP 只解决计算切分，优化器状态冗余要靠 ZeRO"的直观证明。

### 激活的账也要算：B×S×H×layers 与 B×heads×S²

ZeRO 和 TP/PP 的账本里，激活值是独立于模型状态的一项：

- **线性项（量级）**：每个 Transformer 层需要保存的中间激活约与 `B × S × H × layers × 每元素字节数` 成正比，系数在数十（数十个 H/层）量级——即几十个 `B×S×H` 的张量。Llama 3 70B（H=8192、80 层、B=4、S=4096、BF16）仅这一项就在数百 GB 量级。
- **平方项（attention）**：不做优化时，注意力分数矩阵是 `B × heads × S²`，长序列下随 S 平方增长，同样乘上 layers。上例 S=4096 时，仅分数矩阵一项也是数百 GB 量级。
- **FlashAttention 改变的是平方项**：它不把 `B×heads×S²` 的分数矩阵物化到 HBM，而是分块在 SRAM 内计算，激活里就只剩线性项。所以"开 FlashAttention 后激活从平方级降回线性级"，而 ZeRO/TP 对激活的作用是把它**分片**，激活重计算（checkpointing）的作用是**少存**——三者针对的是同一个账本里的不同项。

---

## 优缺点

| 维度 | ZeRO（数据并行分片） | TP/PP（模型并行切计算） |
|------|-------------------|----------------------|
| ✅ 代码侵入 | 几乎为零（封装层面） | 高（模型需按并行规范重写） |
| ✅ 通信位置 | DP 组内，每步一次，可跨机 | TP 层内每层 4 次，必须高带宽域 |
| ✅ 能切单层 | ❌ 不能（单层计算仍完整） | ✅ TP 切层内矩阵、PP 切层段 |
| ✅ 激活处理 | ❌ 不分片激活 | ✅ TP/SP 把激活也切开 |
| ✅ 优化器状态 | ✅ 分片到 1/N | ❌ TP/PP 不消除优化器状态副本（每个 DP 副本仍有一份） |
| ⚠️ 可组合 | ✅ 与 TP/PP 叠加（主流大模型训练标配） | ✅ TP×PP×DP 三维组合 |

**结论**：ZeRO 解决"模型状态冗余"（16 B/参数 × DP 副本数），TP 解决"单层放不下"，PP 解决"层数太多 + 流水线填充"。三者正交，可叠加。

### 选型策略

| 场景 | 推荐方案 | 理由 |
|------|---------|------|
| 每卡模型状态（16 B/参数）能放下 | DDP 或 ZeRO-1 | 无需切计算 |
| 模型状态分片后能放下，单层计算也放得下 | **ZeRO-2/3** | 不增加通信、不写模型代码 |
| 单层激活/权重计算放不下（层内 GEMM 太大） | **TP + ZeRO-2/3** | TP 切层，ZeRO 省模型状态 |
| 超大模型（千亿级以上）预训练 | **TP + PP + DP（ZeRO-1/2）** | Megatron 式 3D 并行 + 数据并行分片 |
| 小模型微调（7B 以下） | **ZeRO-1/2 或 DDP** | 无需复杂方案 |

---

## 🛠 实操指南

### 📋 决策指南：先算每卡预算，再选并行

正确的决策顺序是**先算账，再选工具**：

```python
# 第一步：算每卡模型状态预算
params = 70e9                 # 模型参数量
bytes_per_param = 16          # BF16 权重2 + 梯度2 + FP32 master 4 + Adam m/v 8
total_state_gb = params * bytes_per_param / 1024**3   # ≈ 1,043 GB（十进制约 1.12 TB）

tp = 8                        # 候选 TP 度
dp = 16                       # 数据并行度（总卡数 = tp × dp = 128）
zero = 3                      # ZeRO 阶段（1/2/3）

per_gpu_state = total_state_gb / tp * (1 / dp if zero == 3 else 1)
# zero=3：1,043/8/16 ≈ 8.2 GB/卡   ✅
# 仅 TP=8 不用 ZeRO：1,043/8 ≈ 130 GB/卡  ❌ 超过 80 GB 显存

# 第二步：算单层能否放下（决定要不要 TP）
hidden = 8192
# 单层最大的几个 GEMM 权重（BF16）：4 × H × H × 2 B ≈ 512 MB，TP=8 后约 64 MB
# 激活（B×S×H 量级）也随 TP 切分
per_layer_ok = per_gpu_state + 0.5 * hidden * hidden * 2 / 1024**3 < 80 * 0.6

# 第三步：算激活（决定要不要激活重计算 / FlashAttention）
B, S, heads, layers, bytes_per = 4, 4096, 64, 80, 2
linear_act = 30 * B * S * hidden * layers * bytes_per / 1024**3      # 线性项，系数为经验量级
attn_scores = B * heads * S * S * layers * bytes_per / 1024**3       # 不用 FlashAttention 时的平方项
print(f"线性激活 ~{linear_act:.0f} GB；attention 分数（若物化）~{attn_scores:.0f} GB")
# → 长上下文必开 FlashAttention；激活仍不够时用激活重计算 / 序列并行
```

关键点：

1. **先看每卡模型状态**（16 B/参数 ÷ TP ÷ ZeRO 分片度）：这一步决定 ZeRO 阶段和 DP 度。注意 TP=8 并不消除优化器状态副本——每个 TP 组仍持有一份完整优化器状态，所以 70B 在 8×80 GB 上**光靠 TP 放不下**，必须再叠 ZeRO。
2. **再看单层**：单层的权重和激活切分后能否放进显存，决定 TP 度。这个判断没有普适阈值（"单层占显存 30% 就要 TP"之类只是经验值），要按你的模型 H、序列长度和 batch 实测。
3. **最后算激活**：`B×S×H×layers×bytes` 的线性项 + `B×heads×S²×layers×bytes` 的 attention 项（FlashAttention 后不物化）。

### 💡 一句话记住

> **"ZeRO 分的是存储（模型状态 16 B/参数在 DP 组内分片），TP/PP 切的是计算（层内矩阵、层间流水线）。TP 不消除优化器状态副本——70B 在 8 卡上 TP=8 仍要每卡 140 GB 训练状态，必须再叠 ZeRO。三者正交可叠加：先算每卡状态预算，再定 TP/PP，最后给激活留账。"**

---

## 对后续的铺垫

ZeRO + TP/PP 的组合解决了"放得下"的问题。但在千亿级模型的预训练里，还需要更极致的通信效率、流水线调度和硬件适配——这就是 Megatron-LM/Megatron-Core 作为"重装底盘"的领域。

→ [[03-Megatron-LM]]

---

> **参考资料**：
> - [Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" (2020)](https://arxiv.org/abs/1910.02054)
> - [Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism" (2019)](https://arxiv.org/abs/1909.08053)
> - [Korthikanti et al., "Reducing Activation Recomputation in Large Transformer Models" (2022)](https://arxiv.org/abs/2205.05198)
> - [DeepSpeed 官方文档](https://www.deepspeed.ai/)
> - [PyTorch FSDP 官方文档](https://pytorch.org/docs/stable/fsdp.html)
