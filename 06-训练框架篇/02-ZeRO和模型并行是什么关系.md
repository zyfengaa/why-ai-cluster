# Q: ZeRO 和模型并行（TP/PP）是什么关系？

## 为什么会有这个问题

ZeRO-3 通过分片参数解决了"单卡放不下模型"的问题。TP/PP 也解决了"单卡放不下模型"的问题。那它们有什么区别？是竞争、互补、还是可以组合？

## 两种思路的对比

### ZeRO 的思路：**"按需重组"**

- 每张 GPU 只存模型的一小部分分片
- 前向传播时，从其他 GPU 获取完整参数（AllGather）
- 算完后释放不需要的参数
- **不需要修改模型代码**

### TP/PP 的思路：**"物理切分"**

- TP：把一层矩阵切成多份，每张 GPU 算一份后合并
- PP：把不同层的计算放到不同 GPU 上
- **需要修改模型代码**

### 不是一个维度的优化

```
ZeRO：数据并行中的显存优化（解决"冗余"）
TP：模型并行中的计算切分（解决"装不下"）
PP：模型并行中的流水线调度（解决"串行"）
```

**关键点**：ZeRO 是对**数据并行（DP）**的优化，不是对模型并行的替代。

---

## ZeRO + TP/PP 的联合使用

### 为什么需要联合

|- | ZeRO-3 能解决 | ZeRO-3 不能解决 |
|---|--------------|----------------|
| 模型参数太大 | ✅ 分片到多卡 | — |
| 单层太大放不下 | ❌ 分片只在**存储**时发生；计算某一层前会 AllGather 出该层**完整参数**，单层峰值显存不降 | ✅ **TP 才能切单层**，把层内矩阵也拆开 |

**注意**：ZeRO-3 每步会多出 2 次集体通信（AllGather / ReduceScatter），这是它的性能代价；但那属于"快不快"的问题，与"单层放不放得下"不是一回事，所以不放进上面这张表比较。

**结论**：ZeRO-3 解决"模型整体太大"（存不下所有层），TP 解决"单层太大"（一层就超显存）。两者互补。

### 典型联合配置

```
70B 模型，128 GPU，TP=8，DP=16（通过 ZeRO-3）

整个模型存不下：
  ZeRO-3 分片到 16 路 DP 组 → 每路 DP 存 ~9 GB 参数

单层还是太大（70B hidden_dim=8192）：
  TP=8 切到 8 卡 → 每卡存 ~64 MB

两层分片的叠加：
  每卡实际存 = 模型分片 ÷ TP degree
              = 140 GB ÷ 16 ÷ 8
              = ~1.1 GB 参数/卡
```

---

## ZeRO vs 模型并行的选择策略

| 场景 | 推荐方案 | 理由 |
|------|---------|------|
| 模型能单卡装下但要多卡加速 | **ZeRO-2** | 省显存 + 不增通信 |
| 模型太大但单层能装下 | **ZeRO-3** | 参数分片就够了 |
| 单层太大（hidden_dim > 单卡显存） | **TP + ZeRO-2/3** | TP 切层，ZeRO 省整体 |
| 超大规模预训练（405B+） | **TP + PP + DP（ZeRO-3）** | 3D 并行 + ZeRO 分片 |
| 小模型微调（< 7B） | **ZeRO-1 或不用** | 不需要复杂方案 |

### 2026 年实践共识

1. **ZeRO-3 不是 TP/PP 的替代品**——它们解决不同的问题
2. **ZeRO-3 + TP 是最常见的组合**——TP 在 NVLink 域内（8 卡），ZeRO-3 处理跨域的数据并行
3. **ZeRO-3 + TP + PP 是超大模型的标配**——但 PP 可以用 Zero Bubble 降低气泡
4. **FSDP2 与 ZeRO-3 机制目标等价**（都分片参数/梯度/优化器状态），但**并不等价**：切分粒度（FSDP2 的 per-parameter DTensor vs ZeRO-3 的 flat bucket）、通信调度、offload 成熟度（ZeRO-Infinity 远强于 FSDP2）、checkpoint 生态（FSDP2 绑 DCP）都不同——**两者是不同权衡，不是替代关系**

---

## 🛠 实操指南

### 📋 决策指南：ZeRO 还是 TP？

```
单层的「参数 AllGather 峰值 + 激活峰值」会超过单卡可用显存？
├─ 是 → TP 必须（ZeRO 切不动单层）
└─ 否 →
   模型总显存 > 单卡显存？
   ├─ 是 → ZeRO-3（参数分片即可）
   └─ 否 → 仅 DP + ZeRO-1/2
```

```python
# 判断是否需要 TP：看"单层峰值"是否放得下，而不是看模型总量
hidden_dim = 8192
batch_size = 4
seq_len = 4096
precision = 2  # FP16/BF16 bytes

# ① 参数 AllGather 峰值：ZeRO-3 计算某层前要把该层完整参数聚合出来
#    单层 MLP 是 W1/W2 两个矩阵（不是 4 个）
single_layer_params = 2 * hidden_dim * hidden_dim * precision
# ② 激活峰值：该层前向要保存的中间激活（attention + MLP，粗估 4× hidden）
single_layer_act = batch_size * seq_len * hidden_dim * precision * 4
# ③ 优化器状态已被分片到各卡，单卡只担 1/N，不会成为单层峰值的主导项

single_layer_mem = (single_layer_params + single_layer_act) / 1024**3
gpu_mem = 80  # H100 显存 GB

print(f"单层峰值显存: {single_layer_mem:.1f} GB")
print(f"占 H100 比例: {single_layer_mem/gpu_mem*100:.0f}%")
if single_layer_mem > gpu_mem:
    print("→ 需要 TP")
else:
    print("→ 不需要 TP，ZeRO-3 就够了")
```

### 💡 一句话记住

> **"ZeRO 解决的是模型整体放不下的问题（太多层），TP 解决的是单层放不下的问题（层内矩阵太大）。ZeRO 是'分散存储，按需重组'，TP 是'物理切开，各自计算再合并'。它们不是替代关系，是搭档。"**

---

> **参考资料**：
> - [Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" (2020)](https://arxiv.org/abs/1910.02054)
> - [Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism" (2019)](https://arxiv.org/abs/1909.08053)
> - [DeepSpeed ZeRO 官方教程](https://www.deepspeed.ai/tutorials/zero/)
> - [FSDP2 与 TP 组合实践（`fully_shard` API，PyTorch 2.5+）](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)
> - [Megatron-Core 并行指南](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
