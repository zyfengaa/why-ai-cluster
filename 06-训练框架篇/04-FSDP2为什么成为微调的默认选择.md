# Q: FSDP2 为什么成为微调的默认选择？

## 为什么会有这个问题

2024 年之前，分布式训练框架的格局是：DeepSpeed 做 ZeRO、Megatron 做 TP/PP。它们在各自领域做得很好，但都有一个共同的问题——**框架侵入性太强**，需要用户学习专用的配置和 API。

PyTorch 的回答是：**把分布式训练能力内置到框架中，让用户写原生 PyTorch 代码就能自动获得并行能力。**

## 解决什么问题

FSDP2（Fully Sharded Data Parallelism 2）是 PyTorch 从 2.5 版本开始主推的分布式训练 API。**它是 ZeRO-3 的 PyTorch 原生实现**，但 API 更简洁、生态更统一。

### FSDP1 → FSDP2 的演进

| 特性 | FSDP1（PyTorch 2.0-2.4） | FSDP2（PyTorch 2.5+） |
|------|------------------------|---------------------|
| **核心机制** | 基于 `FlatParameter`（把参数展平成 1D tensor 再分片） | **基于 DTensor**（保持原始形状，逻辑更清晰） |
| **API 风格** | 类封装 `FSDP(model)` | 函数式 `fully_shard(model)` |
| **torch.compile 兼容** | 不兼容 | ✅ **原生兼容** |
| **显存管理** | 显存约节省 7% vs FSDP1 | ✅ **更优** |
| **TP 组合** | 困难 | ✅ **DTensor 原生支持 TP** |
| **弹性 Checkpoint** | 不支持 | ✅ DCP 原生支持 |

### 核心设计：DTensor

FSDP2 的核心抽象是 **DTensor**——一个逻辑上的大张量，物理上分布到多个设备：

```python
from torch.distributed.tensor import DTensor, Shard, Replicate

# 一个 4096×4096 的权重，物理上分片到 8 张 GPU
dt = DTensor.from_local(
    local_tensor,        # 本卡上的 512×4096 切片
    device_mesh,         # 设备网格
    placements=[Shard(0)]  # 沿第 0 维切分
)
```

DTensor 让分片操作变得**透明**——你操作 DTensor 就像操作普通 Tensor，通信和同步由框架自动处理。

---

## FSDP2 在 2026 年的地位

### 为什么它成了默认选择

1. **PyTorch 官方主推**：FSDP1 在 PyTorch 2.11 起已废弃，新项目全部用 FSDP2
2. **生态集成最广**：HuggingFace Transformers、Lightning、Axolotl、TorchTitan 默认使用 FSDP2
3. **zero code intrusion**：`fully_shard()` 包装一下就完成分布式改造
4. **torch.compile 兼容**：在 H100 上 + float8 可以提升 50% 吞吐

### 实际性能数据

| 模型 | GPU 数 | FSDP2 + BF16 | FSDP2 + float8 | DeepSpeed ZeRO-3 |
|------|--------|-------------|----------------|-----------------|
| Llama 7B | 8×H100 | 35-45% MFU | ~50% MFU | 30-40% MFU |
| Llama 70B | 64×H100 | 40-48% MFU | ~55% MFU | 38-45% MFU |
| Llama 405B | 512×H100 | 38-45% MFU（+ TP8 + compile）| — | 35-42% MFU |

**关键发现**：FSDP2 在中小模型上的性能已经反超 DeepSpeed ZeRO-3。在大模型上两者接近，但 FSDP2 的 API 更简洁。

---

## 优缺点

| 优点 | 缺点 |
|------|------|
| ✅ **PyTorch 原生**：无外部依赖 | ❌ **不支持 PP**：没有流水线并行 |
| ✅ **极简 API**：一行 `fully_shard` 搞定 | ❌ **EP 支持有限**：MoE 场景不如 DeepSpeed |
| ✅ **torch.compile 兼容**：大幅加速 | ❌ **超大模型稳定性不如 Megatron**：万亿级尝试较少 |
| ✅ **弹性 Checkpoint**：拓扑变化也能恢复 | ❌ **NVMe Offload 不如 ZeRO-Infinity 成熟** |
| ✅ **HSDP（混合分片）**：节点内分片 + 节点间复制 | ❌ **CPU Offload 性能损失大** |

---

## 🛠 实操指南

### 📋 决策指南：FSDP2 适合你的场景吗？

```python
from torch.distributed.fsdp import ShardingStrategy

# FSDP2 四种分片策略
strategy = "FULL_SHARD"     # ZeRO-3 等价：参数+梯度+优化器状态全分片
# strategy = "SHARD_GRAD_OP"  # ZeRO-2 等价：只分片梯度+优化器状态
# strategy = "NO_SHARD"       # DDP 等价：不分片
# strategy = "HYBRID_SHARD"   # HSDP：节点内 FULL_SHARD + 节点间 NO_SHARD

# 推荐选择：
# < 7B, 显存充裕 → NO_SHARD（DDP）
# 7B-13B, 显存略紧张 → SHARD_GRAD_OP
# 13B-70B → FULL_SHARD
# 多节点，跨机通信慢 → HYBRID_SHARD
```

```bash
# FSDP2 最小示例
torchrun --nproc_per_node=8 train.py
```

```python
# train.py 核心代码
import torch
from torch.distributed.fsdp import fully_shard
from torch.distributed.pipelining import PipelineStage  # 可选

model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3-70b")

# 就这一行！
for layer in model.model.layers:
    fully_shard(layer)

# 享受分布式训练
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
for batch in dataloader:
    loss = model(**batch)
    loss.backward()
    optimizer.step()
```

### ⚠️ 常见误区

- **误区**："FSDP2 和 DeepSpeed ZeRO-3 完全一样"
  - **实际**：功能等价（都做参数分片），但实现不同。FSDP2 基于 DTensor，与 PyTorch 生态集成更好。DeepSpeed 支持更丰富的功能（Ulysses、MoE、NVMe Offload）。**两者不是替代关系，是不同权衡。**

- **误区**："FSDP2 只支持数据并行"
  - **实际**：FSDP2 + DTensor 支持 TP。在 PyTorch 2.5+ 中，`fully_shard()` 可以和 `tensor_parallel()` 注解组合使用。Meta 的 TorchTitan 参考实现展示了 FSDP2 + TP + PP + float8 的 4D 并行。

### 💡 一句话记住

> **"FSDP2 是 PyTorch 对 ZeRO-3 的'原生实现'——API 极致简洁（一行 fully_shard），生态无缝集成（HuggingFace + compile + DCP），而且免费内置于 PyTorch。2026 年微调的首选，没有之一。"**

---

## 对后续的铺垫

FSDP2 适合微调和小规模训练。DeepSpeed 适合内存受限场景。Megatron-Core 适合超大模型预训练。

那在实际项目中，到底应该选哪个？

→ [[05-框架选型决策矩阵]]

---

> **参考资料**：
> - [FSDP2 技术说明 (PyTorch 2.5+)](https://pytorch.org/docs/stable/fsdp.html)
> - [TorchTitan: Meta 的 FSDP2+TP+float8 参考实现](https://github.com/pytorch/torchtitan)
> - ["FSDP2 for 100B+ Models: Multi-Node Deep Dive" (techbytes, 2026)](https://techbytes.com/)
> - [Spheron: "Distributed LLM Training on GPU Cloud: FSDP vs DeepSpeed vs Megatron" (2026)](https://www.spheron.network/blog/)
