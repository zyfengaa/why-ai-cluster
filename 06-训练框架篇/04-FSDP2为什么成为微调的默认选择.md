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
| **显存管理** | 基准 | ✅ FSDP2 相比 FSDP1 约省 7% 显存 |
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

1. **PyTorch 官方主推**：FSDP1 已进入 deprecated 状态（旧的 `fully_shard` 实验前端在 2.4 打弃用警告、2.5 后移除），新项目全部用 FSDP2；生态侧 HF Transformers 已把 FSDP2 设为默认（PR #45640），Axolotl 已彻底移除 FSDP1
2. **生态集成最广**：HuggingFace Transformers、Lightning、Axolotl、TorchTitan 默认使用 FSDP2
3. **zero code intrusion**：`fully_shard()` 包装一下就完成分布式改造
4. **torch.compile 兼容**：在 H100 上叠加 `torch.compile` + FP8 可累计提升约 38% 吞吐（TorchTitan 实测，8B 模型）

### 实际性能数据

下表来自 TorchTitan 在 H100 上对 Llama 3 8B 的实测（来源见文末参考资料的 torchtitan float8 文档）：

| 配置（Llama 3 8B, H100） | 相对 baseline 吞吐 | 备注 |
|------|-------------------|------|
| FSDP2 + BF16（baseline） | 1× | 8B 约 48% MFU |
| + `torch.compile` | +10.7% | |
| + FP8 linear（float8 GEMM） | +33% | |
| + FP8 all-gather | 累计 +38% | 三项叠加 |

**关键发现**：这些收益是可叠加的——`torch.compile`（+10.7%）、`FP8 linear`（+33%），再叠上 `FP8 all-gather` 累计 +38%，8B 模型可达约 48% MFU。FSDP2 的优势主要来自 API 简洁 + 与 compile/float8 的无缝组合，而不是单个"魔法开关"。

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
# FSDP2 不再使用 FSDP1 的 ShardingStrategy 枚举，
# 而是用 reshard_after_forward 布尔值 + device mesh 来表达策略。
# FSDP1 → FSDP2 等价映射：
#   FULL_SHARD          → reshard_after_forward=True
#   SHARD_GRAD_OP       → reshard_after_forward=False
#   HYBRID_SHARD        → reshard_after_forward=True  + 2D device mesh
#   _HYBRID_SHARD_ZERO2 → reshard_after_forward=False + 2D device mesh
#   NO_SHARD            → 不调用 fully_shard（等价 DDP）
reshard_after_forward = True

# 推荐选择：
# < 7B, 显存充裕 → 不调用 fully_shard（等价 NO_SHARD / DDP）
# 7B-13B, 显存略紧张 → reshard_after_forward=False（等价 SHARD_GRAD_OP）
# 13B-70B → reshard_after_forward=True（等价 FULL_SHARD）
# 多节点，跨机通信慢 → 2D mesh（replicate × shard）+ reshard_after_forward=True（等价 HYBRID_SHARD）
```

```bash
# FSDP2 最小示例
torchrun --nproc_per_node=8 train.py
```

```python
# train.py 核心代码
import os
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

# 0. 初始化进程组与 device mesh（FSDP2 必须）
dist.init_process_group("nccl")
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
mesh = init_device_mesh("cuda", (dist.get_world_size(),))

model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-3-70b")

# 1. 自底向上分片：每个 layer 一个 FSDP 组
for layer in model.model.layers:
    fully_shard(layer, mesh=mesh)
# 2. 最后对根模型再调一次（FSDP2 官方要求自底向上）
fully_shard(model, mesh=mesh)

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
  - **实际**：FSDP2 + DTensor 支持 TP。在 PyTorch 2.5+ 中，`fully_shard()` 可以和 `tensor_parallel()` 注解组合使用。Meta 的 TorchTitan 参考实现展示了 FSDP2 + TP + float8 的组合。

- **⚠️ 什么情况下 `torch.compile` 反而更慢**
  - **实际**：`torch.compile` 与 FSDP2 组合存在 graph break 与 grid 限制；已知把 `nn.Linear` 单独 compile 再叠加 selective activation checkpoint 时，会因反复重编译导致 TFLOPS 从 386 掉到 126（torchtitan issue #786）。**不是所有配置下 compile 都会变快**——上线的组合要先做端到端 profiling。

### 💡 一句话记住

> **"FSDP2 是 PyTorch 对 ZeRO-3 的'原生实现'——API 极致简洁（一行 fully_shard），生态无缝集成（HuggingFace + compile + DCP），而且免费内置于 PyTorch。2026 年微调的首选，没有之一。"**

---

## 对后续的铺垫

FSDP2 适合微调和小规模训练。DeepSpeed 适合内存受限场景。Megatron-Core 适合超大模型预训练。

那在实际项目中，到底应该选哪个？

→ [[05-框架选型决策矩阵]]

---

> **参考资料**：
> - [FSDP2 API：`fully_shard`（PyTorch 官方文档）](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)
> - [FSDP 教程（PyTorch 官方）](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
> - [TorchTitan：Meta 的 FSDP2 + TP + float8 参考实现](https://github.com/pytorch/torchtitan)
> - [TorchTitan float8 文档（本文性能数据来源）](https://github.com/pytorch/torchtitan/blob/main/docs/float8.md)
> - [HuggingFace Transformers PR #45640：FSDP2 设为默认](https://github.com/huggingface/transformers/pull/45640)
> - [Axolotl 多卡训练文档（已移除 FSDP1）](https://docs.axolotl.ai/docs/multi-gpu.html)
