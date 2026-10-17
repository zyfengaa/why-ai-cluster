# Q: FSDP2 为什么成为微调的默认选择？

## 为什么会有这个问题

2024 年之前，分布式训练框架的格局是：DeepSpeed 做 ZeRO、Megatron 做 TP/PP。它们在各自领域做得很好，但都有一个共同的问题——**框架侵入性太强**，需要用户学习专用的配置和 API。

PyTorch 的回答是：**把分布式训练能力内置到框架中，让用户写原生 PyTorch 代码就能自动获得并行能力。**

## 解决什么问题

FSDP2（Fully Sharded Data Parallel 2）是 PyTorch 原生的 ZeRO-3 等价实现，从 **PyTorch 2.4/2.5 起可用，2.6 起 API 正式移入 `torch.distributed.fsdp`**。它基于 DTensor 重写了 FSDP1：参数不再被展平成一个 `FlatParameter`，而是保持原始形状按 per-parameter 分片。

### FSDP1 → FSDP2 的演进

| 特性 | FSDP1（PyTorch 2.0+） | FSDP2（PyTorch 2.4/2.5+） |
|------|------------------------|---------------------|
| **核心机制** | 基于 `FlatParameter`（参数展平成 1D 再分片） | **基于 DTensor**（per-parameter 分片，保持原始形状） |
| **API 风格** | 类封装 `FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD)` | 函数式 `fully_shard(module, ...)` + 2D `DeviceMesh` |
| **torch.compile 兼容** | 受限 | ✅ 原生兼容（这是重写 FSDP2 的主要动机之一） |
| **组合性** | 模块粒度粗 | ✅ 每个子模块可独立 `fully_shard`，可与 TP 组合 |
| **Checkpoint** | ✅ 可配合 DCP（torch.distributed.checkpoint，2.2 起） | ✅ 与 DCP 集成更紧密 |

注意：**"FSDP1 没有 DCP"是错误印象**——FSDP1 从 2.2 起就可以配合 Distributed Checkpoint 使用。FSDP2 的真正改进是 per-parameter 分片、与 torch.compile/TP 的组合性，以及整体上更干净的 DCP 集成。

### 核心设计：DTensor

FSDP2 的核心抽象是 **DTensor**——一个逻辑上的大张量，物理上分布到多个设备：

```python
from torch.distributed.tensor import DTensor, Shard, Replicate
from torch.distributed.device_mesh import DeviceMesh

# 一个 4096×4096 的权重，物理上分片到 8 张 GPU
mesh = DeviceMesh("cuda", list(range(8)))
dt = DTensor.from_local(
    local_tensor,          # 本卡上的 512×4096 切片
    mesh,                  # 设备网格
    placements=[Shard(0)], # 沿第 0 维切分
)
```

DTensor 让分片操作变得**透明**——你操作 DTensor 就像操作普通 Tensor，通信和同步由框架自动处理。

### FSDP2 在生态中的地位（2026 年口径）

1. **PyTorch 官方主推方向**：FSDP2 在 PyTorch 原生栈中逐步成为默认；但**生态迁移尚未完成**——TorchTitan 是 FSDP2-first 的参考实现，而 HuggingFace Transformers、Lightning、Axolotl 等框架仍以 FSDP1/DDP/DeepSpeed 为主，FSDP2 支持在逐步增加
2. **zero code intrusion**：对每个子模块调用一次 `fully_shard()` 就完成分片改造
3. **torch.compile 兼容**：可以与 FP8（Transformer Engine / torchao float8）叠加做进一步加速

### 适用边界

- **微调与中等规模预训练是甜点**：7B–70B、几十到几百卡，FSDP2 最省心
- **跨机 All-Gather 是主要开销**：ZeRO-3/FSDP2 类方案每步要 All-Gather 全部参数两次（前向+反向），跨机带宽低（IB 单端口 ~50 GB/s 量级）时，这个开销比 NVLink 域内大一个量级——多节点大规模预训练通常退回"TP/PP 切计算 + ZeRO-1/2 分优化器状态"的 Megatron 式组合
- **与 TP/PP 的组合**：FSDP2 本身只做数据并行分片；TP 通过 DTensor 的 2D DeviceMesh 可以组合（TorchTitan 演示了 FSDP2+TP 的做法），PP 在 PyTorch 侧另有 `torch.distributed.pipelining`，三者组合仍以 TorchTitan 等参考实现为主
- **`reshard_after_forward` 的权衡**：`True`（默认）= 前向后立即释放参数分片，显存最省，但反向前要再做一次 All-Gather（通信多）；`False` = 前向后保留完整参数，反向省一次通信，但参数常驻显存。显存紧就 `True`，通信贵、显存富就 `False`

---

## 优缺点

| 优点 | 缺点 |
|------|------|
| ✅ **PyTorch 原生**：无外部依赖 | ❌ **PP 不内置**：流水线并行需另用 `torch.distributed.pipelining` 或 TorchTitan 组合 |
| ✅ **极简 API**：对子模块逐个 `fully_shard` | ❌ **EP 不是核心能力**：MoE 专家并行需组合其他方案（如 TorchTitan MoE 或 TP/EP 方案） |
| ✅ **torch.compile 兼容**：可与 FP8 栈叠加 | ❌ **超大模型预训练验证少**：万亿级主要靠 Megatron 类方案 |
| ✅ **DCP checkpoint**：支持拓扑变化恢复 | ❌ **NVMe Offload 不如 ZeRO-Infinity 成熟** |
| ✅ **HSDP（混合分片）**：2D mesh 实现节点内分片 + 节点间复制 | ❌ **CPU Offload 支持有限**：FSDP1 的 CPUOffload 成熟，FSDP2 侧仍在完善 |

---

## 🛠 实操指南

### 📋 决策指南：FSDP2 适合你的场景吗？

FSDP2 没有 FSDP1 那种 `ShardingStrategy` 枚举（`FULL_SHARD`/`SHARD_GRAD_OP`/`NO_SHARD`/`HYBRID_SHARD` 是 **FSDP1 的 API**）。FSDP2 用 `reshard_after_forward` 控制前向后是否释放参数分片，用 2D DeviceMesh 表达 HSDP：

```python
# 版本注意：
#   PyTorch 2.5 及更早：fully_shard 在私有路径 torch.distributed.fsdp._fully_shard
#   PyTorch 2.6 起：官方公开路径为 torch.distributed.fsdp（下例以此为准）
from torch.distributed.fsdp import fully_shard
from torch.distributed.device_mesh import init_device_mesh
import torch

# 单机 8 卡：1D mesh，等价于 FSDP1 的 FULL_SHARD（ZeRO-3）
mesh = init_device_mesh("cuda", (8,), mesh_dim_names=("dp",))

# 多节点 HSDP：2D mesh——节点内 8 卡全分片，节点间复制
# mesh = init_device_mesh(
#     "cuda", (num_nodes, 8), mesh_dim_names=("replicate", "shard")
# )

model = LlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B", torch_dtype=torch.bfloat16
)

# 逐层包住：transformer layer 粒度是最常用做法
for layer in model.model.layers:
    fully_shard(layer, mesh=mesh, reshard_after_forward=True)
fully_shard(model, mesh=mesh, reshard_after_forward=False)  # 根模块也包

# 之后就是普通 PyTorch 训练循环
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
for batch in dataloader:
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

`reshard_after_forward` 的选择：显存紧 → `True`（反向多做一次 All-Gather）；通信贵、显存富 → `False`。只对最后一层/根模块通常设 `False`，因为反向紧跟其后，释放再取回纯属浪费。

```bash
torchrun --nproc_per_node=8 train.py
```

### ⚠️ 常见误区

- **误区**："FSDP2 和 DeepSpeed ZeRO-3 完全一样"
  - **实际**：功能等价（都分片参数+梯度+优化器状态），但实现不同。FSDP2 基于 DTensor，与 PyTorch 生态（compile、DCP、TP）集成更好；DeepSpeed 功能面更广（NVMe Offload、Ulysses、MoE）。**两者是不同权衡，不是替代关系。**

- **误区**："FSDP2 只支持数据并行，什么都做不了"
  - **实际**：FSDP2 通过 DTensor/DeviceMesh 可以与 TP 组合（TorchTitan 演示了 FSDP2+TP+PP+float8 的组合），也可以用 2D mesh 做 HSDP。但 EP 不是 FSDP2 核心能力，MoE 需要额外方案。

- **误区**："所有主流框架已经默认 FSDP2"
  - **实际**：只有 TorchTitan 是 FSDP2-first；HF Transformers、Lightning、Axolotl 等仍以 FSDP1/DDP/DeepSpeed 为主。选型前要确认你用的框架版本实际支持哪一代 API。

- **误区**："FSDP2 是 2.5 才有的全新功能，老版本什么都干不了"
  - **实际**：FSDP1 + DCP（2.2 起）同样能分片训练和做弹性 checkpoint；FSDP2 的增量是 per-parameter 分片、compile/TP 组合性和更干净的 API。

### 💡 一句话记住

> **"FSDP2 是 PyTorch 对 ZeRO-3 的原生重写——per-parameter 分片、DTensor 底座、一行 `fully_shard` 加一个 DeviceMesh 搞定，torch.compile 和 DCP 无缝衔接。它是微调场景的甜点：中等规模、代码侵入近乎为零。但跨机 All-Gather 开销和 EP 缺位决定了：超大规模预训练仍属 Megatron 式组合，生态迁移也还没完成。"**

---

## 对后续的铺垫

FSDP2 适合微调和中等规模训练。DeepSpeed 适合内存受限场景。Megatron-Core 适合超大模型预训练。

那在实际项目中，到底应该选哪个？

→ [[05-框架选型决策矩阵]]

---

> **参考资料**：
> - [PyTorch FSDP 官方文档（含 fully_shard，2.6+）](https://pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)
> - [PyTorch FSDP（FSDP1）文档](https://pytorch.org/docs/stable/fsdp.html)
> - [PyTorch TorchTitan GitHub（FSDP2 + TP + PP + float8 参考实现）](https://github.com/pytorch/torchtitan)
> - [Zhao et al., "PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel" (2023)](https://arxiv.org/abs/2304.11277)
> - [TorchTitan 论文 (2024)](https://arxiv.org/abs/2410.06511)
