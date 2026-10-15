# Q: Megatron-LM 为什么是"重装底盘"？

## 为什么会有这个问题

ZeRO 和 FSDP 适合中小规模的分布式训练。但到了**百亿、千亿参数模型的预训练**场景，框架的稳定性、并行能力和通信效率变得至关重要，需要一套"工业级"的解决方案。

## 解决什么问题

Megatron-LM 是 NVIDIA 推出的分布式训练框架。它把 GPU 优化的并行内核抽出来形成了 **Megatron-Core**（同仓库的 `megatron/core/`），二者**并行存在、不是"演进替代"关系**：Megatron-LM 仍是参考训练框架与示例脚本，Megatron-Core 是可复用的并行库；NeMo 构建在 Core 之上，另有 **Megatron-Bridge** 负责 HF ↔ Megatron 的权重互转。它的定位非常明确：**为超大模型预训练提供最高效、最稳定的并行方案。**

### 核心能力

| 能力 | Megatron-Core 支持 | 对比 FSDP2 | 对比 DeepSpeed |
|------|------------------|-----------|---------------|
| 张量并行（TP） | ✅ **原生，最成熟** | ⚠️ 实验性 | ⚠️ 需 Megatron 辅助 |
| 流水线并行（PP） | ✅ **1F1B + Zero Bubble** | ❌ | ✅ PipeDream |
| 序列并行（SP） | ✅ **Megatron-SP + Context Parallel** | ❌ | ✅ Ulysses |
| 专家并行（EP） | ✅ **MoE Parallel Folding** | ❌ | ✅ DeepSpeed-MoE |
| FP8 训练 | ✅ **Transformer Engine 原生集成** | ⚠️ 依赖 torch.compile | ⚠️ 实验性 |
| 5D 并行 | ✅ TP+PP+DP+CP+EP | ❌ | ✅ ZeRO+MoE |
| NeMo 封装 | ✅ 端到端框架 | — | — |

### Megatron 的 TP 为什么是"工业标准"

Shoeybi 等人在 2019 年提出的 Megatron-LM 张量并行方案是所有 TP 实现的**参考实现**：

```
列切分 + 行切分的配对设计（MLP 中 W1 列切分、W2 行切分）：
  MLP 前向：
    [列切分 W1，输入是复制态，前向**不做任何通信**] → GeLU → [行切分 W2 → 一次 AllReduce 输出]
  MLP 反向（输入梯度需要聚合，通信出现在这里）：
    [W2 反向 → 一次 AllReduce 聚合输入梯度] → [W1 反向，输出梯度是复制态，无通信]

即：前向只在行并行层做 1 次 AllReduce，反向也只在对应位置做 1 次 AllReduce

这个看似微小的优化，在 80 层 Transformer 上 = 前向每层少 1 次 All-Reduce，80 层累计约 80 次（反向同理）
```

---

## 2026 年的 Megatron-Core

### 支持的模型规模

| 模型规模 | 并行配置 | GPU 数 |
|---------|---------|--------|
| 70B | TP=8, PP=4, DP=4 | 128 |
| 405B | TP=8, PP=8, DP=8 | 512 |
| 1T+ | TP=8, PP=8, DP=16, CP=4 | 1024+ |

**可核实的 MFU 参考**：MoE Parallel Folding（arXiv 2504.14960）报告 Mixtral 8x22B 在 H100 上达 49.3% MFU、Qwen2-57B-A14B 达 39.0% MFU。其余规模/配置的具体 MFU 随批次、并行度与内核版本差异很大，不宜直接引用。

### 2025-2026 年关键更新

1. **Zero Bubble Pipeline（2024 集成）**：ZB1P 把气泡降到 1F1B 的 1/3（**不是 0**）；ZB2P/ZBV 才能做到 ~0，代价是 2× 激活或 2× 流水线通信
2. **MoE Parallel Folding（2025.04）**：arXiv 2504.14960，支持 TP/EP/DP/CP 折叠，Mixtral 8x22B 上 MFU 达 49.3%
3. **Scalable Training of MoE Models with Megatron Core（2026.03）**：arXiv 2603.07685，MoE 训练的系统级优化
4. **NVFP4 预训练（2026）**：4-bit 精度训练，吞吐提升 1.3-1.7×
5. **NeMo 2.0**：端到端框架，降低 Megatron 的使用门槛
6. **Megatron-Bridge**：HF ↔ Megatron 权重互转 + 现成 recipe，是 2026 年降低 Megatron 上手门槛的关键组件

---

## 优缺点

| 优点 | 缺点 |
|------|------|
| ✅ **MFU 最高**：适配 NVIDIA 硬件到极致 | ❌ **学习曲线极陡**：代码规范性要求高 |
| ✅ **工业级稳定**：万卡训练几乎不崩溃 | ❌ **强绑 NVIDIA**：无法在 AMD/华为上跑 |
| ✅ **NVIDIA 全栈优化**：Transformer Engine + NCCL + NVLink | ❌ **代码侵入性高**：模型需按 Megatron 规范重写 |
| ✅ **生态最完整**：NeMo + Triton + TensorRT | ❌ **小模型杀鸡用牛刀**：7B 模型没必要用 |
| ✅ **FP4/FP8 硬件级支持** | ❌ **配置复杂**：并行策略需手动仔细调整 |

---

## 🛠 实操指南

### 📋 决策指南：你需要 Megatron-Core 吗？

```
模型规模 > 70B？
├─ 否 → FSDP2 就够了（更简单）
└─ 是 →
   团队有分布式训练经验？
   ├─ 否 → 先试 NeMo（Megatron 的封装版本）
   └─ 是 → Megatron-Core（最灵活）

计算资源 > 128 GPU？
├─ 否 → DeepSpeed ZeRO-3 可能更简单
└─ 是 → Megatron-Core 更稳定高效
```

### 🔧 快速开始 Megatron-Core

```bash
# 1. 安装（PyPI 包名是 megatron-core；完整参考实现从 GitHub 仓库获取）
pip install megatron-core

# 2. 启动训练：Megatron-LM/Core 的入口是 CLI flag（不读 YAML），
#    并行配置直接以命令行参数传入
#    TP=8 × PP=4 = 32 卡，--nproc_per_node 需与总卡数一致
torchrun --nproc_per_node=32 \
  pretrain_gpt.py \
  --tensor-model-parallel-size 8 \
  --pipeline-model-parallel-size 4 \
  --context-parallel-size 1 \
  --expert-model-parallel-size 1 \
  --num-layers 80 \
  --hidden-size 8192 \
  --num-attention-heads 64 \
  --seq-length 4096 \
  --use-flash-attn \
  --fp8-format e4m3
```

### 💡 一句话记住

> **"Megatron-Core 是超大模型训练的'重装底盘'——它支持最全面的 5D 并行策略（TP+PP+DP+CP+EP）、与 NVIDIA 硬件深度绑定。代价是学习曲线陡峭、代码侵入性强。2026 年，70B+ 模型的预训练，它是事实上的工业标准。"**

---

## 对后续的铺垫

Megatron-Core 是"重装"，FSDP2 就是"轻骑"。对于大多数微调场景，FSDP2 是 2026 年的默认选择。

→ [[04-FSDP2为什么成为微调的默认选择]]

---

> **参考资料**：
> - [Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism" (2019)](https://arxiv.org/abs/1909.08053)
> - [NVIDIA Megatron-Core 官方文档](https://docs.nvidia.com/megatron-core/)
> - [Megatron-Core Parallelism Guide（并行指南）](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)
> - [Qi et al., "Zero Bubble Pipeline Parallelism" (2024)](https://arxiv.org/abs/2401.10241)
> - [NVIDIA, "MoE Parallel Folding: Heterogeneous Parallelism Mappings for Efficient Large-Scale MoE Model Training with Megatron Core" (2025)](https://arxiv.org/abs/2504.14960)
> - [NVIDIA Megatron-Core, "Scalable Training of Mixture-of-Experts Models with Megatron Core" (2026)](https://arxiv.org/abs/2603.07685)
> - [NVIDIA NeMo 框架文档](https://docs.nvidia.com/nemo-framework/)
> - [Megatron-Bridge（HF ↔ Megatron 权重互转）](https://github.com/NVIDIA-NeMo/Megatron-Bridge)
