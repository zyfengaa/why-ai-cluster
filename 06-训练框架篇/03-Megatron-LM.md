# Q: Megatron-LM 为什么是"重装底盘"？

## 为什么会有这个问题

ZeRO 和 FSDP 适合中小规模的分布式训练。但到了**百亿、千亿参数模型的预训练**场景，框架的稳定性、并行能力和通信效率变得至关重要，需要一套"工业级"的解决方案。

## 解决什么问题

**Megatron-LM** 是 NVIDIA 2019 年（Shoeybi et al.）提出的张量并行训练方案；**Megatron-Core** 是 2023 年从 Megatron-LM 项目中抽出的可复用并行算子与调度库。它的定位非常明确：**为超大模型预训练提供最高效、最稳定的并行方案。**

### 核心能力

| 能力 | Megatron-Core 支持 | 对比 FSDP2 | 对比 DeepSpeed |
|------|------------------|-----------|---------------|
| 张量并行（TP） | ✅ **原生，最成熟** | ⚠️ 实验性 | ⚠️ 需 Megatron 辅助 |
| 流水线并行（PP） | ✅ 1F1B + interleaved 1F1B | ❌ | ✅ PipeDream |
| 序列并行（SP） | ✅ Megatron-SP + Context Parallel | ❌ | ✅ Ulysses |
| 专家并行（EP） | ✅ 原生 MoE 支持 | ❌ | ✅ DeepSpeed-MoE |
| FP8 训练 | ✅ Transformer Engine 原生集成 | ⚠️ 依赖 torch.compile | ⚠️ 实验性 |
| **5D 并行** | ✅ **TP + EP + CP + DP + PP**（见 2025-04 论文） | ❌ | ⚠️ 部分组合 |
| NeMo 封装 | ✅ 端到端框架 | — | — |

**5D 并行的真实含义**：TP（层内张量）+ EP（MoE 专家）+ CP（序列/上下文维度）+ DP（数据）+ PP（流水线），五种并行维度正交组合——例如 TP=8 × EP=8 × CP=2 × DP=4 × PP=4 组成 4,096 卡的作业。这是 Megatron-Core 论文（2025-04，arXiv:2504.14960）里给出的官方口径，与本书第 5、6 篇的用法一致。

### Megatron 的 TP 为什么是"工业标准"

Shoeybi 等人在 2019 年论文中提出的张量并行方案是所有 TP 实现的**参考实现**。核心是列切分（column-parallel）+ 行切分（row-parallel）的配对设计，以及两个共轭算子 f 和 g：

```
MLP 前向（W1 列切分、W2 行切分）：
  输入 X 在 TP 组内是复制的（不需要 All-Gather）
  → f：恒等前向 / 反向 All-Reduce
  → 列切分 W1：每卡算 Y = X·W1_i（输出按列分片，保持分片状态直接喂给下一层）
  → GeLU（每卡独立，无需通信）
  → 行切分 W2：每卡算 Z_i = Y_i·W2_i
  → g：前向 All-Reduce 求和 / 反向恒等
  → 输出 Z 在组内复制

一层 Transformer 前向共 2 次 All-Reduce（attention 输出后 1 次 + MLP 输出后 1 次），
反向同样 2 次。没有 All-Gather——列并行的输出保持分片，行并行的输出做 All-Reduce。
```

每个 All-Reduce 的通信量约 `B×S×H`（两个分片结果相加），必须在 NVLink 域内做——这就是 TP 度通常 ≤ 8（单机 8 卡）的原因。

### 关键更新（时间线核对）

1. **Megatron-Core（2023）**：从 Megatron-LM 抽出的并行库，独立发版
2. **Zero Bubble Pipeline（2024，研究实现）**：PP 气泡进一步压缩。注意它**不是 Megatron-LM 上游内置能力**——来自学术界工作（如 SAIL 等团队/分支的实现），MindSpeed 等第三方做了工程化；上游 Megatron-LM 目前提供的是 1F1B 和 interleaved 1F1B 调度
3. **MoE Parallel Folding（2025-04，arXiv:2504.14960）**：5D 并行下 MoE 训练的异构并行映射，在 **128×H100** 上测得 MoE 模型 MFU 达 **49.3%**（注意是 128 卡规模，不是上千卡）
4. **Transformer Engine FP8/FP4**：与 NVIDIA 硬件逐代联动，FP8 在 Hopper、FP4 在 Blackwell 上原生支持

### 支持的模型规模（公开口径）

Megatron-Core 是 Llama 3 等公开大模型训练的参考实现之一。公开报告的 MFU 大致在 **38–43%**（如 Meta Llama 3 405B）；不同模型/配置差异很大，不要把任何单一数字当成"Megatron 的 MFU"直接套用。

---

## 优缺点

| 优点 | 缺点 |
|------|------|
| ✅ **MFU 高**：适配 NVIDIA 硬件到极致 | ❌ **学习曲线极陡**：代码规范性要求高 |
| ✅ **工业级稳定**：万卡训练经过验证 | ❌ **强绑 NVIDIA**：跨平台要靠 MindSpeed 等移植 |
| ✅ **NVIDIA 全栈优化**：Transformer Engine + NCCL + NVLink | ❌ **代码侵入性高**：模型需按 Megatron 规范重写 |
| ✅ **并行维度最全**：TP/PP/CP/EP/DP 5D 组合 | ❌ **小模型杀鸡用牛刀**：7B 以下没必要用 |
| ✅ **MoE/长序列原生支持** | ❌ **Zero Bubble 等新调度不在上游**：需分支或第三方实现 |

---

## 🛠 实操指南

### 📋 决策指南：你需要 Megatron-Core 吗？

```
模型规模 > 70B？
├─ 否 → FSDP2/ZeRO 就够了（更简单）
└─ 是 →
   团队有分布式训练经验？
   ├─ 否 → 先试 NeMo / Megatron-Bridge（Megatron 的封装）
   └─ 是 → Megatron-Core（最灵活）

计算资源 > 128 GPU？
├─ 否 → DeepSpeed ZeRO-3 可能更简单
└─ 是 → Megatron-Core 更稳定高效
```

### 🔧 安装与启动（命令以官方 README 为准）

```bash
# 1. 安装：PyPI 包名是 megatron-core（不是 nvidia-megatron-core）
pip install megatron-core
# 或从 GitHub 源码安装：
#   git clone https://github.com/NVIDIA/Megatron-LM.git
#   pip install -e .

# 2. 启动训练：真实入口是 pretrain_gpt.py（不存在 pretrain_llama.py，也没有 --config YAML 入口）。
#    并行度全部用命令行参数指定。以下以 Megatron-LM 主线版本为例，
#    flag 名可能随版本变化，请以 `pretrain_gpt.py --help` 与官方 README 核对：
torchrun --nproc_per_node=8 pretrain_gpt.py \
  --tensor-model-parallel-size 8 \
  --pipeline-model-parallel-size 4 \
  --num-layers 80 \
  --hidden-size 8192 \
  --num-attention-heads 64 \
  --seq-length 4096 \
  --micro-batch-size 1 \
  --global-batch-size 256 \
  --use-flash-attn \
  --fp8-format hybrid        # FP8 走 Transformer Engine；也有 --fp8-param-gather
```

说明：`--fp8-format` 的可选值在当前主线是 `e4m3` 和 `hybrid`（前向 E4M3、反向 E5M2）；FP8 需 Ada（算力 8.9）/Hopper（9.0）及以上硬件并安装 Transformer Engine。

### 💡 一句话记住

> **"Megatron-Core 是超大模型训练的'重装底盘'——2019 年 Megatron 论文确立了'每层 2 次前向 All-Reduce'的张量并行范式，2023 年抽成 Megatron-Core，如今支持 TP+EP+CP+DP+PP 的 5D 并行组合。代价是学习曲线陡峭、代码侵入性强、深绑 NVIDIA。千亿级以上预训练，它是事实上的工业标准。"**

---

## 对后续的铺垫

Megatron-Core 是"重装"，FSDP2 就是"轻骑"。对于大多数微调场景，PyTorch 原生方案更轻、更快上手，也在逐步成为默认选择。

→ [[04-FSDP2为什么成为微调的默认选择]]

---

> **参考资料**：
> - [Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism" (2019)](https://arxiv.org/abs/1909.08053)
> - [NVIDIA/Megatron-LM GitHub（含 Megatron-Core）](https://github.com/NVIDIA/Megatron-LM)
> - ["MoE Parallel Folding: Heterogeneous Parallelism Mappings for Efficient Large-Scale MoE Model Training with Megatron Core" (2025)](https://arxiv.org/abs/2504.14960)
> - [Megatron-Core 官方开发者文档](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/index.html)
> - [NVIDIA NeMo 框架文档](https://docs.nvidia.com/nemo-framework/)
> - [PyPI: megatron-core](https://pypi.org/project/megatron-core/)
