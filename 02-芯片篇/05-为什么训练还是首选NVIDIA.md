# Q: 为什么 AI 训练还是首选 NVIDIA？AMD 和华为不行吗？

## 为什么会有这个问题

你可能会想：既然 GPU 架构本质是大量 ALU 并行，那 AMD MI300X（比 H100 显存更大、价格更低）、华为昇腾 910B、Google TPU 等也都是 GPU 或加速器，为什么主流大模型训练还是用 NVIDIA？

## 解决什么问题：CUDA 生态壁垒

答案不是"NVIDIA 硬件最快"，而是它的软件生态实在太强。

### 核心：CUDA + cuDNN + NCCL

NVIDIA 构建了三层软件栈，每一层都是实质性的竞争壁垒：

```
上层：训练框架
  PyTorch / TensorFlow / JAX / Megatron-LM / DeepSpeed
      ↓  （全部基于 CUDA）
中层：NVIDIA 加速库
  cuDNN（神经网络算子）  cuBLAS（矩阵运算）  NCCL（通信）
      ↓
底层：CUDA 编程模型
  驱动 → 编译器 → Runtime → 工具链
```

所有主流框架（PyTorch、TensorFlow、JAX、DeepSpeed、Megatron-LM）的核心算子都深度依赖 CUDA。这意味着：

1. **跑新模型没有任何适配成本** — 下载即跑
2. **新特性第一时间支持** — FP8 训练、Flash Attention 2/3、MoE 原生支持，都是 NVIDIA 先出
3. **调试工具链成熟** — Nsight Systems、Nsight Compute、DCGM、CUDA-GDB

### 竞争对手的处境

| 厂商 | 硬件 | 软件生态 | 主要障碍 |
|------|------|---------|---------|
| **AMD** | MI300X（192GB HBM3，强于 H100） | ROCm，兼容 CUDA 代码 | PyTorch/Triton 适配仍有 gap；新算子需要手动写 ROCm 实现 |
| **华为** | 昇腾 910B/910C | CANN + MindSpore | PyTorch 适配靠第三方插件；CUDA 兼容层性能损失 10-20% |
| **Google** | TPU v7 Ironwood | JAX/XLA（封闭生态） | 仅供 Google Cloud 内部使用，不对外单独销售 |
| **Intel** | Gaudi 3 | OneAPI + PyTorch Extension | 市场份额极小，社区支持弱 |

---

## AMD 2025-2026 的追赶

AMD 的 MI300X 在硬件上很有竞争力：
- **192 GB HBM3** vs H100 的 80 GB
- 显存带宽 **5.2 TB/s** vs H100 的 3.35 TB/s
- 价格低 30-50%

但根据 2026 年初的测试，跑 Llama 3 70B 微调时，MI300X 集群的 MFU 仍然比 H100 集群低约 15-20%，主要差在：
1. ROCm 对 Flash Attention 的支持不如 CUDA 成熟
2. NCCL 的 AMD 替代（RCCL）在大规模跨机通信时性能不稳定
3. PyTorch 2.x 中部分 CUDA 优化算子无法自动翻译到 ROCm

### 不过有好消息

AMD 的 ROCm 6.0+ 在 2025-2026 年进步很快，PyTorch 官方已原生支持 ROCm。**如果你的训练完全基于 PyTorch（不需要 DeepSpeed ZeRO-3 或 Megatron-LM）**，MI300X 已经是一个可行的替代方案。

---

## 华为昇腾的进展

华为昇腾 910B 在 2025 年底开始批量出货：

| 规格 | 昇腾 910B | 昇腾 910C（2026） |
|------|----------|-----------------|
| FP16 算力 | ~320 TFLOPS | ~400+ TFLOPS |
| 显存 | 64 GB HBM2e | 96 GB HBM3 |
| 互联 | HCCS（华为私有） | HCCS 2.0 |

**最大的挑战**：虽然 PyTorch 通过 torch_npu 插件可以适配昇腾，但 DeepSpeed、Megatron 的很多高级特性（ZeRO-3 Offload、Sequence Parallelism）在昇腾上兼容性差。目前昇腾主要用于"国产化要求"的推理部署场景，训练场景仍然有限。

---

## 🛠 实操指南

### 📋 决策指南：2026 年你该选哪家 GPU？

| 你的场景 | 首选 | 备选 | 理由 |
|---------|------|------|------|
| 预训练 70B+ 模型 | **NVIDIA H100/B200** | 无 | CUDA + NCCL + Megatron 生态，别无选择 |
| 全量微调 13B-34B | **NVIDIA H200** | AMD MI300X | H200 显存 141 GB，生态最成熟 |
| LoRA 微调 70B | **NVIDIA H200/GB300** | 无 | 大显存 + 稳定 CUDA 支持 |
| 推理 70B（高并发） | **NVIDIA B200（FP8）** | AMD MI350 | B200 FP8 推理是 H100 的 4× |
| 推理 70B（成本敏感） | **AMD MI300X** | 华为昇腾 910C | 192 GB 单卡装下 70B，价格低 30-50% |
| 推理（国产化要求） | **华为昇腾 910C** | 寒武纪 | 国产合规，性能够用 |
| 小模型推理（<7B） | **任何 GPU 都行** | CPU 也可 | 负载轻，选便宜的 |
| 研究/实验 | **NVIDIA（CUDA 生态）** | AMD ROCm | 新论文/新模型第一个支持 CUDA |

**关键判断：如果你的工作流涉及 DeepSpeed ZeRO-3、Megatron-LM、或大规模跨机分布式训练，2026 年 NVIDIA 仍是唯一可行选项。AMD 和华为在推理场景已经够用。**

### 🔧 自己动手验：你的代码能不能在非 NVIDIA GPU 上跑？

```bash
# 1. 检查你的 PyTorch 代码是否依赖 CUDA-only 算子
python -c "
import torch

# 检查当前可用的设备
print(f'CUDA 可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU 型号: {torch.cuda.get_device_name(0)}')
    print(f'CUDA 版本: {torch.version.cuda}')
    # 检查是否 NVIDIA
    if 'NVIDIA' in torch.cuda.get_device_name(0):
        print('→ NVIDIA GPU，CUDA 生态完整')
    else:
        print('→ 非 NVIDIA GPU，检查 ROCm/昇腾兼容性')
"

# 2. 检查你的代码库用了哪些 CUDA-specific 功能
# 搜索以下关键词：
grep -r "torch.cuda" --include="*.py" . | head -20
grep -r "amp" --include="*.py" . | head -10
grep -r "FusedAdam\|FlashAttention\|cuBLAS\|cuDNN" --include="*.py" . | head -10

# 如果大量使用 FlashAttention（NVIDIA 独占优化）→ 迁移到 AMD/华为难度大
# 如果只用标准 torch.nn 和 torch.optim → 迁移难度低

# 3. 关键检查：是否用了 NCCL？
grep -r "nccl\|NCCL\|distributed" --include="*.py" . | head -10
# 如果用了 torch.distributed 且指定了 nccl backend →
# 迁移到 AMD 需要改用 gloo backend（性能损失 30-50%）或 RCCL
```

### ⚠️ 常见误区

- **误区**："AMD MI300X 的硬件规格比 H100 强，所以训练更快"
  - **实际**：2025-2026 年第三方实测：MI300X 实际 AI 训练吞吐只有 H100 的 **60-70%**，尽管 FP16 峰值算力高出 32%。原因：① ROCm 编译器和算子库不如 CUDA 成熟，实际利用率仅 45%（H100 可达 93%）；② 跨机通信 RCCL 在大规模下性能不稳定；③ Flash Attention 等关键优化在 ROCm 上支持不全。**硬件规格不等于实际性能。**

- **误区**："昇腾 910C 是国产替代，可以完全替代 H100"
  - **实际**：昇腾 910C 目前的定位是"国产化要求"的**推理部署**场景，大规模预训练场景兼容性还很差。DeepSpeed ZeRO-3、Megatron、Sequence Parallelism 等高级特性需要大量适配工作。**可以替代 H100 做推理，但做大规模训练还要等生态成熟。**

- **误区**："CUDA 是闭源的，迟早会被开源生态取代"
  - **实际**：CUDA 已经积累了 **17 年的开发历史**，超过 500 万开发者和上千个加速库。即使 ROCm 完全兼容 CUDA API，NVIDIA 每代新硬件（FP8 Transformer Engine、NVLink SHARP、GDS）都在增加新的专属优化，竞争对手至少落后 1-2 代。**2026 年 CUDA 的生态优势不是在缩小，而是在扩大。** AMD 证券研究（ainvest.com, 2026.06）数据：NVIDIA 训练市场占有率仍高达 80-92%。

### 📊 底线数字速查

| 维度 | NVIDIA H100 | AMD MI300X | 华为昇腾 910C |
|------|------------|-----------|--------------|
| 训练市场占有率（2026） | **~80-92%** | ~5-7% | 极低（以推理为主）|
| 实际训练吞吐（8×GPU） | **147K tok/s**（基线） | 101K tok/s（-31%） | 无公开可比数据 |
| 显存 | 80 GB HBM3 | 192 GB HBM3 | 96 GB HBM3 |
| 生态成熟度 | ✅ **业界标准** | ⚠️ 进步中，训练仍有 gap | ❌ 推理可用，训练受限 |
| FP8 训练 | ✅ 原生支持 | ❌ 软件模拟 | ❌ 不支持 |
| 开源框架支持 | ✅ 框架原生 | ✅ PyTorch 官方 | ⚠️ 需要 torch_npu 插件 |
| 大规模分布式训练 | ✅ 万卡验证 | ⚠️ 千卡级别 | ❌ 少有大规模训练案例 |
| 价格 | ~$28K | ~$15K（-47%） | ~$10-15K（预估） |
| 最佳场景 | **训练 + 推理** | **推理（成本敏感）** | **推理（国产化）** |

### 💡 一句话记住

> **"2026 年训练还是选 NVIDIA——不是因为它硬件最快，是因为从 PyTorch 到 NCCL 到 Megatron，所有训练工具链都是为它造的。AMD 和华为在推理场景已经够用了，但大规模训练还是 CUDA 的天下。"**

---

## 对后续的铺垫

NVIDIA 的生态优势意味着：**即使 AMD/华为的硬件有竞争力，2026 年大规模分布式训练的首选仍然是 NVIDIA。**

但这不意味着你可以随便用——NVIDIA 自己的产品线也越来越复杂：H100、B200、GB300、Rubin……选哪代？多少显存？什么互联？

→ [[06-Blackwell到Rubin在解决什么问题]]

---

> **参考资料**：
> - [MLPerf Training v4.0 / v5.0 基准对比](https://mlcommons.org)
> - [AMD ROCm 6.0 发布说明](https://rocm.docs.amd.com/en/docs-6.0.0/about/release-notes.html)
> - [华为昇腾产品页面](https://www.huawei.com/)
> - [SemiAnalysis: "NVIDIA vs AMD for AI: The Real Story" (2025)](https://semianalysis.com/)
> - 2026 年各 GPU 厂商 Benchmark 实测对比报告
