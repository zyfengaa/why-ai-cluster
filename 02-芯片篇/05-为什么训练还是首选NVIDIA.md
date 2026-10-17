# Q: 为什么 AI 训练还是首选 NVIDIA？AMD 和华为不行吗？

## 为什么会有这个问题

你可能会想：既然 GPU 架构本质是大量 ALU 并行，那 AMD MI300X（比 H100 显存更大、价格更低）、华为昇腾 910B/910C、Google TPU 等也都是 GPU 或加速器，为什么主流大模型训练还是用 NVIDIA？

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
2. **新特性第一时间支持** — FP8 训练（Transformer Engine）、Flash Attention 2/3、MoE 原生支持，都是 NVIDIA 先出
3. **调试工具链成熟** — Nsight Systems、Nsight Compute、DCGM、CUDA-GDB

### 竞争对手的处境

| 厂商 | 硬件 | 软件生态 | 主要障碍 |
|------|------|---------|---------|
| **AMD** | MI300X（192 GB HBM3，FP8 原生支持） | ROCm | PyTorch/Triton 适配仍有 gap；新算子常需要手动写 ROCm 实现 |
| **华为** | 昇腾 910B/910C | CANN + MindSpore | PyTorch 靠 torch_npu 插件适配；高级分布式特性适配滞后 |
| **Google** | TPU v7 Ironwood | JAX/XLA（封闭生态） | 仅供 Google Cloud 使用，不对外单独销售 |
| **Intel** | Gaudi 3 | OneAPI + PyTorch Extension | 市场份额小，社区支持弱 |

---

## AMD 的追赶

AMD 的 MI300X 在硬件上很有竞争力（AMD 官方规格口径）：
- **192 GB HBM3** vs H100 的 80 GB
- 显存带宽 **约 5.3 TB/s** vs H100 的 3.35 TB/s
- **FP8 原生支持**（CDNA 3 架构）——不是软件模拟

但端到端训练性能上，MI300X 与 H100 的差距通常在 **20-40% 量级**（不同模型、框架、互联下差异很大；微基准可以接近甚至持平，但端到端不等于微基准），主要差在：
1. ROCm 对 Flash Attention 等关键算子的支持不如 CUDA 成熟
2. RCCL（NCCL 的 AMD 等价物）在大规模跨机通信时稳定性/性能仍在改进
3. PyTorch 中部分 CUDA 优化算子无法自动翻译到 ROCm

### 不过有好消息

AMD 的 ROCm 6.x 系列进步很快，PyTorch 官方已原生支持 ROCm。**如果你的训练完全基于 PyTorch（不依赖 DeepSpeed ZeRO-3 或 Megatron-LM 的高级特性）**，MI300X 已经是一个可行的替代方案。

---

## 华为昇腾的进展

昇腾 910B 已在 **2023-2024 年**规模部署，910C 于 **2025 年**开始部署（公开拆解/厂商口径，非官方完整规格）：

| 规格 | 昇腾 910B（2023-2024 部署） | 昇腾 910C（2025 年部署） |
|------|----------|-----------------|
| FP16 算力 | 约 320 TFLOPS（公开拆解口径） | 约 640-800 TFLOPS（公开拆解/厂商口径） |
| 显存 | 64 GB HBM2e | 约 128 GB HBM（公开拆解口径） |
| 互联 | HCCS（华为私有） | HCCS（华为私有） |

**最大的挑战**：虽然 PyTorch 通过 torch_npu 插件可以适配昇腾，但 DeepSpeed、Megatron 的很多高级特性（ZeRO-3 Offload、Sequence Parallelism）在昇腾上需要额外适配。目前昇腾主要用于"国产化要求"的推理部署场景，以及受供应限制下的训练场景。

---

## 🛠 实操指南

### 📋 决策指南：你该选哪家 GPU？

| 你的场景 | 首选 | 备选 | 理由 |
|---------|------|------|------|
| 预训练 70B+ 模型 | **NVIDIA H100/B200** | — | CUDA + NCCL + Megatron 生态最成熟 |
| 全量微调 13B-34B | **NVIDIA H200** | AMD MI300X | H200 显存 141 GB，生态最成熟 |
| LoRA 微调 70B | **NVIDIA H200/GB300** | AMD MI300X | 大显存 + 稳定 CUDA 支持 |
| 推理 70B（高并发） | **NVIDIA B200（FP8）** | AMD MI350 | B200 FP8 推理吞吐明显高于 H100 |
| 推理 70B（成本敏感） | **AMD MI300X** | 华为昇腾 910C | 192 GB 单卡装下 70B，报价通常更低 |
| 推理（国产化要求） | **华为昇腾 910C** | 其他国产 | 国产合规，性能够用 |
| 小模型推理（<7B） | **任何 GPU 都行** | CPU 也可 | 负载轻，选便宜的 |
| 研究/实验 | **NVIDIA（CUDA 生态）** | AMD ROCm | 新论文/新模型第一个支持 CUDA |

**关键判断：如果你的工作流涉及 DeepSpeed ZeRO-3、Megatron-LM、或大规模跨机分布式训练，NVIDIA 仍是生态上最稳妥的选项。AMD 和华为在推理场景已经够用。**

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

# 如果大量使用 FlashAttention（NVIDIA 优化）→ 迁移到 AMD/华为难度大
# 如果只用标准 torch.nn 和 torch.optim → 迁移难度低

# 3. 关键检查：是否用了 NCCL？
grep -r "nccl\|NCCL\|distributed" --include="*.py" . | head -10
# 如果用了 torch.distributed 且指定了 nccl backend →
# 迁移到 AMD 需要改用 gloo backend（性能损失大）或 RCCL
```

### ⚠️ 常见误区

- **误区**："AMD MI300X 的硬件规格比 H100 强，所以训练更快"
  - **实际**：微基准测试可以接近甚至超过 H100，但**端到端训练性能不等于微基准**。MI300X 与 H100 的端到端训练差距常见 20-40%（取决于模型、框架、互联）。真正的原因不在 FP8——MI300X 的 CDNA 3 架构**原生支持 FP8**——而在软件栈（ROCm 算子覆盖、编译器成熟度）和集群互联。**硬件规格不等于实际性能。**

- **误区**："昇腾 910C 是国产替代，可以完全替代 H100"
  - **实际**：不要轻信"完全替代"或"差距只剩 X%"这类无来源结论。客观地说：910C 在国产化推理部署场景已经规模落地，训练场景也在推进，但 DeepSpeed ZeRO-3、Megatron、Sequence Parallelism 等高级特性仍需要大量适配工作，且公开可比的万卡训练案例少于 NVIDIA。**具体能不能替代，取决于你的模型、框架和合规要求，建议先做 PoC。**

- **误区**："CUDA 是闭源的，迟早会被开源生态取代"
  - **实际**：CUDA 已积累了近二十年的开发历史和庞大的开发者/库生态。即使 ROCm 持续改善兼容性，NVIDIA 每代新硬件（FP8 Transformer Engine、NVLink SHARP、GDS）都在增加新的专属优化。**生态差距不是一朝一夕能缩小的。**

### 📊 底线数字速查

| 维度 | NVIDIA H100 | AMD MI300X | 华为昇腾 910C |
|------|------------|-----------|--------------|
| 显存 | 80 GB HBM3 | 192 GB HBM3 | 约 128 GB（公开拆解口径） |
| FP16/BF16 峰值 | 989 TFLOPS 稠密 | 高于 H100（AMD 官方口径） | 约 640-800 TFLOPS（公开拆解/厂商口径） |
| FP8 | ✅ 原生支持 | ✅ **原生支持（CDNA 3）** | 以官方发布为准 |
| 微基准利用率 | 可达 90%+ | 可接近 | — |
| 真实训练 MFU | 通常 30-50%（Llama 3 405B 公开值 38-43%） | 低于 H100 | — |
| 端到端训练差距（vs H100） | — | 常见 20-40%（随模型/框架/互联变化） | 无足够公开可比数据 |
| 生态成熟度 | ✅ **业界标准** | ⚠️ 进步中，训练仍有 gap | ❌ 推理可用，训练适配中 |
| 开源框架支持 | ✅ 框架原生 | ✅ PyTorch 官方 | ⚠️ 需要 torch_npu 插件 |
| 大规模分布式训练 | ✅ 万卡验证 | ⚠️ 千卡级公开案例 | ⚠️ 公开案例有限 |
| 最佳场景 | **训练 + 推理** | **推理（成本敏感）** | **推理（国产化）** |

### 💡 一句话记住

> **"训练还是选 NVIDIA——不是因为它硬件最快，是因为从 PyTorch 到 NCCL 到 Megatron，所有训练工具链都是为它造的。AMD 和华为在推理场景已经够用了，但大规模训练还是 CUDA 的天下。"**

---

## 对后续的铺垫

NVIDIA 的生态优势意味着：**即使 AMD/华为的硬件有竞争力，大规模分布式训练的首选仍然是 NVIDIA。**

但这不意味着你可以随便用——NVIDIA 自己的产品线也越来越复杂：H100、B200、GB300、Rubin……选哪代？多少显存？什么互联？

→ [[06-Blackwell到Rubin在解决什么问题]]

---

> **参考资料**：
> - [AMD Instinct MI300X 官方产品页](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html) — 192 GB HBM3、带宽、CDNA 3 FP8 原生支持
> - [AMD ROCm 文档](https://rocm.docs.amd.com/) / [ROCm 发布说明](https://rocm.docs.amd.com/en/latest/about/release-notes.html)
> - [MLCommons MLPerf 结果（NVIDIA 开发者博客聚合）](https://developer.nvidia.com/blog/tag/mlperf/) — 端到端训练基准
> - [华为昇腾产品页](https://e.huawei.com/en/products/computing/ascend) / [昇腾社区](https://www.hiascend.com/en/) — 昇腾处理器与 Atlas 系列官方资料
> - [Llama 3 论文 (arXiv:2407.21783)](https://arxiv.org/abs/2407.21783) — 405B 训练 MFU 38-43% 等公开数据
