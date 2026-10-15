# Q: Lustre vs WEKA vs IBM Storage Scale（原 GPFS）怎么选？

## 为什么会有这个问题

并行文件系统不止一种。Lustre、WEKA、IBM Storage Scale（原 GPFS）、BeeGFS……每个有各自的设计哲学和适用场景。选错了，不仅成本高，还可能成为集群的性能陷阱。

## 核心对比

| 特性 | **Lustre** | **IBM Storage Scale（原 GPFS）** | **WEKA** | **BeeGFS** |
|------|-----------|--------------------------|---------|-----------|
| 架构 | 内核模块，**元数据服务器分离** | 内核模块，企业级 | **软件定义**，NVMe 原生 | 用户态，轻量 |
| 元数据 | 支持多 MDT + 条带化目录（Lustre 2.4/2.5 起 DNE，2.8 起 `lfs mkdir -c <count>`，多 MDT 可 active-active 故障切换） | **分布式** | **全分布式** | 分布式 |
| 最大吞吐 | 100s GB/s - 1 TB/s | 100s GB/s | **1+ TB/s**（实测 2+ TB/s） | 500+ GB/s |
| **GPU Direct Storage** | ✅ DDN EXAScaler / AWS FSx for Lustre 官方支持 | ✅ 支持 | ✅ **原生支持** | ✅ 支持（nvidia-fs） |
| 小文件性能 | 差（需打包为 TAR） | 良好 | **优秀**（千万级文件秒级 list） | 良好 |
| 安装难度 | ❌ 困难 | ❌ 困难 | ✅ 简单 | 中等 |
| 许可证 | 开源 (GPL) | 商业（贵） | 商业（中等） | 开源 |
| 典型场景 | HPC/科研，传统超算 | 企业/HPC，金融 | **AI/HPC 前沿** | HPC/科研 |

> 注：GPU Direct Storage 由 NVIDIA 的 `nvidia-fs` 提供，受支持的文件系统除上表外还包括 VAST、DDN EXAScaler 等（以 NVIDIA 官方 nvidia-fs 支持列表为准）。

---

## 深度看 Lustre

**最广泛使用的 HPC 并行文件系统**（TOP100 中 60%+ 的在用；Hyperion Research 2022 年全市场装机份额：Lustre 41%、IBM Spectrum Scale 17%）。

### 部署要点

```
核心组件：
  MGS（管理服务器）+ MDS/MDT（元数据）+ OSS/OST（存储）

关键调优命令：
  # Checkpoint 目录 — 全条带化，最大化写带宽
  lfs setstripe -c -1 -S 4M /checkpoints
  
  # 数据集目录 — 少量条带，减少元数据开销
  lfs setstripe -c 4 -S 1M /datasets
```

### Lustre 的适用边界

| ✅ 适合 | ❌ 不太适合 |
|--------|-----------|
| 有专业 HPC 运维团队 | 没有 HPC 经验的小团队 |
| 大量顺序大文件（100 MB+） | 海量小文件（如大量 JSON） |
| 预算有限（Lustre 免费） | 需要 GPU Direct Storage |
| 128+ 节点集群 | 8-16 节点小规模（杀鸡用牛刀） |

**128 节点规模**需要 128 个 OST（对象存储目标）+ 4 个 MDT（元数据目标，striped），若 OST 为 **NVMe** 介质可达约 480 GB/s 聚合写吞吐（约 3.75 GB/s/OST；HDD 组成的 OST 典型只有 1-1.5 GB/s，128 个仅约 150-190 GB/s）。

**网络要求**：推荐 O2IB/OFED（InfiniBand）或 RoCEv2；Lustre 也能跑 TCP（`socklnd`），但性能明显更差，仅适合小规模或过渡环境。

---

## 深度看 WEKA

**2025-2026 年在 AI 场景增长最快的并行文件系统**。它专为 AI 训练设计，而不是从 HPC 改造来的。

### 架构核心优势

1. **DPDK 内核绕过**：传统 I/O 路径 = 用户态 → 内核态 → NVMe 驱动。WEKA 直接从**用户态访问 NVMe**，延迟 <1 ms。

2. **元数据全分布式**：不需要独立的 MDS 节点，消除了 Lustre 中最大的瓶颈——集中式元数据服务器。

3. **GPU Direct Storage (GDS) 认证**：
   - Checkpoint 写吞吐从 8 GB/s → 20 GB/s（提升 2.5x）
   - CPU 利用率降低 4-6x

4. **内置 S3 自动分层**：热数据留在 NVMe，冷 checkpoint 自动迁移到对象存储。

### 经济性分析（从测试报告中看）

32 GPU B200 集群（数据引自 Spheron 基准）：
- WEKA 额外存储成本：~$2/小时
- GPU 利用率提升：72% → 99%（+27%）
- 省下的 GPU 价值：~$4.93/小时
- **盈亏平衡时间：~41 分钟**（成本 ÷ 净收益 = 2 ÷ (4.93−2) ≈ 0.68 h）

**如果你在意 GPU 利用率，WEKA 几乎肯定值这个钱。**

**但 WEKA 也有明确的适用边界**，别只看优势：
- **成本高于自建 Lustre**：软件许可叠加全闪存介质，百节点以上规模的总拥有成本通常高于用 HDD/混合介质自建 Lustre。
- **强依赖客户端软件栈**：需要在各计算节点安装 WEKA 客户端与 DPDK/网络驱动，内核升级与大版本兼容性要额外验证。
- **并非纯 POSIX 语义**：并行写同一文件、目录语义等在边缘场景与 POSIX 存在取舍，迁移既有应用前需逐项核对。

---

## 实际场景 Benchmark

### Checkpoint 写吞吐（70B 模型，32 GPU B200）

下表沿用 Spheron 基准的 BF16 权重快照（140 GB）；完整可续训状态约 1.12 TB，见 01 篇。

| 存储 | 写吞吐 | Checkpoint 耗时 | GPU 占空比（每 45 分钟一次） | GPU 有效利用 |
|------|--------|----------------|---------------------------|------------|
| NFS | 1.1 GB/s | 127s | 4.7% | 72% |
| BeeOND（临时） | 28 GB/s | 5s | 0.19% | 96% |
| Lustre（32 OST, RoCE） | 38 GB/s | 3.7s | 0.14% | 98% |
| **WEKA（32 GPU）** | **44 GB/s** | **3.2s** | **0.12%** | **99%** |

### 数据加载

| 存储 | 读吞吐 | Dataloader 停顿 |
|------|--------|----------------|
| NFS | 0.9 GB/s | 31% |
| BeeOND | 18 GB/s | 2% |
| Lustre | 26 GB/s | 1.1% |
| **WEKA** | **34 GB/s** | **0.7%** |

---

## 选型决策树

```
你的团队有 HPC 运维经验吗？
├─ 没有 →
│  集群规模 > 32 节点？
│  ├─ 是 → WEKA（买省心）
│  └─ 否 → BeeGFS 或 WEKA
└─ 有 →
   预算充足？
   ├─ 是 → WEKA（性能天花板最高）
   └─ 否 → Lustre（免费但运维成本高）
```

---

## 🛠 实操指南

### 📋 决策指南：你的存储到底该花多少钱？

| 集群规模 | 存储方案 | 额外存储成本/月 | GPU 利用率提升 | 回本周期（估算） |
|---------|--------|---------------|--------------|---------|
| 32 GPU（4 台） | BeeOND（临时） | $0（跑在 GPU 节点本地盘） | 72%→96% | 即时（无额外成本） |
| 32 GPU（4 台） | WEKA | ~$1,440/月 | 72%→99% | **~41 分钟** |
| 128 GPU（16 台） | Lustre（自建） | ~$5,000/月（运维人力，估算） | 75%→97% | ~1 周 |
| 128 GPU（16 台） | WEKA | ~$5,760/月 | 75%→99% | ~3 天 |
| 1,024 GPU（128 台） | WEKA 或 Lustre | ~$46,000/月 | 70%→98% | ~半天 |

**关键结论**：从 NFS 换到并行文件系统的成本，**通常在数小时到数天内被 GPU 利用率提升覆盖**。存储不是成本中心——它是投资回报率最高的优化点。

### 🔧 自己动手验：你的存储够快吗？

```bash
# 1. 检查 Dataloader 停顿率
# 在你的训练日志里搜索类似：
#   "DataLoader time: XX ms"
#   或看 Nsight Systems 里的 CPU-GPU 同步等待

# 如果 Dataloader 停顿 > 5% → 存储是瓶颈

# 2. 用 fio 测存储读吞吐（单节点，多 job、多队列深度）
fio --name=read --rw=read --bs=1M --direct=1 --numjobs=8 --iodepth=32 \
    --filename=/path/to/dataset/file --runtime=30 --time_based --group_reporting
# 单节点期望：> 4.5 GB/s（40GbE 理论上限 5 GB/s，留出协议开销）
# 注：单线程 dd 驱动不到 40G 链路（通常封顶 1-3 GB/s）；GDS 场景请用 gdsio

# 3. 测试多节点并行读（用 mpirun）
# 每个节点同时读取不同的文件，看聚合吞吐
# 128 节点集群期望聚合读吞吐 > 640 GB/s
```

### ⚠️ 常见误区

- **误区**："存储就是硬盘，随便买个 NFS 就好"
  - **实际**：NFS 在 AI 训练中是"省小钱花大钱"的典型。32 GPU 集群用 NFS vs WEKA，GPU 利用率 72% vs 99%，**存储多花的钱约 41 分钟就从净省下的 GPU 时间中收回**。
- **误区**："WEKA 太贵了，用免费的 Lustre 省预算"
  - **实际**：Lustre 免费但运维成本高。你需要：懂的运维工程师（年薪 $20 万+）、元数据 MDS 优化、条带化配置、定期健康检查。**32 节点以下 Lustre 的总成本可能高于 WEKA。**
- **误区**："Checkpoint 就几百 MB，几秒钟就能写完"
  - **实际**：70B 模型完整可续训 Checkpoint 需要：fp16 权重 140 GB + fp16 梯度 140 GB + fp32 master weights 280 GB + Adam 动量/方差 560 GB ≈ **1.12 TB/次**（按 16 bytes/参数口径；NVIDIA 官方按组件拆解口径约 782 GB）。按 NFS 的 1.1 GB/s 写，每次需要约 17 分钟。每天 32 次 Checkpoint = **约 9 小时 GPU 纯闲置**（未考虑异步 checkpoint）。

### 📊 底线数字速查

| 存储系统 | 适合规模 | 聚合写吞吐（32 OST/节点） | 部署复杂度 | 每 GPU 月成本估算 |
|---------|---------|------------------------|-----------|----------------|
| NFS | < 16 GPU | 1-2 GB/s | 极低 | $0（已有基础设施） |
| BeeOND | 8-64 GPU | 28 GB/s（临时） | 低 | $0（用本地盘） |
| Lustre | 64-1,024+ GPU | 38-480 GB/s | ❌ 困难 | ~$40-80（运维分摊） |
| WEKA | 32-1,024+ GPU | 44-2,000+ GB/s | ✅ 简单 | ~$45-90/GPU/月 |
| IBM Storage Scale（原 GPFS） | 企业级 | 100+ GB/s | ❌ 困难 | ~$80-150（许可证贵） |

### 💡 一句话记住

> **"存储是 AI 集群性价比最高的投资——花在存储上的每一块钱，能帮你省下约 2.5 块的 GPU 闲置成本。但前提是别用 NFS。"**

---

## 对后续的铺垫

选好了存储系统，数据还是按"存储 → CPU 内存 → GPU HBM"的传统路径流动。但 GPU Direct Storage 可以直接把这个路径砍掉一半。

→ [[04-GPU-Direct-Storage]]

---

> **参考资料**：
> - [Spheron: "Parallel File Systems for AI on GPU Cloud" (2026.05，厂商博客，数据仅供参考)](https://www.spheron.network/blog/parallel-file-systems-ai-gpu-cloud-wekaio-lustre-beegfs-guide/)
> - [Lustre 官方手册](https://doc.lustre.org/lustre_manual.xhtml)
> - [Lustre Wiki](https://wiki.lustre.org/)
> - [Lustre 发布页（Whamcloud）](https://wiki.whamcloud.com/)
> - [OpenSFS](https://www.opensfs.org/)
> - [WEKA 官方文档](https://docs.weka.io/)
> - [IBM Storage Scale 官方文档](https://www.ibm.com/products/storage-scale)
