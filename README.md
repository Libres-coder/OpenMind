# OpenMind - 自进化多模态大模型

打造具有**持续学习**和**自我迭代**能力的工程级多模态模型，在记忆、推理、视觉理解和多模态处理（特别是视频）方面达到优秀水平。

## 🎯 项目愿景

实现一个真正有意义的AI模型，能够：
- 🧠 **长期记忆**：支持128K-1M上下文
- 🤔 **推理能力**：CoT推理，超越现有开源模型
- 👁️ **视觉理解**：先进的视觉编码器，优秀的图像/视频理解
- 🔄 **自我进化**：自动更新知识库，持续学习，无需人工干预

## 🚀 快速开始

### 本地开发（验证流程）

```bash
# 1. 克隆项目
git clone https://github.com/Libres-coder/OpenMind.git
cd OpenMind

# 2. 测试改进的视觉编码器
python src/improved_vision_encoder.py

# 3. 轻量级训练验证（CPU友好）
python scripts/train_week1_lite.py
```

### AutoDL云平台训练（推荐）

**成本**：RTX 3090约¥2/小时，Week 1训练仅需¥6

```bash
# 1. AutoDL创建实例（RTX 3090）
# 2. 克隆项目
cd /root/autodl-tmp
git clone https://github.com/Libres-coder/OpenMind.git
cd OpenMind

# 3. 快速部署
bash scripts/autodl_setup.sh

# 4. 运行完整训练
python scripts/train_week1.py
```

详细指南：[AutoDL部署文档](docs/AUTODL_DEPLOYMENT.md)

## 📁 项目结构

```
OpenMind/
├── src/                          # 核心代码
│   ├── improved_vision_encoder.py   # 改进的视觉编码器（SigLIP + Flash Attention）
│   ├── model_architecture.py        # 多模态模型架构
│   ├── production_trainer.py        # 生产级训练器
│   └── data_pipeline.py             # 数据处理
├── scripts/                      # 训练和评估脚本
│   ├── train_week1.py              # Week 1 完整训练
│   ├── train_week1_lite.py         # 轻量级训练（CPU）
│   └── autodl_setup.sh             # AutoDL部署脚本
├── configs/                      # 配置文件
│   └── week1_training.yaml         # Week 1 训练配置
├── docs/                         # 文档
│   ├── AUTODL_DEPLOYMENT.md        # AutoDL部署指南
│   ├── SELF_EVOLUTION_SYSTEM.md    # 自进化系统设计
│   └── QUICK_START_GUIDE.md        # 快速开始指南
└── outputs/                      # 训练输出
    └── week1/                      # Week 1 checkpoint
```

## 🎓 核心技术

### Week 1-2：基础架构 ✅
- [x] 改进的视觉编码器（SigLIP + Token Pooling，token减少73%）
- [x] 生产级训练器（梯度监控、Loss检测、自动恢复）
- [x] 训练流程验证（轻量级31.6M模型稳定训练）

### Week 3-4：长文本能力
- [ ] 128K上下文支持
- [ ] 位置编码优化（ALiBi/RoPE）
- [ ] 长文本数据集准备

### Week 5-6：推理能力
- [ ] Chain-of-Thought训练
- [ ] 自我验证机制
- [ ] 推理benchmark评估

### Week 7-12：自进化系统
- [ ] 实时知识更新（联网检索）
- [ ] HuggingFace数据集自动监控
- [ ] 增量训练与防遗忘
- [ ] 自动Prompt优化

## 📊 Week 1 成果

**本地验证**（CPU）：
- ✅ 视觉编码器集成成功
- ✅ 训练器稳定运行3 epochs
- ✅ Loss正常下降（7.08 → 7.02）
- ✅ Checkpoint自动保存

**预期性能**（AutoDL RTX 3090）：
- 训练速度：30秒/epoch（比CPU快100倍）
- 显存占用：<10GB
- 成本：¥6（3小时）

## 🛠️ 开发计划

| Week | 任务 | 状态 |
|------|------|------|
| 1-2 | 核心模型基础 - 集成视觉编码器+训练循环 | ✅ 已完成 |
| 3-4 | 长文本能力 - 128K上下文+测试验证 | 🔄 进行中 |
| 5-6 | 推理能力 - CoT训练+benchmark评估 | 📅 计划中 |
| 7-8 | 知识更新 - 联网获取+向量数据库 | 📅 计划中 |
| 9-10 | HuggingFace集成 - 数据集监控+自动下载 | 📅 计划中 |
| 11-12 | 增量训练 - LoRA微调+防遗忘机制 | 📅 计划中 |

## 💡 特色功能

### 1. 改进的视觉编码器
- SigLIP架构（性能提升5%）
- Token Pooling（减少73% tokens）
- Flash Attention（节省50%显存）

### 2. 生产级训练器
- 梯度异常检测
- Loss突增检测
- 自动checkpoint保存
- 完整训练日志

### 3. 自进化系统（开发中）
- 实时知识库更新
- 自主数据收集
- 增量训练
- 性能监控与回滚

## 📈 性能指标

### 当前（Week 1）
- 模型规模：31.6M参数（轻量级验证）
- 训练速度：2.5秒/step（CPU）
- 内存占用：<4GB

### 目标（Week 12）
- 模型规模：7B-14B参数（MoE架构）
- 长文本：128K-1M上下文
- 推理能力：超越GPT-3.5
- 视频理解：帧级精确定位

## 🔗 相关资源

- [自进化系统设计](docs/SELF_EVOLUTION_SYSTEM.md)
- [AutoDL部署指南](docs/AUTODL_DEPLOYMENT.md)
- [快速开始指南](docs/QUICK_START_GUIDE.md)
- [训练计划详解](DEVELOPMENT_ROADMAP.md)

## 📝 License

MIT License

## 🤝 贡献

欢迎提出Issue和Pull Request！

---

**开始你的AI自进化之旅** 🚀

```bash
# 本地验证
python scripts/train_week1_lite.py

# AutoDL完整训练
python scripts/train_week1.py
```
