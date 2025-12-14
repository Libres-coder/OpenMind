# 多模态智能模型训练指南 🚀

本项目提供了一套完整的多模态智能模型训练框架，支持视觉、音频、文本等多种模态的融合处理，并具备强大的推理能力。

## 📋 目录

- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [环境配置](#环境配置)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型评估](#模型评估)
- [模型部署](#模型部署)
- [常见问题](#常见问题)

---

## 🚀 快速开始

### 1. 环境安装

```bash
# 克隆项目
git clone https://github.com/your-org/OpenMind.git
cd OpenMind

# 运行环境配置脚本
bash scripts/setup_environment.sh

# 激活虚拟环境
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\activate   # Windows
```

### 2. 准备数据

```bash
# 创建数据目录
mkdir -p data/{pretrain,sft,eval}

# 下载示例数据（需要配置数据源）
python scripts/download_datasets.py --config configs/data_config.yaml
```

### 3. 开始训练

```bash
# 阶段1: 多模态预训练
python src/train_multimodal.py \
    --config configs/training_config.yaml \
    --stage pretrain \
    --num_epochs 10

# 阶段2: 指令微调
python src/train_multimodal.py \
    --config configs/training_config.yaml \
    --stage sft \
    --num_epochs 5
```

---

## 📁 项目结构

```
OpenMind/
├── configs/                      # 配置文件目录
│   ├── training_config.yaml      # 训练配置
│   ├── model_config.yaml         # 模型配置
│   └── data_config.yaml          # 数据配置
│
├── data/                         # 数据目录
│   ├── pretrain/                 # 预训练数据
│   ├── sft/                      # 指令微调数据
│   └── eval/                     # 评估数据
│
├── src/                          # 源代码目录
│   ├── model_architecture.py     # 模型架构
│   ├── data_pipeline.py          # 数据处理
│   ├── train_multimodal.py       # 训练脚本
│   ├── evaluate_model.py         # 评估脚本
│   └── inference.py              # 推理脚本
│
├── scripts/                      # 工具脚本
│   ├── setup_environment.sh      # 环境配置
│   ├── download_datasets.py      # 数据下载
│   └── convert_checkpoint.py     # 模型转换
│
├── checkpoints/                  # 模型检查点
├── logs/                         # 训练日志
├── outputs/                      # 输出结果
│
├── docs/                         # 文档目录
│   └── MULTIMODAL_MODEL_TRAINING_PLAN.md
│
└── README_TRAINING.md            # 本文件
```

---

## ⚙️ 环境配置

### 硬件要求

#### 最低配置
- GPU: 1x RTX 4090 (24GB) 或 A6000 (48GB)
- RAM: 32GB
- 存储: 500GB SSD

#### 推荐配置
- GPU: 4x A100 (80GB) 或 8x H100 (80GB)
- RAM: 256GB
- 存储: 2TB NVMe SSD

#### 云服务推荐
- AWS: `p4d.24xlarge` (8x A100 80GB)
- 阿里云: `ecs.gn7i-c64g1.24xlarge` (8x A100 80GB)
- 腾讯云: `GT4.20XLARGE464` (8x A100 80GB)

### 软件要求

- Python 3.9+
- CUDA 11.8+ / 12.1+
- PyTorch 2.1+
- Transformers 4.36+

---

## 📦 数据准备

### 数据格式

#### 预训练数据格式 (JSONL)

```json
{
  "text": "图片描述文本",
  "image": "/path/to/image.jpg",
  "metadata": {
    "source": "laion",
    "quality_score": 0.85
  }
}
```

#### 指令微调数据格式

```json
{
  "conversations": [
    {"from": "user", "value": "这张图片中有什么？"},
    {"from": "assistant", "value": "图片中显示了一只可爱的猫咪..."}
  ],
  "image": "/path/to/image.jpg"
}
```

### 推荐数据集

#### 图像-文本数据
- LAION-5B: 50亿图文对
- CC12M: 1200万高质量图文对
- COCO: 33万标注图像
- Visual Genome: 10.8万图像关系数据

#### 视频数据
- WebVid: 1000万视频片段
- HowTo100M: 1.36亿视频片段

#### 文档理解
- DocVQA: 5万文档问答
- ChartQA: 3.2万图表问答
- TextVQA: 4.5万文本问答

#### 推理数据
- GSM8K: 8000+数学问题
- MATH: 12000高级数学题
- HumanEval: 164代码问题

### 数据下载脚本

```bash
# 下载LAION数据子集
python scripts/download_datasets.py \
    --dataset laion \
    --subset 2B \
    --output data/pretrain/laion

# 下载COCO数据
python scripts/download_datasets.py \
    --dataset coco \
    --year 2017 \
    --output data/pretrain/coco
```

---

## 🎓 模型训练

### 三阶段训练策略

#### 阶段1: 多模态预训练 (40-60天)

**目标**: 建立跨模态对齐能力

```bash
python src/train_multimodal.py \
    --config configs/training_config.yaml \
    --stage pretrain \
    --num_epochs 10 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --output_dir checkpoints/pretrain
```

**关键参数**:
- `batch_size`: 每GPU批次大小
- `gradient_accumulation_steps`: 梯度累积步数
- `learning_rate`: 学习率 (推荐 1e-5 到 5e-5)

#### 阶段2: 指令微调 (15-30天)

**目标**: 提升任务遵循和对话能力

```bash
python src/train_multimodal.py \
    --config configs/training_config.yaml \
    --stage sft \
    --num_epochs 5 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-6 \
    --output_dir checkpoints/sft
```

#### 阶段3: 强化学习 (20-40天，可选)

**目标**: 激发推理能力，对齐人类偏好

```bash
python src/train_rlhf.py \
    --config configs/rl_config.yaml \
    --base_model checkpoints/sft/best_model \
    --reward_model checkpoints/reward_model \
    --num_epochs 3 \
    --output_dir checkpoints/rl
```

### 分布式训练

#### 使用 DeepSpeed

```bash
deepspeed --num_gpus=8 src/train_multimodal.py \
    --config configs/training_config.yaml \
    --deepspeed configs/deepspeed_config.json
```

#### 使用 Accelerate

```bash
accelerate launch --multi_gpu --num_processes=8 \
    src/train_multimodal.py \
    --config configs/training_config.yaml
```

### 训练监控

#### 使用 Wandb

```bash
# 在配置文件中启用wandb
use_wandb: true
project_name: "multimodal-training"

# 或命令行启用
python src/train_multimodal.py \
    --config configs/training_config.yaml \
    --use_wandb \
    --wandb_project "multimodal-training"
```

#### 使用 TensorBoard

```bash
# 启动TensorBoard
tensorboard --logdir logs/

# 浏览器访问: http://localhost:6006
```

---

## 📊 模型评估

### 运行评估

```bash
python src/evaluate_model.py \
    --model_config configs/model_config.yaml \
    --checkpoint checkpoints/sft/checkpoint-epoch-5/model.pt \
    --eval_config configs/eval_config.yaml
```

### 评估指标

#### 视觉理解
- **VQA准确率**: Visual Question Answering
- **COCO CIDEr**: 图像描述质量
- **TextVQA准确率**: 文本识别问答

#### 推理能力
- **GSM8K准确率**: 数学推理
- **MATH准确率**: 高级数学
- **HumanEval Pass@1**: 代码生成

#### 语言能力
- **Perplexity**: 困惑度
- **BLEU/ROUGE**: 文本生成质量

### 基准测试

```bash
# 运行完整基准测试
bash scripts/run_benchmarks.sh \
    --model checkpoints/sft/best_model \
    --output results/benchmarks.json
```

---

## 🚀 模型部署

### 本地推理

```python
from src.model_architecture import create_model
import torch

# 加载模型
config = {
    'base_model': 'Qwen/Qwen2-7B',
    'vision_model': 'openai/clip-vit-large-patch14',
    # ...其他配置
}
model = create_model(config)
model.load_state_dict(torch.load('checkpoints/best_model.pt'))
model.eval()

# 推理
input_ids = tokenizer("你好，请描述这张图片", return_tensors='pt').input_ids
images = load_image("test.jpg")

with torch.no_grad():
    output = model.generate(
        input_ids=input_ids,
        images=images,
        max_length=512
    )

print(tokenizer.decode(output[0]))
```

### API 服务部署

```bash
# 使用 FastAPI 部署
python scripts/serve_api.py \
    --model checkpoints/best_model \
    --port 8000 \
    --workers 4
```

### 模型量化

```bash
# INT8量化
python scripts/quantize_model.py \
    --model checkpoints/best_model \
    --bits 8 \
    --output checkpoints/quantized_int8

# INT4量化 (需要bitsandbytes)
python scripts/quantize_model.py \
    --model checkpoints/best_model \
    --bits 4 \
    --output checkpoints/quantized_int4
```

---

## ❓ 常见问题

### Q1: 显存不足怎么办？

**解决方案**:
1. 减小 `batch_size`，增大 `gradient_accumulation_steps`
2. 启用梯度检查点: `use_gradient_checkpointing: true`
3. 使用混合精度训练: `mixed_precision: "bf16"`
4. 使用 LoRA 微调: `use_lora: true`
5. 冻结视觉编码器: `freeze_vision: true`

### Q2: 训练速度太慢？

**解决方案**:
1. 使用 Flash Attention: `use_flash_attention: true`
2. 启用 DeepSpeed ZeRO优化
3. 增加 `num_workers` 提高数据加载速度
4. 使用 WebDataset 格式的数据
5. 启用编译优化: `torch.compile(model)`

### Q3: 如何继续训练？

```bash
python src/train_multimodal.py \
    --config configs/training_config.yaml \
    --resume_from_checkpoint checkpoints/checkpoint-epoch-5
```

### Q4: 如何只微调部分参数？

```yaml
# 在配置文件中启用LoRA
use_lora: true
lora_config:
  r: 8
  lora_alpha: 16
  target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
```

### Q5: 推理时如何切换思考模式？

```python
# 启用推理模式
output = model.generate(
    input_ids=input_ids,
    images=images,
    use_reasoning=True,  # 启用思维链推理
    max_length=1024
)
```

---

## 📚 参考资源

### 论文
- [DeepSeek-V3 Technical Report](https://arxiv.org/pdf/2412.19437)
- [Qwen3 Technical Report](https://github.com/QwenLM/Qwen3)
- [LLaVA: Visual Instruction Tuning](https://arxiv.org/abs/2304.08485)

### 代码库
- [DeepSeek-AI](https://github.com/deepseek-ai)
- [QwenLM](https://github.com/QwenLM)
- [HuggingFace Transformers](https://github.com/huggingface/transformers)

### 社区
- [HuggingFace Forums](https://discuss.huggingface.co/)
- [Discord](https://discord.gg/huggingface)

---

## 📄 许可证

本项目采用 Apache 2.0 许可证。详见 [LICENSE](LICENSE) 文件。

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

**最后更新**: 2025年1月  
**维护者**: OpenMind团队  
**联系方式**: [待补充]
