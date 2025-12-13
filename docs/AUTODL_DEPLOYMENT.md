# AutoDL 云平台部署指南

## 为什么选择AutoDL

- ✅ **便宜**：RTX 3090约¥2/小时，比AWS/阿里云便宜70%
- ✅ **快速**：预装PyTorch、CUDA，5分钟启动
- ✅ **灵活**：按需付费，无需包月
- ✅ **大陆访问快**：国内服务器，HuggingFace模型下载快

## 快速开始（5分钟）

### Step 1: 注册AutoDL
1. 访问 https://www.autodl.com/
2. 注册账号并充值（建议50元起）
3. 实名认证（训练需要）

### Step 2: 创建实例

**推荐配置**（Week 1-2训练）：
```
GPU: RTX 3090 (24GB) - ¥1.99/小时
CPU: 8核
内存: 30GB
硬盘: 50GB（系统盘）+ 数据盘按需
镜像: PyTorch 2.1.0 / Python 3.10 / CUDA 12.1
```

**创建步骤**：
1. 控制台 → 容器实例 → 租用新实例
2. 选择"RTX 3090"
3. 镜像选择"PyTorch" → "2.1.0-py3.10-cuda12.1-cudnn8-devel"
4. 点击"立即创建"

### Step 3: 连接实例

**方式1：Web终端**（最简单）
```bash
# 点击"JupyterLab"即可使用
```

**方式2：SSH连接**（推荐）
```bash
# AutoDL提供的SSH命令，如：
ssh -p 12345 root@region-1.autodl.com
# 密码在控制台显示
```

**方式3：VSCode Remote SSH**（最佳开发体验）
```
1. VSCode安装"Remote - SSH"插件
2. 配置SSH：
   Host autodl
     HostName region-1.autodl.com
     User root
     Port 12345
3. 连接后可以远程编辑代码
```

---

## 部署OpenMind项目

### 1. 上传代码

**方式A：Git克隆**（推荐）
```bash
cd /root/autodl-tmp
git clone https://github.com/Libres-coder/OpenMind.git
cd OpenMind
```

**方式B：文件上传**
```bash
# AutoDL控制台 → 文件管理 → 上传
# 或使用JupyterLab上传整个文件夹
```

### 2. 安装依赖

```bash
# 进入项目目录
cd /root/autodl-tmp/OpenMind

# 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 如果没有requirements.txt，手动安装：
pip install transformers datasets accelerate pyyaml pillow tqdm -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. 测试环境

```bash
# 测试GPU
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# 应该输出：
# True
# NVIDIA GeForce RTX 3090

# 测试改进的视觉编码器
python src/improved_vision_encoder.py
```

### 4. 运行训练

```bash
# Week 1训练（完整版本）
python scripts/train_week1.py

# 预期：
# - 自动下载Qwen2-0.5B（约1GB）
# - 训练速度：约20-30秒/epoch（比CPU快100倍）
# - 每个epoch约5-10分钟
```

---

## 成本估算

### Week 1-2 训练（基础验证）
```
GPU: RTX 3090
时长: 3小时
成本: 3h × ¥2/h = ¥6
```

### Week 3-6 训练（长文本+推理）
```
GPU: RTX 4090 或 A100
时长: 20小时
成本: 20h × ¥3-5/h = ¥60-100
```

### 完整项目（Week 1-12）
```
预计总成本: ¥200-500
（远低于购买GPU或租云服务器）
```

---

## 优化技巧

### 1. 节省费用
```bash
# 训练时开机，不训练时关机
# AutoDL按秒计费，暂停后不收费

# 控制台 → 关机（数据保留）
# 需要时再开机继续
```

### 2. 数据持久化
```bash
# AutoDL的/root/autodl-tmp目录数据持久保存
# 建议：
/root/autodl-tmp/OpenMind        # 代码
/root/autodl-tmp/data            # 数据集
/root/autodl-tmp/outputs         # 模型checkpoint
```

### 3. 下载HuggingFace模型
```bash
# AutoDL已配置镜像，下载很快
# 如果需要手动设置：
export HF_ENDPOINT=https://hf-mirror.com
```

### 4. 使用学术加速
```bash
# 免费加速HuggingFace、GitHub下载
pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
```

---

## 常见问题

### Q1: 如何上传本地checkpoint？
```bash
# 方式1：JupyterLab文件上传
# 方式2：scp命令
scp -P 12345 -r outputs/ root@region-1.autodl.com:/root/autodl-tmp/OpenMind/
```

### Q2: 训练中断怎么办？
```bash
# 我们的ProductionTrainer有自动checkpoint
# 重新运行时会从最新checkpoint恢复
```

### Q3: 显存不足（OOM）？
```bash
# 减小batch_size
# configs/week1_training.yaml:
batch_size: 1  # 从2改为1
gradient_accumulation: 16  # 从8改为16
```

### Q4: 如何查看训练日志？
```bash
# 实时查看
tail -f outputs/week1/training.log

# 查看训练曲线
cat outputs/week1/training_log.json
```

---

## AutoDL vs 本地训练对比

| 项目 | 本地CPU | AutoDL RTX 3090 |
|------|---------|-----------------|
| 速度 | 1× | **100×** |
| 内存限制 | ❌ 经常OOM | ✅ 24GB VRAM |
| 每epoch时间 | 60分钟 | **30秒** |
| Week 1训练时间 | 3小时 | **2分钟** |
| 成本 | 免费但浪费时间 | ¥6（3小时） |

---

## 立即开始

1. **注册AutoDL**：https://www.autodl.com/register
2. **充值**：建议50元
3. **创建RTX 3090实例**
4. **上传代码**：
   ```bash
   git clone https://github.com/Libres-coder/OpenMind.git
   cd OpenMind
   pip install -r requirements.txt
   ```
5. **运行训练**：
   ```bash
   python scripts/train_week1.py
   ```

**预期结果**：
- ✅ 3个epoch在2分钟内完成
- ✅ Loss正常下降
- ✅ 模型保存在`outputs/week1/`

---

## 下一步

训练完成后：
1. 下载checkpoint到本地
2. 进行本地推理测试
3. 继续Week 2-3的长文本能力开发

有问题随时问我！
