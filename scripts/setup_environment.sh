#!/bin/bash

echo "=========================================="
echo "Setting up Multimodal Model Training Environment"
echo "=========================================="

# 创建虚拟环境
echo "Creating virtual environment..."
python -m venv venv
source venv/bin/activate

# 升级pip
echo "Upgrading pip..."
pip install --upgrade pip

# 安装PyTorch (根据CUDA版本选择)
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装Transformers和相关库
echo "Installing Transformers and related libraries..."
pip install transformers==4.36.0
pip install accelerate==0.25.0
pip install datasets==2.16.0
pip install sentencepiece==0.1.99
pip install protobuf==3.20.3

# 安装训练工具
echo "Installing training tools..."
pip install deepspeed==0.12.6
pip install flash-attn==2.5.0 --no-build-isolation
pip install xformers==0.0.23

# 安装数据处理工具
echo "Installing data processing tools..."
pip install webdataset==0.2.86
pip install img2dataset==1.45.0
pip install video2dataset==1.2.0
pip install pillow==10.1.0

# 安装音频处理
echo "Installing audio processing libraries..."
pip install torchaudio==2.1.2
pip install librosa==0.10.1
pip install soundfile==0.12.1

# 安装评估工具
echo "Installing evaluation tools..."
pip install scikit-learn==1.3.2
pip install nltk==3.8.1
pip install rouge-score==0.1.2
pip install bert-score==0.3.13

# 安装监控工具
echo "Installing monitoring tools..."
pip install wandb==0.16.1
pip install tensorboard==2.15.1

# 安装其他依赖
echo "Installing other dependencies..."
pip install pyyaml==6.0.1
pip install tqdm==4.66.1
pip install numpy==1.24.3
pip install pandas==2.1.4
pip install requests==2.31.0

# 创建必要的目录
echo "Creating necessary directories..."
mkdir -p data
mkdir -p checkpoints
mkdir -p logs
mkdir -p outputs
mkdir -p configs

# 下载NLTK数据
echo "Downloading NLTK data..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

echo ""
echo "=========================================="
echo "Environment setup completed!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "Next steps:"
echo "1. Prepare your training data in data/ directory"
echo "2. Configure training settings in configs/training_config.yaml"
echo "3. Start training with: python src/train_multimodal.py"
