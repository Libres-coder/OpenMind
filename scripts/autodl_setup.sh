#!/bin/bash
# AutoDL 快速部署脚本

echo "=========================================="
echo "OpenMind AutoDL 部署脚本"
echo "=========================================="

# 1. 检查环境
echo -e "\n[1/5] 检查GPU环境..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# 2. 安装依赖（使用清华镜像加速）
echo -e "\n[2/5] 安装依赖..."
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 3. 测试改进的视觉编码器
echo -e "\n[3/5] 测试视觉编码器..."
python src/improved_vision_encoder.py

# 4. 测试模型集成
echo -e "\n[4/5] 测试模型集成..."
python scripts/test_model_integration.py

# 5. 创建必要目录
echo -e "\n[5/5] 创建输出目录..."
mkdir -p outputs/week1
mkdir -p data/images

echo -e "\n=========================================="
echo "✅ 环境配置完成！"
echo "=========================================="
echo ""
echo "现在可以运行训练："
echo "  python scripts/train_week1.py"
echo ""
echo "或运行轻量级测试："
echo "  python scripts/train_week1_lite.py"
echo ""
