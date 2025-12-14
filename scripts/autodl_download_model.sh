#!/bin/bash
# AutoDL模型下载脚本 - 使用国内镜像

echo "============================================"
echo "配置Hugging Face镜像并下载模型"
echo "============================================"

# 设置环境变量使用镜像
export HF_ENDPOINT=https://hf-mirror.com

# 创建模型目录
mkdir -p /root/autodl-tmp/models

# 使用Python下载模型
python << EOF
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from transformers import AutoModel, AutoTokenizer

print("开始下载Qwen2-0.5B模型...")
try:
    # 下载并保存到本地
    model = AutoModel.from_pretrained("Qwen/Qwen2-0.5B", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B", trust_remote_code=True)
    
    # 保存到本地目录
    model.save_pretrained("/root/autodl-tmp/models/Qwen2-0.5B")
    tokenizer.save_pretrained("/root/autodl-tmp/models/Qwen2-0.5B")
    
    print("✅ 模型下载成功！保存在 /root/autodl-tmp/models/Qwen2-0.5B")
except Exception as e:
    print(f"❌ 下载失败: {e}")
    print("尝试使用modelscope...")
    
    # 备用方案：使用modelscope
    from modelscope import snapshot_download
    model_dir = snapshot_download('qwen/Qwen2-0.5B', cache_dir='/root/autodl-tmp/models')
    print(f"✅ 模型下载成功！保存在 {model_dir}")
EOF

echo "============================================"
echo "下载完成！"
echo "============================================"
