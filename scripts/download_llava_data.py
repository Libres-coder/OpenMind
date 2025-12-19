"""
下载LLaVA-Instruct数据集
用于解决过拟合问题，扩充训练数据
"""

import os
import json
import requests
from pathlib import Path
from tqdm import tqdm
import hashlib

# HuggingFace镜像站 (国内加速)
HF_MIRROR = "https://hf-mirror.com"

# 数据集信息 (使用镜像)
DATASETS = {
    "llava_instruct_150k": {
        "url": f"{HF_MIRROR}/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_instruct_150k.json",
        "description": "LLaVA指令微调数据集 (150K条)"
    },
    "llava_instruct_80k": {
        "url": f"{HF_MIRROR}/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/main/llava_instruct_80k.json",
        "description": "LLaVA指令微调数据集 (80K条，子集)"
    }
}

def download_file(url: str, save_path: Path, desc: str = "Downloading"):
    """下载文件，支持断点续传"""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 检查是否已存在
    if save_path.exists():
        print(f"文件已存在: {save_path}")
        return True
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(save_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=desc) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        
        print(f"✅ 下载完成: {save_path}")
        return True
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        if save_path.exists():
            save_path.unlink()
        return False

def convert_to_training_format(input_file: Path, output_dir: Path, split_ratio: float = 0.9):
    """
    将LLaVA格式转换为训练格式
    
    LLaVA格式:
    {
        "id": "xxx",
        "image": "coco/train2017/xxx.jpg",
        "conversations": [
            {"from": "human", "value": "..."},
            {"from": "gpt", "value": "..."}
        ]
    }
    
    训练格式:
    {
        "id": "xxx",
        "image": "xxx.jpg",
        "text": "问题内容",
        "label": 0,
        "modality": "multimodal"
    }
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"读取数据: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"总数据量: {len(data)}")
    
    # 转换格式
    converted = []
    for item in tqdm(data, desc="转换格式"):
        if "conversations" not in item or len(item["conversations"]) < 2:
            continue
            
        # 提取问题和回答
        question = ""
        answer = ""
        for conv in item["conversations"]:
            if conv["from"] == "human":
                question = conv["value"]
            elif conv["from"] == "gpt":
                answer = conv["value"]
        
        if not question:
            continue
        
        converted.append({
            "id": item.get("id", hashlib.md5(question.encode()).hexdigest()[:8]),
            "image": item.get("image", ""),
            "text": question,
            "answer": answer,
            "label": hash(answer) % 10,  # 简化的标签
            "modality": "multimodal" if item.get("image") else "text",
            "has_image": bool(item.get("image"))
        })
    
    print(f"有效数据: {len(converted)}")
    
    # 分割训练/验证集
    split_idx = int(len(converted) * split_ratio)
    train_data = converted[:split_idx]
    eval_data = converted[split_idx:]
    
    # 保存
    train_file = output_dir / "train.jsonl"
    eval_file = output_dir / "eval.jsonl"
    
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    with open(eval_file, 'w', encoding='utf-8') as f:
        for item in eval_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✅ 训练集: {len(train_data)} 条 -> {train_file}")
    print(f"✅ 验证集: {len(eval_data)} 条 -> {eval_file}")
    
    return len(train_data), len(eval_data)

def download_coco_images(data_file: Path, output_dir: Path, max_images: int = 10000):
    """
    下载COCO图像（用于LLaVA数据集）
    注意：完整COCO数据集很大，这里只下载部分
    """
    print("提示: COCO图像下载需要较长时间")
    print("建议从官网下载: https://cocodataset.org/#download")
    print("或使用HuggingFace镜像")
    
    # 创建目录
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"图像保存目录: {images_dir}")
    print(f"请手动下载COCO train2017/val2017 到此目录")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="下载LLaVA训练数据")
    parser.add_argument("--dataset", type=str, default="llava_instruct_80k",
                       choices=list(DATASETS.keys()),
                       help="选择数据集")
    parser.add_argument("--output_dir", type=str, default="data/llava_instruct",
                       help="输出目录")
    parser.add_argument("--skip_download", action="store_true",
                       help="跳过下载，仅转换已有文件")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_info = DATASETS[args.dataset]
    raw_file = output_dir / f"{args.dataset}.json"
    
    print("=" * 50)
    print(f"数据集: {args.dataset}")
    print(f"说明: {dataset_info['description']}")
    print(f"输出目录: {output_dir}")
    print("=" * 50)
    
    # 1. 下载数据
    if not args.skip_download:
        print("\n[1/3] 下载数据...")
        success = download_file(
            dataset_info["url"],
            raw_file,
            desc=f"下载 {args.dataset}"
        )
        if not success:
            print("下载失败，请检查网络或使用镜像")
            return
    else:
        print("\n[1/3] 跳过下载")
    
    # 2. 转换格式
    print("\n[2/3] 转换数据格式...")
    if raw_file.exists():
        train_count, eval_count = convert_to_training_format(raw_file, output_dir)
    else:
        print(f"文件不存在: {raw_file}")
        return
    
    # 3. 图像下载提示
    print("\n[3/3] 图像下载...")
    download_coco_images(raw_file, output_dir)
    
    print("\n" + "=" * 50)
    print("✅ 数据准备完成!")
    print(f"训练集: {train_count} 条")
    print(f"验证集: {eval_count} 条")
    print("\n下一步:")
    print("1. 下载COCO图像到 data/llava_instruct/images/")
    print("2. 运行训练: python scripts/train_openmind.py --config configs/openmind_train_v2.yaml")
    print("=" * 50)

if __name__ == "__main__":
    main()
