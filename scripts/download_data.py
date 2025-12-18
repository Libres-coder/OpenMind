"""
数据集下载脚本
支持COCO Captions等数据集的下载
"""

import os
import sys
import argparse
import urllib.request
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    """下载进度条"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: str, desc: str = None):
    """下载文件并显示进度"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.exists():
        print(f"文件已存在: {output_path}")
        return str(output_path)
    
    print(f"下载: {url}")
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
        urllib.request.urlretrieve(url, output_path, reporthook=t.update_to)
    
    return str(output_path)


def extract_zip(zip_path: str, output_dir: str):
    """解压ZIP文件"""
    print(f"解压: {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
    print(f"解压完成: {output_dir}")


def download_coco_val(output_dir: str = "data/coco"):
    """
    下载COCO 2017验证集（较小，约1GB图像 + 241MB annotations）
    用于快速验证训练流程
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 50)
    print("下载COCO 2017验证集")
    print("=" * 50)
    
    # 验证集图像 (~1GB, 5000张图)
    val_images_url = "http://images.cocodataset.org/zips/val2017.zip"
    val_zip = download_file(val_images_url, output_dir / "val2017.zip", "val2017.zip")
    extract_zip(val_zip, output_dir)
    
    # Annotations (~241MB)
    annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    ann_zip = download_file(annotations_url, output_dir / "annotations.zip", "annotations.zip")
    extract_zip(ann_zip, output_dir)
    
    print("\n✅ COCO验证集下载完成!")
    print(f"图像目录: {output_dir / 'val2017'}")
    print(f"Annotations: {output_dir / 'annotations/captions_val2017.json'}")
    
    return {
        'image_dir': str(output_dir / "val2017"),
        'annotation_file': str(output_dir / "annotations/captions_val2017.json")
    }


def download_coco_train(output_dir: str = "data/coco"):
    """
    下载COCO 2017训练集（较大，约18GB图像）
    用于完整训练
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 50)
    print("下载COCO 2017训练集 (约18GB)")
    print("=" * 50)
    
    # 训练集图像 (~18GB, 118287张图)
    train_images_url = "http://images.cocodataset.org/zips/train2017.zip"
    train_zip = download_file(train_images_url, output_dir / "train2017.zip", "train2017.zip")
    extract_zip(train_zip, output_dir)
    
    # Annotations（如果还没下载）
    ann_path = output_dir / "annotations/captions_train2017.json"
    if not ann_path.exists():
        annotations_url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        ann_zip = download_file(annotations_url, output_dir / "annotations.zip", "annotations.zip")
        extract_zip(ann_zip, output_dir)
    
    print("\n✅ COCO训练集下载完成!")
    print(f"图像目录: {output_dir / 'train2017'}")
    print(f"Annotations: {output_dir / 'annotations/captions_train2017.json'}")
    
    return {
        'image_dir': str(output_dir / "train2017"),
        'annotation_file': str(output_dir / "annotations/captions_train2017.json")
    }


def create_sample_dataset(output_dir: str = "data/sample"):
    """
    创建小型示例数据集用于测试
    不需要下载，直接生成模拟数据
    """
    import json
    import numpy as np
    from PIL import Image
    
    output_dir = Path(output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 50)
    print("创建示例数据集")
    print("=" * 50)
    
    # 生成100张随机图像
    samples = []
    for i in range(100):
        # 创建随机图像
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img_path = image_dir / f"image_{i:04d}.jpg"
        img.save(img_path)
        
        # 创建样本
        samples.append({
            "image": f"images/image_{i:04d}.jpg",
            "text": f"This is sample image number {i}",
            "label": i % 10
        })
    
    # 保存JSONL
    train_file = output_dir / "train.jsonl"
    eval_file = output_dir / "eval.jsonl"
    
    with open(train_file, 'w') as f:
        for sample in samples[:80]:
            f.write(json.dumps(sample) + '\n')
    
    with open(eval_file, 'w') as f:
        for sample in samples[80:]:
            f.write(json.dumps(sample) + '\n')
    
    print(f"\n✅ 示例数据集创建完成!")
    print(f"训练数据: {train_file} (80条)")
    print(f"评估数据: {eval_file} (20条)")
    print(f"图像目录: {image_dir}")
    
    return {
        'train_file': str(train_file),
        'eval_file': str(eval_file),
        'image_dir': str(output_dir)
    }


def main():
    parser = argparse.ArgumentParser(description="下载训练数据集")
    parser.add_argument("--dataset", type=str, default="sample",
                       choices=["coco_val", "coco_train", "sample"],
                       help="要下载的数据集")
    parser.add_argument("--output", type=str, default="data",
                       help="输出目录")
    args = parser.parse_args()
    
    if args.dataset == "coco_val":
        download_coco_val(f"{args.output}/coco")
    elif args.dataset == "coco_train":
        download_coco_train(f"{args.output}/coco")
    elif args.dataset == "sample":
        create_sample_dataset(f"{args.output}/sample")


if __name__ == "__main__":
    main()
