import sys
import torch
import platform

def verify_environment():
    print("=" * 60)
    print("🔍 多模态模型训练环境验证")
    print("=" * 60)
    
    print(f"\n📌 系统信息:")
    print(f"  操作系统: {platform.system()} {platform.release()}")
    print(f"  Python版本: {sys.version.split()[0]}")
    print(f"  处理器: {platform.processor()}")
    
    print(f"\n📦 核心依赖:")
    try:
        print(f"  PyTorch: {torch.__version__}")
    except:
        print("  ❌ PyTorch未安装")
        return False
    
    try:
        import transformers
        print(f"  Transformers: {transformers.__version__}")
    except:
        print("  ⚠️  Transformers未安装（推荐安装）")
    
    try:
        import accelerate
        print(f"  Accelerate: {accelerate.__version__}")
    except:
        print("  ⚠️  Accelerate未安装（推荐安装）")
    
    print(f"\n🎮 GPU信息:")
    cuda_available = torch.cuda.is_available()
    print(f"  CUDA可用: {'✅ 是' if cuda_available else '❌ 否'}")
    
    if cuda_available:
        print(f"  CUDA版本: {torch.version.cuda}")
        print(f"  cuDNN版本: {torch.backends.cudnn.version()}")
        gpu_count = torch.cuda.device_count()
        print(f"  GPU数量: {gpu_count}")
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            print(f"\n  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    总显存: {props.total_memory / 1024**3:.2f} GB")
            print(f"    计算能力: {props.major}.{props.minor}")
            
            if torch.cuda.is_available():
                torch.cuda.set_device(i)
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                print(f"    已分配显存: {allocated:.2f} GB")
                print(f"    已保留显存: {reserved:.2f} GB")
    else:
        print("  ⚠️  未检测到GPU，将使用CPU训练（速度会很慢）")
    
    print(f"\n🧪 快速功能测试:")
    try:
        print("  测试张量创建...", end=" ")
        x = torch.randn(100, 100)
        print("✅")
        
        if cuda_available:
            print("  测试GPU张量...", end=" ")
            x_gpu = torch.randn(100, 100).cuda()
            print("✅")
            
            print("  测试GPU计算...", end=" ")
            y = torch.matmul(x_gpu, x_gpu)
            print("✅")
    except Exception as e:
        print(f"❌ {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ 环境验证完成！")
    
    if not cuda_available:
        print("\n⚠️  警告: 未检测到GPU")
        print("建议:")
        print("  1. 检查NVIDIA驱动是否正确安装")
        print("  2. 确认PyTorch安装的CUDA版本与系统匹配")
        print("  3. 运行: nvidia-smi 查看GPU状态")
    
    print("\n📚 下一步:")
    print("  1. 运行: python scripts/test_model_architecture.py")
    print("  2. 运行: python scripts/create_sample_data.py")
    print("  3. 运行: python scripts/test_data_pipeline.py")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = verify_environment()
    sys.exit(0 if success else 1)
