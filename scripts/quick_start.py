"""
快速开始脚本 - 一键运行完整验证流程
"""
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 错误: {e}")
        print(e.stderr)
        return False

def main():
    print("""
    ╔════════════════════════════════════════════════════════╗
    ║     多模态智能模型 - 快速开始向导                         ║
    ║                                                        ║
    ║     本脚本将执行以下操作:                                 ║
    ║     1. 验证环境配置                                      ║
    ║     2. 创建示例数据                                      ║
    ║     3. 测试模型架构                                      ║
    ║     4. 测试数据加载                                      ║
    ║     5. 运行微型训练测试                                   ║
    ╚════════════════════════════════════════════════════════╝
    """)
    
    input("按回车键开始...")
    
    scripts = [
        ("python scripts/verify_environment.py", "步骤1: 验证环境"),
        ("python scripts/create_sample_data.py", "步骤2: 创建示例数据"),
        ("python scripts/test_model_architecture.py", "步骤3: 测试模型架构"),
        ("python scripts/test_data_pipeline.py", "步骤4: 测试数据加载"),
    ]
    
    for cmd, desc in scripts:
        if not run_command(cmd, desc):
            print(f"\n❌ {desc} 失败，请检查错误信息")
            return False
    
    print("\n" + "="*60)
    print("🎉 所有基础验证测试通过！")
    print("="*60)
    
    print("\n📝 下一步建议:")
    print("  1. 查看开发路线图: DEVELOPMENT_ROADMAP.md")
    print("  2. 准备真实训练数据")
    print("  3. 配置训练参数: configs/training_config.yaml")
    print("  4. 开始小规模训练测试")
    
    print("\n💡 快速训练命令:")
    print("  python src/train_multimodal.py \\")
    print("      --config configs/training_config.yaml \\")
    print("      --stage pretrain \\")
    print("      --num_epochs 1 \\")
    print("      --max_steps 50")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
