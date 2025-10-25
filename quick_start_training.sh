#!/bin/bash
# ReAct 模型训练快速启动脚本

set -e  # 遇到错误立即停止

echo "🚀 ReAct 模型训练快速启动"
echo "================================"

# 检查 Python 版本
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "✓ Python 版本: $python_version"

# 1. 准备训练数据
echo ""
echo "📊 Step 1: 准备训练数据..."
if [ ! -f "react_train_alpaca.json" ]; then
    python prepare_training_data.py
    echo "✓ 数据转换完成"
else
    echo "✓ 训练数据已存在"
fi

# 2. 选择训练方法
echo ""
echo "🎯 Step 2: 选择训练方法"
echo "1) LLaMA-Factory (推荐新手，有 Web UI)"
echo "2) Unsloth (最快，命令行)"
echo "3) 手动配置"
read -p "请选择 [1-3]: " choice

case $choice in
    1)
        echo ""
        echo "📦 安装 LLaMA-Factory..."
        
        if [ ! -d "LLaMA-Factory" ]; then
            git clone https://github.com/hiyouga/LLaMA-Factory.git
            cd LLaMA-Factory
            pip install -e .
            cd ..
        else
            echo "✓ LLaMA-Factory 已安装"
        fi
        
        # 复制数据文件
        echo "📋 复制数据文件到 LLaMA-Factory..."
        cp react_*.json LLaMA-Factory/data/
        
        # 更新 dataset_info.json
        cat > LLaMA-Factory/data/dataset_info.json << 'EOF'
{
  "react_training": {
    "file_name": "react_train_alpaca.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  },
  "react_testing": {
    "file_name": "react_test_alpaca.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  }
}
EOF
        
        echo ""
        echo "✅ 准备完成！"
        echo ""
        echo "下一步操作："
        echo "1. 启动 Web UI:"
        echo "   cd LLaMA-Factory && llamafactory-cli webui"
        echo ""
        echo "2. 在浏览器中访问: http://localhost:7860"
        echo ""
        echo "3. 配置训练参数:"
        echo "   - 模型: Qwen/Qwen2.5-7B-Instruct"
        echo "   - 数据集: react_training"
        echo "   - 微调方法: lora"
        echo "   - 学习率: 5e-5"
        echo ""
        read -p "是否立即启动 Web UI? [y/n]: " start_ui
        if [ "$start_ui" = "y" ]; then
            cd LLaMA-Factory
            llamafactory-cli webui
        fi
        ;;
        
    2)
        echo ""
        echo "📦 安装 Unsloth..."
        pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
        pip install --no-deps trl peft accelerate bitsandbytes
        
        echo ""
        echo "✅ Unsloth 安装完成！"
        echo ""
        echo "下一步："
        echo "1. 查看 train_with_unsloth.py 脚本"
        echo "2. 根据需要修改参数"
        echo "3. 运行: python train_with_unsloth.py"
        ;;
        
    3)
        echo ""
        echo "📖 手动配置指南："
        echo ""
        echo "请参考 TRAINING_GUIDE.md 文件中的详细说明"
        echo "主要步骤："
        echo "1. 安装训练框架（LLaMA-Factory/Unsloth/Axolotl）"
        echo "2. 准备数据（已完成）"
        echo "3. 配置训练参数"
        echo "4. 开始训练"
        echo ""
        echo "配置文件位置："
        echo "- llama_factory_train_config.yaml"
        echo "- TRAINING_GUIDE.md"
        ;;
        
    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac

echo ""
echo "🎉 准备工作完成！"
echo ""
echo "📚 更多信息请查看:"
echo "   - TRAINING_GUIDE.md (完整训练指南)"
echo "   - llama_factory_train_config.yaml (训练配置)"
echo "   - prepare_training_data.py (数据准备脚本)"

