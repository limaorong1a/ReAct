#!/bin/bash
# RTX 5090 服务器自动训练脚本
# 运行方式: bash server_auto_train.sh

set -e

# ====================================
# 配置区域
# ====================================

PROJECT_DIR="/data_nvme/react_training"
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
OUTPUT_NAME="qwen2.5-7b-react"

# 训练参数（针对 RTX 5090 优化）
BATCH_SIZE=4
GRAD_ACCUM=4
LORA_RANK=32
EPOCHS=3
LEARNING_RATE=5e-5

# ====================================

echo "🚀 ReAct 模型自动训练脚本"
echo "================================"
echo "GPU: RTX 5090 D"
echo "模型: ${MODEL_NAME}"
echo "输出: ${OUTPUT_NAME}"
echo "================================"
echo ""

# Step 1: 环境检查
echo "🔍 Step 1: 环境检查"
echo "-------------------"

# 检查 CUDA
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ 未检测到 NVIDIA GPU"
    exit 1
fi

echo "GPU 信息:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 检查 Python
if ! command -v python &> /dev/null; then
    echo "❌ 未找到 Python"
    exit 1
fi

echo "Python 版本: $(python --version)"

# 检查虚拟环境
if [ ! -d "${PROJECT_DIR}/react_env" ]; then
    echo "⚠️  虚拟环境不存在，正在创建..."
    cd ${PROJECT_DIR}
    python -m venv react_env
fi

# 激活虚拟环境
echo "✅ 激活虚拟环境"
source ${PROJECT_DIR}/react_env/bin/activate

# Step 2: 安装依赖
echo ""
echo "📦 Step 2: 检查依赖"
echo "-------------------"

if ! python -c "import llamafactory" 2>/dev/null; then
    echo "⚠️  LLaMA-Factory 未安装，正在安装..."
    
    cd ${PROJECT_DIR}
    
    # 安装 PyTorch（如果需要）
    if ! python -c "import torch" 2>/dev/null; then
        echo "安装 PyTorch..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    fi
    
    # 克隆并安装 LLaMA-Factory
    if [ ! -d "LLaMA-Factory" ]; then
        git clone https://github.com/hiyouga/LLaMA-Factory.git
    fi
    
    cd LLaMA-Factory
    pip install -e .[torch,metrics]
    
    # 安装加速库（可选）
    echo "安装加速库..."
    pip install flash-attn --no-build-isolation || echo "⚠️  Flash Attention 安装失败（可选）"
    
    cd ${PROJECT_DIR}
else
    echo "✅ LLaMA-Factory 已安装"
fi

# 验证 PyTorch
echo ""
echo "验证 PyTorch 和 CUDA:"
python << 'PYTHON_CHECK'
import torch
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
PYTHON_CHECK

# Step 3: 检查数据
echo ""
echo "📊 Step 3: 检查训练数据"
echo "-------------------"

if [ ! -f "${PROJECT_DIR}/data/react_train_alpaca.json" ]; then
    echo "❌ 未找到训练数据"
    echo "请先运行: python prepare_training_data.py"
    echo "或使用上传脚本上传数据"
    exit 1
fi

echo "✅ 训练数据: $(wc -l < ${PROJECT_DIR}/data/react_train_alpaca.json) 行"
echo "✅ 测试数据: $(wc -l < ${PROJECT_DIR}/data/react_test_alpaca.json) 行"

# Step 4: 配置 dataset_info.json
echo ""
echo "⚙️  Step 4: 配置数据集"
echo "-------------------"

cat > ${PROJECT_DIR}/LLaMA-Factory/data/dataset_info.json << EOF
{
  "react_training": {
    "file_name": "${PROJECT_DIR}/data/react_train_alpaca.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  },
  "react_testing": {
    "file_name": "${PROJECT_DIR}/data/react_test_alpaca.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  }
}
EOF

echo "✅ dataset_info.json 配置完成"

# Step 5: 开始训练
echo ""
echo "🎯 Step 5: 开始训练"
echo "-------------------"
echo ""
echo "训练配置:"
echo "  - Batch Size: ${BATCH_SIZE}"
echo "  - Gradient Accumulation: ${GRAD_ACCUM}"
echo "  - Effective Batch: $((BATCH_SIZE * GRAD_ACCUM))"
echo "  - LoRA Rank: ${LORA_RANK}"
echo "  - Epochs: ${EPOCHS}"
echo "  - Learning Rate: ${LEARNING_RATE}"
echo ""

read -p "确认开始训练？(y/n): " confirm
if [ "$confirm" != "y" ]; then
    echo "❌ 训练已取消"
    exit 0
fi

# 创建日志目录
mkdir -p ${PROJECT_DIR}/logs

# 记录训练开始时间
echo "训练开始时间: $(date)" > ${PROJECT_DIR}/logs/train_start.log

cd ${PROJECT_DIR}/LLaMA-Factory

# 启动训练（使用 nohup 后台运行）
nohup llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path ${MODEL_NAME} \
    --dataset react_training \
    --template qwen \
    --finetuning_type lora \
    --lora_target all \
    --lora_rank ${LORA_RANK} \
    --lora_alpha $((LORA_RANK * 2)) \
    --lora_dropout 0.05 \
    --use_rslora true \
    --output_dir ${PROJECT_DIR}/output/${OUTPUT_NAME} \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 2048 \
    --preprocessing_num_workers 8 \
    --per_device_train_batch_size ${BATCH_SIZE} \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --gradient_accumulation_steps ${GRAD_ACCUM} \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_ratio 0.1 \
    --save_steps 100 \
    --eval_steps 100 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --learning_rate ${LEARNING_RATE} \
    --num_train_epochs ${EPOCHS} \
    --val_size 0.1 \
    --plot_loss \
    --bf16 \
    --gradient_checkpointing \
    --optim adamw_torch_fused \
    --logging_dir ${PROJECT_DIR}/logs \
    --report_to tensorboard \
    > ${PROJECT_DIR}/logs/train.log 2>&1 &

# 记录进程 ID
TRAIN_PID=$!
echo ${TRAIN_PID} > ${PROJECT_DIR}/train.pid

echo ""
echo "================================"
echo "✨ 训练已在后台启动！"
echo "================================"
echo ""
echo "进程 ID: ${TRAIN_PID}"
echo "日志文件: ${PROJECT_DIR}/logs/train.log"
echo ""
echo "监控命令:"
echo "  - 查看日志: tail -f ${PROJECT_DIR}/logs/train.log"
echo "  - 监控 GPU: watch -n 1 nvidia-smi"
echo "  - TensorBoard: tensorboard --logdir=${PROJECT_DIR}/logs --host=0.0.0.0 --port=6006"
echo ""
echo "管理命令:"
echo "  - 查看进程: ps -p ${TRAIN_PID}"
echo "  - 停止训练: kill ${TRAIN_PID}"
echo ""

# 启动 GPU 监控（可选）
read -p "是否启动 GPU 监控？(y/n): " monitor
if [ "$monitor" = "y" ]; then
    nohup bash -c 'while true; do nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv >> '${PROJECT_DIR}'/logs/gpu_monitor.csv; sleep 60; done' > /dev/null 2>&1 &
    GPU_MONITOR_PID=$!
    echo ${GPU_MONITOR_PID} > ${PROJECT_DIR}/gpu_monitor.pid
    echo "✅ GPU 监控已启动 (PID: ${GPU_MONITOR_PID})"
    echo "   日志: ${PROJECT_DIR}/logs/gpu_monitor.csv"
fi

# 等待几秒并显示初始日志
echo ""
echo "初始日志输出:"
echo "-------------------"
sleep 5
tail -n 20 ${PROJECT_DIR}/logs/train.log

echo ""
echo "💡 提示: 训练预计需要 3-6 小时"
echo "💡 建议: 使用 tmux 或定期检查日志"

