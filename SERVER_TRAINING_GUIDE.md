# RTX 5090 服务器端训练指南

## 🖥️ 你的服务器配置

```
GPU: NVIDIA GeForce RTX 5090 D × 1 (24GB 显存)
CPU: 16 vCPU
内存: 47GB RAM
存储: /data_nvme (NVMe SSD)
```

**性能评估**: ⭐⭐⭐⭐⭐ 顶级配置！
- 可以训练 Qwen2.5-14B 甚至更大的模型
- 支持较大的 batch size，训练速度快
- 内存和存储充足，无需担心

---

## 📋 完整训练流程

### 阶段 1: 服务器环境配置

#### Step 1: 连接到服务器

```bash
# SSH 连接（替换为你的服务器地址和用户名）
ssh username@your-server-ip

# 或使用密钥
ssh -i /path/to/key.pem username@your-server-ip
```

#### Step 2: 检查 GPU 和 CUDA

```bash
# 检查 GPU
nvidia-smi

# 应该看到 RTX 5090 的信息
# 记下 CUDA Version（例如 12.2）

# 检查 CUDA 路径
echo $CUDA_HOME
nvcc --version
```

#### Step 3: 创建工作目录

```bash
# 使用 NVMe 数据盘（速度快）
cd /data_nvme

# 创建项目目录
mkdir -p react_training
cd react_training

# 创建子目录
mkdir -p {data,models,output,logs,scripts}
```

#### Step 4: 配置 Python 环境

```bash
# 检查 Python 版本（需要 3.8+）
python --version

# 创建虚拟环境
python -m venv react_env

# 激活环境
source react_env/bin/activate

# 升级 pip
pip install --upgrade pip setuptools wheel

# 安装基础依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# 注意：根据你的 CUDA 版本选择对应的 PyTorch 版本
# cu121 = CUDA 12.1, cu118 = CUDA 11.8

# 验证 PyTorch 和 GPU
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

---

### 阶段 2: 安装训练框架

#### 选项 A: LLaMA-Factory（推荐，功能全面）

```bash
cd /data_nvme/react_training

# 克隆仓库
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory

# 安装
pip install -e .[torch,metrics]

# 安装额外依赖（可选但推荐）
pip install flash-attn --no-build-isolation  # 加速训练
pip install deepspeed  # 支持大模型训练

# 验证安装
llamafactory-cli version
```

#### 选项 B: Unsloth（速度最快）

```bash
# Unsloth 安装
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes

# 验证
python -c "import unsloth; print('Unsloth installed successfully')"
```

---

### 阶段 3: 上传数据到服务器

#### 方法 1: 使用 SCP（推荐）

```bash
# 在本地电脑运行（Windows PowerShell 或 Mac/Linux Terminal）

# 首先在本地运行数据准备脚本
python prepare_training_data.py

# 上传数据文件到服务器
scp react_train_alpaca.json username@server-ip:/data_nvme/react_training/data/
scp react_test_alpaca.json username@server-ip:/data_nvme/react_training/data/
scp react_train_sharegpt.json username@server-ip:/data_nvme/react_training/data/
scp react_test_sharegpt.json username@server-ip:/data_nvme/react_training/data/

# 或批量上传
scp react_*.json username@server-ip:/data_nvme/react_training/data/
```

#### 方法 2: 使用 rsync（更高效）

```bash
# 在本地运行
rsync -avz --progress react_*.json username@server-ip:/data_nvme/react_training/data/
```

#### 方法 3: 使用 SFTP 或 FTP 客户端

- **Windows**: 使用 WinSCP, FileZilla
- **Mac**: 使用 Cyberduck, FileZilla
- **Linux**: 使用 FileZilla

#### 方法 4: 在服务器上直接准备数据（如果有原始数据）

```bash
# 上传原始 JSON 文件夹
scp -r generated_samples_* username@server-ip:/data_nvme/react_training/

# 上传准备脚本
scp prepare_training_data.py username@server-ip:/data_nvme/react_training/

# 在服务器上运行
ssh username@server-ip
cd /data_nvme/react_training
source react_env/bin/activate
python prepare_training_data.py
```

---

### 阶段 4: 配置训练参数

#### 创建训练配置文件

在服务器上创建 `/data_nvme/react_training/train_config.yaml`：

```bash
cat > /data_nvme/react_training/train_config.yaml << 'EOF'
# RTX 5090 优化配置 - Qwen2.5-7B LoRA 微调

### 模型配置
model_name_or_path: Qwen/Qwen2.5-7B-Instruct
# 或使用本地路径（如果已下载）
# model_name_or_path: /data_nvme/react_training/models/Qwen2.5-7B-Instruct

### 数据配置
dataset: react_training
template: qwen
cutoff_len: 2048
preprocessing_num_workers: 8

### LoRA 配置
finetuning_type: lora
lora_target: all
lora_rank: 32  # RTX 5090 显存充足，可以用更大的 rank
lora_alpha: 64
lora_dropout: 0.05
use_rslora: true  # 提升训练稳定性

### 训练参数（优化用于 RTX 5090）
num_train_epochs: 3
per_device_train_batch_size: 4  # RTX 5090 可以用更大的 batch
per_device_eval_batch_size: 4
gradient_accumulation_steps: 4  # 有效 batch = 4*4 = 16
learning_rate: 5.0e-5
lr_scheduler_type: cosine
warmup_ratio: 0.1
max_grad_norm: 1.0

### 优化器配置
optim: adamw_torch_fused  # RTX 5090 支持 fused 优化器，更快
weight_decay: 0.01

### 输出配置
output_dir: /data_nvme/react_training/output/qwen2.5-7b-react
logging_dir: /data_nvme/react_training/logs
logging_steps: 10
save_steps: 100
save_total_limit: 3  # 只保留最近 3 个检查点
eval_steps: 100

### 评估配置
evaluation_strategy: steps
val_size: 0.1
load_best_model_at_end: true
metric_for_best_model: eval_loss

### 性能优化（RTX 5090 专属）
bf16: true  # RTX 5090 支持 BF16，比 FP16 更稳定
# fp16: false  # 不使用 FP16
gradient_checkpointing: true
ddp_find_unused_parameters: false

# Flash Attention 2（如果已安装）
use_flash_attention_2: true

### 推理配置
predict_with_generate: true
max_new_tokens: 1024
do_sample: false  # 评估时使用 greedy decoding

### 其他配置
report_to: tensorboard
overwrite_output_dir: true
save_only_model: false
seed: 42
EOF
```

#### 配置数据集信息

```bash
# 如果使用 LLaMA-Factory
cat > /data_nvme/react_training/LLaMA-Factory/data/dataset_info.json << 'EOF'
{
  "react_training": {
    "file_name": "/data_nvme/react_training/data/react_train_alpaca.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  },
  "react_testing": {
    "file_name": "/data_nvme/react_training/data/react_test_alpaca.json",
    "columns": {
      "prompt": "instruction",
      "query": "input",
      "response": "output"
    }
  }
}
EOF
```

---

### 阶段 5: 开始训练

#### 方法 1: 使用 LLaMA-Factory 命令行

创建训练脚本 `/data_nvme/react_training/scripts/train.sh`：

```bash
cat > /data_nvme/react_training/scripts/train.sh << 'EOF'
#!/bin/bash

# 激活虚拟环境
source /data_nvme/react_training/react_env/bin/activate

# 设置工作目录
cd /data_nvme/react_training/LLaMA-Factory

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export WANDB_DISABLED=true  # 如果不使用 wandb

# 开始训练
llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --dataset react_training \
    --template qwen \
    --finetuning_type lora \
    --lora_target all \
    --lora_rank 32 \
    --lora_alpha 64 \
    --output_dir /data_nvme/react_training/output/qwen2.5-7b-react \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 2048 \
    --preprocessing_num_workers 8 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_ratio 0.1 \
    --save_steps 100 \
    --eval_steps 100 \
    --evaluation_strategy steps \
    --load_best_model_at_end \
    --learning_rate 5e-5 \
    --num_train_epochs 3.0 \
    --val_size 0.1 \
    --plot_loss \
    --bf16 \
    --use_flash_attention_2 \
    --gradient_checkpointing
EOF

# 添加执行权限
chmod +x /data_nvme/react_training/scripts/train.sh
```

运行训练：

```bash
# 后台运行（推荐）
nohup /data_nvme/react_training/scripts/train.sh > /data_nvme/react_training/logs/train.log 2>&1 &

# 记录进程 ID
echo $! > /data_nvme/react_training/train.pid

# 查看实时日志
tail -f /data_nvme/react_training/logs/train.log
```

#### 方法 2: 使用 tmux 或 screen（推荐）

```bash
# 安装 tmux（如果没有）
sudo apt-get install tmux

# 创建新会话
tmux new -s react_training

# 在 tmux 中运行训练
source /data_nvme/react_training/react_env/bin/activate
cd /data_nvme/react_training/scripts
./train.sh

# 分离会话（训练继续运行）
# 按 Ctrl+B，然后按 D

# 重新连接
tmux attach -t react_training

# 查看所有会话
tmux ls
```

---

### 阶段 6: 监控训练进度

#### 1. 查看实时日志

```bash
# 实时查看日志
tail -f /data_nvme/react_training/logs/train.log

# 查看最近的错误
grep -i error /data_nvme/react_training/logs/train.log
```

#### 2. 监控 GPU 使用

```bash
# 实时监控 GPU
watch -n 1 nvidia-smi

# 或使用 gpustat（更友好）
pip install gpustat
gpustat -i 1

# 后台监控并记录
while true; do
    nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv >> /data_nvme/react_training/logs/gpu_monitor.log
    sleep 60
done &
```

#### 3. 使用 TensorBoard 可视化

```bash
# 在服务器上启动 TensorBoard
source /data_nvme/react_training/react_env/bin/activate
tensorboard --logdir=/data_nvme/react_training/logs --host=0.0.0.0 --port=6006

# 通过 SSH 隧道访问（在本地电脑运行）
ssh -L 6006:localhost:6006 username@server-ip

# 然后在本地浏览器打开
# http://localhost:6006
```

#### 4. 检查训练进度

```bash
# 查看输出目录
ls -lh /data_nvme/react_training/output/qwen2.5-7b-react/

# 查看检查点
ls -lh /data_nvme/react_training/output/qwen2.5-7b-react/checkpoint-*/

# 查看训练曲线（如果启用了 plot_loss）
ls /data_nvme/react_training/output/qwen2.5-7b-react/training_loss.png
```

---

### 阶段 7: 训练完成后的处理

#### 1. 评估模型

```bash
# 在测试集上评估
cd /data_nvme/react_training/LLaMA-Factory

llamafactory-cli train \
    --stage sft \
    --do_eval \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --adapter_name_or_path /data_nvme/react_training/output/qwen2.5-7b-react \
    --dataset react_testing \
    --template qwen \
    --finetuning_type lora \
    --output_dir /data_nvme/react_training/evaluation \
    --per_device_eval_batch_size 4 \
    --predict_with_generate \
    --bf16
```

#### 2. 合并 LoRA 权重

```bash
# 将 LoRA 合并到基础模型
llamafactory-cli export \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --adapter_name_or_path /data_nvme/react_training/output/qwen2.5-7b-react \
    --template qwen \
    --finetuning_type lora \
    --export_dir /data_nvme/react_training/merged_model \
    --export_size 2 \
    --export_device cpu \
    --export_legacy_format False
```

#### 3. 下载模型到本地

```bash
# 在本地电脑运行

# 下载合并后的模型
scp -r username@server-ip:/data_nvme/react_training/merged_model ./

# 或下载 LoRA 适配器（更小）
scp -r username@server-ip:/data_nvme/react_training/output/qwen2.5-7b-react ./

# 使用 rsync（断点续传）
rsync -avz --progress username@server-ip:/data_nvme/react_training/merged_model ./
```

---

## 🚀 快速启动脚本

创建一键部署脚本：

```bash
cat > /data_nvme/react_training/quick_start.sh << 'EOF'
#!/bin/bash
set -e

echo "🚀 ReAct 模型训练 - 快速启动"
echo "================================"

# 1. 创建目录
echo "📁 创建目录结构..."
mkdir -p /data_nvme/react_training/{data,models,output,logs,scripts}

# 2. 创建虚拟环境
echo "🐍 配置 Python 环境..."
cd /data_nvme/react_training
python -m venv react_env
source react_env/bin/activate

# 3. 安装依赖
echo "📦 安装依赖（这可能需要几分钟）..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 4. 安装 LLaMA-Factory
echo "🔧 安装 LLaMA-Factory..."
if [ ! -d "LLaMA-Factory" ]; then
    git clone https://github.com/hiyouga/LLaMA-Factory.git
fi
cd LLaMA-Factory
pip install -e .[torch,metrics]

# 5. 验证
echo "✅ 验证安装..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

echo ""
echo "✨ 环境配置完成！"
echo ""
echo "下一步："
echo "1. 上传数据到 /data_nvme/react_training/data/"
echo "2. 配置 dataset_info.json"
echo "3. 运行训练: ./scripts/train.sh"
EOF

chmod +x /data_nvme/react_training/quick_start.sh
```

---

## 📊 预期性能

基于 RTX 5090 的配置：

| 模型 | Batch Size | 训练时间 | 显存占用 | 推荐 |
|------|-----------|---------|---------|------|
| Qwen2.5-3B | 8 | ~1.5h | ~12GB | ⭐⭐⭐ |
| Qwen2.5-7B | 4 | ~3h | ~18GB | ⭐⭐⭐⭐⭐ |
| Qwen2.5-14B | 2 | ~6h | ~22GB | ⭐⭐⭐⭐ |
| Qwen2.5-32B | 1 (QLoRA) | ~15h | ~23GB | ⭐⭐⭐ |

**推荐**: Qwen2.5-7B，性能与速度最平衡

---

## 💡 优化建议

### 提升训练速度

1. **使用 Flash Attention 2**
```bash
pip install flash-attn --no-build-isolation
# 在配置中添加 use_flash_attention_2: true
```

2. **增大 Batch Size**
```yaml
per_device_train_batch_size: 6  # RTX 5090 可以支持
gradient_accumulation_steps: 2
```

3. **使用 BF16**（RTX 5090 支持）
```yaml
bf16: true
fp16: false
```

4. **使用 Fused Optimizer**
```yaml
optim: adamw_torch_fused
```

### 节省显存

1. **使用梯度检查点**
```yaml
gradient_checkpointing: true
```

2. **减小序列长度**（如果可能）
```yaml
cutoff_len: 1536  # 从 2048 减小
```

3. **使用 QLoRA**
```yaml
quantization_bit: 4
```

---

## ❗ 常见问题

### Q1: 连接断开后训练会停止吗？

**A**: 如果使用 `nohup` 或 `tmux`，不会停止。推荐使用 tmux。

### Q2: 如何暂停和恢复训练？

**A**: LLaMA-Factory 支持断点续训：
```bash
# 从检查点恢复
--resume_from_checkpoint /data_nvme/react_training/output/qwen2.5-7b-react/checkpoint-XXX
```

### Q3: 显存溢出怎么办？

**A**: 减小 batch size 或使用梯度累积：
```yaml
per_device_train_batch_size: 2  # 减小
gradient_accumulation_steps: 8  # 增大
```

### Q4: 训练太慢？

**A**: 
1. 检查是否安装 Flash Attention
2. 使用 BF16
3. 增大 batch size
4. 确保使用 NVMe 存储

---

## 🎯 完整命令速查表

```bash
# 连接服务器
ssh username@server-ip

# 激活环境
source /data_nvme/react_training/react_env/bin/activate

# 查看 GPU
nvidia-smi

# 开始训练（tmux）
tmux new -s react_training
./scripts/train.sh

# 分离 tmux
Ctrl+B, D

# 重新连接
tmux attach -t react_training

# 查看日志
tail -f /data_nvme/react_training/logs/train.log

# 监控 GPU
watch -n 1 nvidia-smi

# 查看训练进度
ls -lh /data_nvme/react_training/output/qwen2.5-7b-react/

# 启动 TensorBoard
tensorboard --logdir=/data_nvme/react_training/logs --host=0.0.0.0 --port=6006

# 下载模型
rsync -avz --progress username@server-ip:/data_nvme/react_training/merged_model ./
```

---

祝训练顺利！🎉

