# ReAct 模型训练完整指南

## 📋 目录

1. [环境准备](#环境准备)
2. [数据准备](#数据准备)
3. [模型训练](#模型训练)
4. [模型评估](#模型评估)
5. [模型部署到 Ollama](#模型部署到-ollama)
6. [常见问题](#常见问题)

---

## 🛠️ 环境准备

### 硬件要求

| 模型大小 | 训练方法 | 显存需求 | 推荐配置 |
|---------|---------|---------|---------|
| Qwen2.5-0.5B | LoRA | 4-6 GB | GTX 1660 Ti |
| Qwen2.5-3B | LoRA | 8-12 GB | RTX 3060 |
| Qwen2.5-7B | LoRA | 16-20 GB | RTX 4060 Ti 16GB |
| Qwen2.5-7B | QLoRA | 10-14 GB | RTX 3060 12GB |
| Qwen2.5-14B | QLoRA | 18-24 GB | RTX 4090 |

### 软件安装

#### 方法 1：使用 LLaMA-Factory（推荐新手）

```bash
# 1. 克隆仓库
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory

# 2. 安装依赖
pip install -e .

# 3. 启动 Web UI（可选）
llamafactory-cli webui
# 打开浏览器访问 http://localhost:7860
```

#### 方法 2：使用 Unsloth（推荐追求速度）

```bash
# 安装 Unsloth
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
```

#### 方法 3：使用 Axolotl（推荐高级用户）

```bash
git clone https://github.com/OpenAccess-AI-Collective/axolotl
cd axolotl
pip install packaging
pip install -e '.[flash-attn,deepspeed]'
```

---

## 📊 数据准备

### Step 1: 运行数据转换脚本

```bash
python prepare_training_data.py
```

这将生成：
- `react_train_alpaca.json` - Alpaca 格式训练集
- `react_test_alpaca.json` - Alpaca 格式测试集
- `react_train_sharegpt.json` - ShareGPT 格式训练集
- `react_test_sharegpt.json` - ShareGPT 格式测试集

### Step 2: 配置 LLaMA-Factory 数据集

创建 `LLaMA-Factory/data/dataset_info.json`，添加：

```json
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
```

将生成的 JSON 文件复制到 `LLaMA-Factory/data/` 目录：

```bash
cp react_*.json LLaMA-Factory/data/
```

---

## 🚀 模型训练

### 方法 A：使用 LLaMA-Factory Web UI（最简单）

1. 启动 Web UI：
```bash
cd LLaMA-Factory
llamafactory-cli webui
```

2. 在浏览器中配置：
   - **模型名称**: Qwen/Qwen2.5-7B-Instruct
   - **微调方法**: lora
   - **数据集**: react_training
   - **学习率**: 5e-5
   - **训练轮数**: 3
   - **LoRA rank**: 16

3. 点击"开始训练"

### 方法 B：使用命令行训练

```bash
cd LLaMA-Factory

# LoRA 训练
llamafactory-cli train \
    --stage sft \
    --do_train \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --dataset react_training \
    --template qwen \
    --finetuning_type lora \
    --lora_target all \
    --output_dir ./saves/qwen2.5-7b-react \
    --overwrite_cache \
    --overwrite_output_dir \
    --cutoff_len 2048 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
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
    --fp16
```

### 方法 C：使用 Unsloth（最快）

创建 `train_with_unsloth.py`:

```python
from unsloth import FastLanguageModel
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset

# 1. 加载模型
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B-Instruct",
    max_seq_length=2048,
    dtype=None,  # 自动检测
    load_in_4bit=True,  # 使用 4-bit 量化
)

# 2. 配置 LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)

# 3. 加载数据
dataset = load_dataset("json", data_files={
    "train": "react_train_alpaca.json",
    "test": "react_test_alpaca.json"
})

# 4. 格式化数据
def formatting_func(examples):
    texts = []
    for instruction, input_text, output in zip(
        examples["instruction"],
        examples["input"],
        examples["output"]
    ):
        text = f"### Instruction:\n{instruction}\n\n"
        if input_text:
            text += f"### Input:\n{input_text}\n\n"
        text += f"### Response:\n{output}"
        texts.append(text)
    return {"text": texts}

train_dataset = dataset["train"].map(formatting_func, batched=True)
eval_dataset = dataset["test"].map(formatting_func, batched=True)

# 5. 训练
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=2048,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_ratio=0.1,
        num_train_epochs=3,
        learning_rate=5e-5,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir="./output/qwen2.5-7b-react",
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=100,
    ),
)

trainer.train()

# 6. 保存模型
model.save_pretrained("./output/qwen2.5-7b-react-final")
tokenizer.save_pretrained("./output/qwen2.5-7b-react-final")
```

运行训练：
```bash
python train_with_unsloth.py
```

---

## 📈 模型评估

### 使用 LLaMA-Factory 评估

```bash
llamafactory-cli train \
    --stage sft \
    --do_eval \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --adapter_name_or_path ./saves/qwen2.5-7b-react \
    --dataset react_testing \
    --template qwen \
    --finetuning_type lora \
    --output_dir ./evaluation \
    --per_device_eval_batch_size 2 \
    --predict_with_generate
```

### 手动测试

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 加载基础模型
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)

# 加载 LoRA 适配器
model = PeftModel.from_pretrained(base_model, "./saves/qwen2.5-7b-react")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# 测试
test_input = """You are in the middle of a room. Looking quickly around you, you see a cabinet 10, a cabinet 9, a cabinet 8, a cabinet 7, a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 3, a countertop 2, a countertop 1, a diningtable 1, a drawer 6, a drawer 5, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.
Your task is to: heat some potato and put it in diningtable.

Please solve this task using ReAct (Reasoning + Acting) approach."""

inputs = tokenizer(test_input, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=1024)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(response)
```

---

## 🔄 模型部署到 Ollama

### Step 1: 合并 LoRA 权重

```bash
cd LLaMA-Factory

# 合并 LoRA 到基础模型
llamafactory-cli export \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --adapter_name_or_path ./saves/qwen2.5-7b-react \
    --template qwen \
    --finetuning_type lora \
    --export_dir ./merged_model \
    --export_size 2 \
    --export_device cpu \
    --export_legacy_format False
```

### Step 2: 转换为 GGUF 格式（Ollama 需要）

```bash
# 安装 llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make

# 转换为 GGUF
python convert.py /path/to/merged_model --outfile qwen2.5-7b-react.gguf --outtype f16

# 量化（可选，减小模型大小）
./quantize qwen2.5-7b-react.gguf qwen2.5-7b-react-q4_k_m.gguf Q4_K_M
```

### Step 3: 创建 Modelfile

创建 `Modelfile`:

```dockerfile
FROM ./qwen2.5-7b-react-q4_k_m.gguf

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
{{ end }}{{ .Response }}<|im_end|>
"""

PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.7
PARAMETER top_p 0.8
PARAMETER top_k 20

SYSTEM """You are a helpful AI assistant trained to solve tasks using ReAct (Reasoning + Acting) approach. You think step by step, take actions, and observe results."""
```

### Step 4: 导入到 Ollama

```bash
ollama create qwen2.5-react -f Modelfile
```

### Step 5: 测试

```bash
ollama run qwen2.5-react

>>> You are in the middle of a room. Your task is to: heat some egg and put it in diningtable. Please solve this.
```

---

## ❓ 常见问题

### Q1: 显存不足怎么办？

**方案 1**: 使用 QLoRA（4-bit 量化）
```yaml
quantization_bit: 4  # 在配置中添加
```

**方案 2**: 减小批次大小和序列长度
```yaml
per_device_train_batch_size: 1
cutoff_len: 1024
```

**方案 3**: 使用梯度检查点
```yaml
gradient_checkpointing: true
```

**方案 4**: 选择更小的模型（Qwen2.5-3B 或 0.5B）

### Q2: 训练速度太慢？

1. **使用 Unsloth**（提速 2-5 倍）
2. **使用 Flash Attention 2**
```bash
pip install flash-attn --no-build-isolation
```
3. **减少序列长度**（如果示例允许）
4. **使用多 GPU**（DDP/FSDP）

### Q3: 模型过拟合？

1. **增加 Dropout**
```yaml
lora_dropout: 0.1
```
2. **使用权重衰减**
```yaml
weight_decay: 0.01
```
3. **减少训练轮数**
```yaml
num_train_epochs: 2
```
4. **增加训练数据**（生成更多示例）

### Q4: 如何监控训练？

使用 TensorBoard:
```bash
tensorboard --logdir ./logs
```

或使用 Weights & Biases:
```bash
pip install wandb
wandb login
# 在配置中设置 report_to: wandb
```

### Q5: 如何选择合适的学习率？

- **LoRA 微调**: 1e-4 到 5e-5
- **QLoRA 微调**: 1e-4 到 2e-4
- **全量微调**: 1e-5 到 5e-6

建议使用学习率预热（warmup_ratio: 0.1）

---

## 📚 参考资源

- **LLaMA-Factory**: https://github.com/hiyouga/LLaMA-Factory
- **Unsloth**: https://github.com/unslothai/unsloth
- **Axolotl**: https://github.com/OpenAccess-AI-Collective/axolotl
- **Qwen 模型**: https://huggingface.co/Qwen
- **Ollama**: https://ollama.ai/

---

## 💡 训练建议

1. **从小模型开始**：先用 Qwen2.5-3B 测试流程
2. **使用验证集**：监控过拟合情况
3. **保存检查点**：定期保存，防止训练中断
4. **记录实验**：使用 wandb 或 tensorboard
5. **测试生成质量**：不只看 loss，要实际测试生成效果

祝训练顺利！🎉

