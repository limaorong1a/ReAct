"""
将生成的 ReAct 示例转换为训练格式
支持 LLaMA-Factory 和 Unsloth 格式
"""

import json
import os
from pathlib import Path
from typing import List, Dict

def load_all_samples(base_dir: str = ".") -> List[Dict]:
    """加载所有任务类型的示例"""
    
    task_types = [
        "generated_samples_put",
        "generated_samples_clean", 
        "generated_samples_cool",
        "generated_samples_heat",
        "generated_samples_puttwo",
        "generated_samples_examine"
    ]
    
    all_samples = []
    
    for task_type in task_types:
        task_dir = Path(base_dir) / task_type
        if not task_dir.exists():
            print(f"⚠️  目录不存在: {task_dir}")
            continue
            
        # 读取所有 sample_*.json 文件
        sample_files = sorted(task_dir.glob("sample_*.json"))
        
        for sample_file in sample_files:
            try:
                with open(sample_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    all_samples.append({
                        "task_type": task_type,
                        "file": str(sample_file),
                        "data": data
                    })
            except Exception as e:
                print(f"❌ 读取失败 {sample_file}: {e}")
    
    print(f"✅ 成功加载 {len(all_samples)} 个示例")
    return all_samples


def convert_to_llama_factory_format(samples: List[Dict]) -> List[Dict]:
    """转换为 LLaMA-Factory 格式（Alpaca/ShareGPT）"""
    
    training_data = []
    
    for idx, sample in enumerate(samples):
        # 提取示例内容
        data = sample["data"]
        
        # 通常 JSON 格式是: {"task_name": "完整示例文本"}
        for key, value in data.items():
            if key.startswith("generated_") or key.startswith("react_"):
                example_text = value
                
                # 分离任务描述和执行过程
                lines = example_text.strip().split('\n')
                
                # 提取环境描述和任务
                environment = lines[0] if lines else ""
                task = lines[1] if len(lines) > 1 else ""
                
                # 提取 ReAct 执行过程（从第三行开始）
                react_process = '\n'.join(lines[2:]) if len(lines) > 2 else ""
                
                # 构建训练样本
                training_sample = {
                    "instruction": f"{environment}\n{task}\n\nPlease solve this task using ReAct (Reasoning + Acting) approach. Think step by step, take actions, and observe the results.",
                    "input": "",
                    "output": react_process
                }
                
                training_data.append(training_sample)
                break
    
    return training_data


def convert_to_sharegpt_format(samples: List[Dict]) -> List[Dict]:
    """转换为 ShareGPT 格式（推荐用于对话模型）"""
    
    training_data = []
    
    for sample in samples:
        data = sample["data"]
        
        for key, value in data.items():
            if key.startswith("generated_") or key.startswith("react_"):
                example_text = value
                lines = example_text.strip().split('\n')
                
                environment = lines[0] if lines else ""
                task = lines[1] if len(lines) > 1 else ""
                react_process = '\n'.join(lines[2:]) if len(lines) > 2 else ""
                
                # ShareGPT 格式
                conversation = {
                    "conversations": [
                        {
                            "from": "human",
                            "value": f"{environment}\n{task}\n\nPlease solve this task using ReAct (Reasoning + Acting) approach."
                        },
                        {
                            "from": "gpt",
                            "value": react_process
                        }
                    ]
                }
                
                training_data.append(conversation)
                break
    
    return training_data


def split_train_test(data: List[Dict], test_ratio: float = 0.1):
    """划分训练集和测试集"""
    
    import random
    random.shuffle(data)
    
    split_idx = int(len(data) * (1 - test_ratio))
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    return train_data, test_data


def main():
    print("🚀 开始准备训练数据...")
    
    # 1. 加载所有示例
    samples = load_all_samples()
    
    if not samples:
        print("❌ 未找到任何示例数据！")
        return
    
    # 2. 转换为 Alpaca 格式（适合指令微调）
    print("\n📝 转换为 Alpaca 格式...")
    alpaca_data = convert_to_llama_factory_format(samples)
    train_alpaca, test_alpaca = split_train_test(alpaca_data)
    
    with open("react_train_alpaca.json", 'w', encoding='utf-8') as f:
        json.dump(train_alpaca, f, ensure_ascii=False, indent=2)
    
    with open("react_test_alpaca.json", 'w', encoding='utf-8') as f:
        json.dump(test_alpaca, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Alpaca 格式: 训练集 {len(train_alpaca)} 条, 测试集 {len(test_alpaca)} 条")
    
    # 3. 转换为 ShareGPT 格式（适合对话模型）
    print("\n💬 转换为 ShareGPT 格式...")
    sharegpt_data = convert_to_sharegpt_format(samples)
    train_sharegpt, test_sharegpt = split_train_test(sharegpt_data)
    
    with open("react_train_sharegpt.json", 'w', encoding='utf-8') as f:
        json.dump(train_sharegpt, f, ensure_ascii=False, indent=2)
    
    with open("react_test_sharegpt.json", 'w', encoding='utf-8') as f:
        json.dump(test_sharegpt, f, ensure_ascii=False, indent=2)
    
    print(f"✅ ShareGPT 格式: 训练集 {len(train_sharegpt)} 条, 测试集 {len(test_sharegpt)} 条")
    
    # 4. 生成数据统计
    print("\n📊 数据统计:")
    print(f"  - 总样本数: {len(samples)}")
    print(f"  - 训练样本: {len(train_alpaca)}")
    print(f"  - 测试样本: {len(test_alpaca)}")
    
    # 统计任务类型分布
    task_counts = {}
    for sample in samples:
        task_type = sample["task_type"]
        task_counts[task_type] = task_counts.get(task_type, 0) + 1
    
    print("\n📋 任务类型分布:")
    for task_type, count in sorted(task_counts.items()):
        print(f"  - {task_type}: {count} 个")
    
    print("\n✅ 数据准备完成！")
    print("\n下一步:")
    print("1. 使用 LLaMA-Factory 训练: 将 JSON 文件放到 LLaMA-Factory/data/ 目录")
    print("2. 或使用 Unsloth/Axolotl 训练")


if __name__ == "__main__":
    main()

