"""
合并生成的示例文件
将多个独立的JSON文件合并成一个大的JSON文件，方便使用
"""

import json
import os
from pathlib import Path

def merge_samples(input_dir, output_file, format_type='unified'):
    """
    合并所有生成的示例文件
    
    Args:
        input_dir: 包含生成示例的目录
        output_file: 输出文件路径
        format_type: 输出格式
            - 'unified': 统一的大JSON对象
            - 'list': JSON数组格式
            - 'jsonl': 每行一个JSON对象
    """
    print(f"开始合并示例文件...")
    print(f"输入目录: {input_dir}")
    print(f"输出文件: {output_file}")
    print(f"输出格式: {format_type}")
    print("-"*60)
    
    # 查找所有生成的示例文件
    sample_files = sorted([f for f in os.listdir(input_dir) 
                          if f.startswith('generated_react_put_') and f.endswith('.json')])
    
    print(f"找到 {len(sample_files)} 个示例文件")
    
    if len(sample_files) == 0:
        print("错误：未找到任何示例文件")
        return
    
    samples = []
    merged_data = {}
    success_count = 0
    error_count = 0
    
    # 读取所有示例文件
    for i, filename in enumerate(sample_files, 1):
        filepath = os.path.join(input_dir, filename)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                # 提取示例内容
                for key, value in data.items():
                    if key != 'metadata':
                        # 检查是否成功生成
                        metadata = data.get('metadata', {})
                        if metadata.get('success', True):
                            if format_type == 'list':
                                samples.append({
                                    'id': key,
                                    'content': value,
                                    'metadata': metadata
                                })
                            elif format_type == 'jsonl':
                                samples.append({
                                    key: value
                                })
                            else:  # unified
                                merged_data[key] = value
                            success_count += 1
                        else:
                            error_count += 1
                            print(f"  跳过失败的示例: {filename}")
        
        except Exception as e:
            print(f"  读取文件 {filename} 失败: {e}")
            error_count += 1
        
        if i % 10 == 0:
            print(f"  进度: {i}/{len(sample_files)}")
    
    # 保存合并后的文件
    print(f"\n保存合并文件...")
    
    try:
        if format_type == 'jsonl':
            # JSONL 格式：每行一个JSON对象
            with open(output_file, 'w', encoding='utf-8') as f:
                for sample in samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        else:
            # JSON 格式
            output_data = samples if format_type == 'list' else merged_data
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"✓ 保存成功!")
        
    except Exception as e:
        print(f"✗ 保存失败: {e}")
        return
    
    # 统计信息
    print("\n" + "="*60)
    print("合并完成！")
    print("="*60)
    print(f"总文件数: {len(sample_files)}")
    print(f"成功合并: {success_count}")
    print(f"失败/跳过: {error_count}")
    print(f"输出文件: {output_file}")
    
    # 文件大小
    file_size = os.path.getsize(output_file)
    if file_size > 1024 * 1024:
        print(f"文件大小: {file_size / 1024 / 1024:.2f} MB")
    else:
        print(f"文件大小: {file_size / 1024:.2f} KB")
    print("="*60)


def create_training_dataset(input_dir, output_file):
    """
    创建训练数据集格式
    适合用于模型微调
    """
    print(f"创建训练数据集...")
    
    sample_files = sorted([f for f in os.listdir(input_dir) 
                          if f.startswith('generated_react_put_') and f.endswith('.json')])
    
    training_data = []
    
    for filename in sample_files:
        filepath = os.path.join(input_dir, filename)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
                for key, value in data.items():
                    if key != 'metadata':
                        metadata = data.get('metadata', {})
                        if metadata.get('success', True):
                            # 构造训练样本
                            # 可以根据需要调整格式
                            training_sample = {
                                "instruction": "Generate a ReAct task example for putting an object in a location.",
                                "input": "",
                                "output": value
                            }
                            training_data.append(training_sample)
        
        except Exception as e:
            print(f"  处理文件 {filename} 失败: {e}")
    
    # 保存训练数据集
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 训练数据集创建完成")
    print(f"  样本数: {len(training_data)}")
    print(f"  输出文件: {output_file}")


if __name__ == "__main__":
    # 配置
    INPUT_DIR = "generated_samples"
    
    print("="*60)
    print("示例文件合并工具")
    print("="*60)
    print()
    
    # 1. 合并为统一格式（推荐用于后续生成）
    print("[任务1] 合并为统一JSON格式...")
    merge_samples(
        input_dir=INPUT_DIR,
        output_file="generated_samples_merged.json",
        format_type='unified'
    )
    print()
    
    # 2. 合并为列表格式（方便查看和处理）
    print("[任务2] 合并为JSON数组格式...")
    merge_samples(
        input_dir=INPUT_DIR,
        output_file="generated_samples_list.json",
        format_type='list'
    )
    print()
    
    # 3. 合并为JSONL格式（方便大规模数据处理）
    print("[任务3] 合并为JSONL格式...")
    merge_samples(
        input_dir=INPUT_DIR,
        output_file="generated_samples.jsonl",
        format_type='jsonl'
    )
    print()
    
    # 4. 创建训练数据集
    print("[任务4] 创建训练数据集...")
    create_training_dataset(
        input_dir=INPUT_DIR,
        output_file="training_dataset.json"
    )
    print()
    
    print("="*60)
    print("所有任务完成！")
    print("="*60)
    print("\n生成的文件：")
    print("  1. generated_samples_merged.json - 统一JSON格式")
    print("  2. generated_samples_list.json - JSON数组格式")
    print("  3. generated_samples.jsonl - JSONL格式")
    print("  4. training_dataset.json - 训练数据集格式")
    print()

