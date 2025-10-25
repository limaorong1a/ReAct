import datetime
import urllib.parse
import base64
import hmac
import hashlib
import requests
import uuid
import json
import time
import csv
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue

# 全局锁，用于线程安全的文件写入
file_lock = threading.Lock()

def get_sign_params(method: str, url: str, url_params: dict[str, str] | None,
                     access_key_id: str, access_key_secret: str):
    """
    获取签名参数
    """
    url = url.lstrip('https://').lstrip('http://')
    sign_params = {
        **(url_params or {}),
        'AccessKeyId': access_key_id,
        'Expires': 60,
        'Timestamp': datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    }
    sorted_sign_params_str = urllib.parse.urlencode({key: sign_params[key] for key in sorted(sign_params.keys())})
    sign_str = f'{method.upper()}{url}?{sorted_sign_params_str}'
    sign = urllib.parse.quote(
        base64.b64encode(
            hmac.new(
                access_key_secret.encode('utf-8'),
                sign_str.encode('utf-8'),
                hashlib.sha1
            ).digest()), safe='')
    return {
        'AccessKeyId': sign_params.get('AccessKeyId'),
        'Expires': sign_params.get('Expires'),
        'Timestamp': sign_params.get('Timestamp'),
        'Signature': sign
    }


def create_conversation(access_key_id: str, access_key_secret: str, 
                       agent_id: str, user: str, inputs: dict = None,
                       base_url: str = "https://api-bj.clink.cn", max_retries: int = 3):
    """
    创建会话
    Args:
        access_key_id: 访问密钥ID
        access_key_secret: 访问密钥密文
        agent_id: 智能体应用ID
        user: 用户唯一标识
        inputs: 随路参数，默认为空字典
        base_url: API基础URL
        max_retries: 最大重试次数，默认为3次
    Returns:
        会话创建结果，包含conversation_id等信息
    """
    # API路径
    api_path = "/agent/v1/create-conversation"
    url = f"{base_url}{api_path}"
    
    # 构造请求体
    request_body = {
        "agent_id": agent_id,
        "user": user,
        "inputs": inputs or {},
        "created_at": int(datetime.datetime.now().timestamp() * 1000)
    }

    for attempt in range(max_retries + 1):
        try:
            # 获取签名参数
            sign_params = get_sign_params(
                method="POST",
                url=url,
                url_params=None,
                access_key_id=access_key_id,
                access_key_secret=access_key_secret
            )

            # 构造完整的请求URL(带签名参数)
            request_url = f"{url}?{'&'.join([f'{k}={v}' for k,v in sign_params.items()])}"

            # 发送请求
            headers = {
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                url=request_url,
                headers=headers,
                json=request_body
            )

            result = response.json()
            
            # 检查是否为请求超限错误
            if 'error' in result and result['error'].get('code') == 'TooManyRequests':
                if attempt < max_retries:
                    print(f"请求超限，等待10秒后重试... (第{attempt + 1}次重试)")
                    time.sleep(10)
                    continue
                else:
                    print(f"请求超限，已达到最大重试次数({max_retries})，返回错误结果")
                    return result
            
            # 如果没有错误或者是其他错误，直接返回结果
            return result
            
        except Exception as e:
            if attempt < max_retries:
                print(f"请求异常，等待10秒后重试... (第{attempt + 1}次重试): {e}")
                time.sleep(10)
                continue
            else:
                print(f"请求异常，已达到最大重试次数({max_retries}): {e}")
                raise e
    
    # 理论上不会执行到这里
    return {"error": {"code": "MaxRetriesExceeded", "message": "已达到最大重试次数"}}


def send_chat_message_stream(access_key_id: str, access_key_secret: str, 
                            agent_id: str, conversation_id: str, message: str, 
                            user: str, inputs: dict = None,
                            base_url: str = "https://api-bj.clink.cn"):
    """
    发送对话消息（流式输出）
    Args:
        access_key_id: 访问密钥ID
        access_key_secret: 访问密钥密文
        agent_id: 智能体应用ID
        conversation_id: 会话ID
        message: 消息内容
        user: 用户唯一标识
        inputs: 随路参数，默认为空字典
        base_url: API基础URL
    Returns:
        流式响应生成器，包含时间统计信息
    """
    # API路径
    api_path = "/agent/v1/chat-messages"
    url = f"{base_url}{api_path}"
    
    # 构造请求体
    request_body = {
        "agent_id": agent_id,
        "conversation_id": conversation_id,
        "user": user,
        "query": [
            {
                "content": message,
                "content_type": "text",
                "created_at": int(datetime.datetime.now().timestamp() * 1000)
            }
        ],
        "inputs": inputs or {},
        "response_mode": "blocking"  # 流式响应
    }

    # 获取签名参数
    sign_params = get_sign_params(
        method="POST",
        url=url,
        url_params=None,
        access_key_id=access_key_id,
        access_key_secret=access_key_secret
    )

    # 构造完整的请求URL(带签名参数)
    request_url = f"{url}?{'&'.join([f'{k}={v}' for k,v in sign_params.items()])}"

    # 发送请求
    headers = {
        "Content-Type": "application/json"
    }
    
    content = "报错"

    try:
        response = requests.post(
            url=request_url,
            headers=headers,
            json=request_body
        )
        
        res = response.json()
        content = res.get("answer")[0].get("content")
    except Exception as e:
        print(e)
    
    return content

def process_single_question(question_data, config):
    """
    处理单个问题的函数
    Args:
        question_data: (index, question) 元组
        config: 配置字典，包含API密钥等信息
    Returns:
        处理结果字典
    """
    index, user_message = question_data
    thread_id = threading.current_thread().ident
    
    # print(f"[线程{thread_id}] 处理第 {index} 个问题: {user_message[:50]}...")
    
    conversation_id = ""
    result = ""
    
    try:
        # 1. 创建会话
        conversation_result = create_conversation(
            access_key_id=config['ACCESS_KEY_ID'],
            access_key_secret=config['ACCESS_KEY_SECRET'],
            agent_id=config['AGENT_ID'],
            user=config['USER']
        )
        conversation_id = conversation_result['conversation_id']
        # print(f"[线程{thread_id}] 会话创建成功，conversation_id: {conversation_id}")

        # 2. 发送消息并获取结果
        result = send_chat_message_stream(
            access_key_id=config['ACCESS_KEY_ID'],
            access_key_secret=config['ACCESS_KEY_SECRET'],
            agent_id=config['AGENT_ID'],
            conversation_id=conversation_id,
            message=user_message,
            user=config['USER']
        )
        
        # print(f"[线程{thread_id}] 问题 {index} 处理完成")
        
    except Exception as e:
        print(f"[线程{thread_id}] 处理问题 {index} 失败: {e}")
        result = f'错误: {str(e)}'
        print(e)
    
    return {
        'index': index,
        '会话ID': conversation_id,
        '用户问题': user_message,
        '命中结果': result
    }

def write_result_to_csv(result_data, result_file_path):
    """
    线程安全地写入结果到CSV文件
    """
    with file_lock:
        try:
            with open(result_file_path, 'a', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['会话ID', '用户问题', '命中结果']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writerow({
                    '会话ID': result_data['会话ID'],
                    '用户问题': result_data['用户问题'],
                    '命中结果': result_data['命中结果']
                })
                csvfile.flush()  # 立即刷新到磁盘
            return True
        except Exception as e:
            print(f"写入CSV文件失败: {e}")
            return False

# 使用示例
if __name__ == "__main__":
    # 配置参数
    config = {
        'ACCESS_KEY_ID': "fe3006cf3d9d466f5211247183817aaec",
        'ACCESS_KEY_SECRET': "tSlLp5N41jT1tV25vX2t9",
        'AGENT_ID': "1-778978de-6fbc-4716-bc27-c20fb849d2b3",
        'USER': "yinshu"
    }

    # 文件路径
    question_file_path = "/Users/tzc/Documents/workspace/cursor/test/data_yinshu/data/user_question.txt"
    result_file_path = "/Users/tzc/Documents/workspace/cursor/test/data_yinshu/data/result.csv"
    
    # 线程池配置
    MAX_WORKERS = 8 # 可以根据API限制调整线程数
    
    # 读取问题文件
    questions = []
    try:
        with open(question_file_path, 'r', encoding='utf-8') as f:
            questions = [line.strip() for line in f.readlines() if line.strip()]
        print(f"成功读取 {len(questions)} 个问题")
    except Exception as e:
        print(f"读取问题文件失败: {e}")
        exit(1)

    # 创建CSV文件并写入表头
    try:
        with open(result_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['会话ID', '用户问题', '命中结果']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            csvfile.flush()  # 立即刷新到磁盘
        print(f"CSV文件已创建: {result_file_path}")
    except Exception as e:
        print(f"创建CSV文件失败: {e}")
        exit(1)
    
    # 使用线程池处理问题
    processed_count = 0
    failed_count = 0
    start_time = time.time()
    
    print(f"开始使用 {MAX_WORKERS} 个线程处理 {len(questions)} 个问题...")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        future_to_question = {
            executor.submit(process_single_question, (i+1, question), config): (i+1, question)
            for i, question in enumerate(questions)
        }
        
        # 处理完成的任务
        for future in as_completed(future_to_question):
            try:
                result_data = future.result()
                
                # 写入结果到CSV
                if write_result_to_csv(result_data, result_file_path):
                    processed_count += 1
                    print(f"进度: {processed_count}/{len(questions)} ({processed_count/len(questions)*100:.1f}%)")
                else:
                    failed_count += 1
                    
            except Exception as e:
                failed_count += 1
                question_index, question_text = future_to_question[future]
                print(f"任务执行失败 (问题 {question_index}): {e}")
                
                # 即使任务失败也要记录
                error_result = {
                    'index': question_index,
                    '会话ID': '',
                    '用户问题': question_text,
                    '命中结果': f'任务执行失败: {str(e)}'
                }
                write_result_to_csv(error_result, result_file_path)

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\n=== 处理完成 ===")
    print(f"总问题数: {len(questions)}")
    print(f"成功处理: {processed_count}")
    print(f"失败数量: {failed_count}")
    print(f"总耗时: {elapsed_time:.2f} 秒")
    print(f"平均每个问题: {elapsed_time/len(questions):.2f} 秒")
    print(f"结果已保存到: {result_file_path}")
            

            






