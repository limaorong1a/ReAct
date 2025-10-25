"""
ReAct 示例批量生成工具
使用 Dify 智能体 API 批量生成 ReAct 任务示例
"""

import datetime
import urllib.parse
import base64
import hmac
import hashlib
import requests
import json
import time
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# 全局锁
file_lock = threading.Lock()


def get_sign_params(method: str, url: str, url_params: dict[str, str] | None,
                     access_key_id: str, access_key_secret: str):
    """获取签名参数"""
    url = url.lstrip('https://').lstrip('http://')
    sign_params = {
        **(url_params or {}),
        'AccessKeyId': access_key_id,
        'Expires': 60,
        'Timestamp': datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')
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
    """创建会话"""
    api_path = "/agent/v1/create-conversation"
    url = f"{base_url}{api_path}"
    
    request_body = {
        "agent_id": agent_id,
        "user": user,
        "inputs": inputs or {},
        "created_at": int(datetime.datetime.now().timestamp() * 1000)
    }

    for attempt in range(max_retries + 1):
        try:
            sign_params = get_sign_params(
                method="POST",
                url=url,
                url_params=None,
                access_key_id=access_key_id,
                access_key_secret=access_key_secret
            )

            request_url = f"{url}?{'&'.join([f'{k}={v}' for k,v in sign_params.items()])}"
            headers = {"Content-Type": "application/json"}
            response = requests.post(url=request_url, headers=headers, json=request_body)
            result = response.json()
            
            if 'error' in result and result['error'].get('code') == 'TooManyRequests':
                if attempt < max_retries:
                    print(f"⚠ 请求超限，等待10秒后重试... (第{attempt + 1}次)")
                    time.sleep(10)
                    continue
            
            return result
            
        except Exception as e:
            if attempt < max_retries:
                print(f"⚠ 请求异常，等待10秒后重试... (第{attempt + 1}次): {e}")
                time.sleep(10)
                continue
            else:
                raise e
    
    return {"error": {"code": "MaxRetriesExceeded", "message": "已达到最大重试次数"}}


def send_chat_message(access_key_id: str, access_key_secret: str, 
                     agent_id: str, conversation_id: str, message: str, 
                     user: str, inputs: dict = None,
                     base_url: str = "https://api-bj.clink.cn"):
    """发送对话消息"""
    api_path = "/agent/v1/chat-messages"
    url = f"{base_url}{api_path}"
    
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
        "response_mode": "blocking"
    }

    sign_params = get_sign_params(
        method="POST",
        url=url,
        url_params=None,
        access_key_id=access_key_id,
        access_key_secret=access_key_secret
    )

    request_url = f"{url}?{'&'.join([f'{k}={v}' for k,v in sign_params.items()])}"
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url=request_url, headers=headers, json=request_body)
        res = response.json()
        content = res.get("answer", [{}])[0].get("content", "")
        return content
    except Exception as e:
        print(f"⚠ 发送消息失败: {e}")
        return f"错误: {str(e)}"


def generate_single_sample(sample_id, config, user_message):
    """生成单个示例"""
    thread_id = threading.current_thread().ident
    
    try:
        # 创建会话
        conversation_result = create_conversation(
            access_key_id=config['ACCESS_KEY_ID'],
            access_key_secret=config['ACCESS_KEY_SECRET'],
            agent_id=config['AGENT_ID'],
            user=config['USER']
        )
        
        conversation_id = conversation_result.get('conversation_id', '')
        if not conversation_id:
            raise Exception("创建会话失败")
        
        # 发送消息生成示例
        generated_content = send_chat_message(
            access_key_id=config['ACCESS_KEY_ID'],
            access_key_secret=config['ACCESS_KEY_SECRET'],
            agent_id=config['AGENT_ID'],
            conversation_id=conversation_id,
            message=user_message,
            user=config['USER']
        )
        
        return {
            'sample_id': sample_id,
            'conversation_id': conversation_id,
            'content': generated_content,
            'success': True
        }
        
    except Exception as e:
        print(f"✗ [线程{thread_id}] 生成示例 {sample_id} 失败: {e}")
        return {
            'sample_id': sample_id,
            'conversation_id': '',
            'content': f'错误: {str(e)}',
            'success': False
        }


def save_sample(sample_data, output_dir):
    """保存单个示例"""
    with file_lock:
        try:
            sample_id = sample_data['sample_id']
            file_path = os.path.join(output_dir, f'sample_{sample_id:03d}.json')
            
            json_data = {
                f"generated_react_put_{sample_id}": sample_data['content'],
                "metadata": {
                    "conversation_id": sample_data['conversation_id'],
                    "generated_at": datetime.datetime.now().isoformat(),
                    "success": sample_data['success']
                }
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            
            return True
        except Exception as e:
            print(f"✗ 保存示例 {sample_data['sample_id']} 失败: {e}")
            return False


if __name__ == "__main__":
    print("=" * 70)
    print("ReAct 示例批量生成工具")
    print("=" * 70)
    
    # ========== 配置区域 ==========
    config = {
        'ACCESS_KEY_ID': "79c536f734569ae28ab2328b1f6abf9d",
        'ACCESS_KEY_SECRET': "0b4nUUD0nd4l4hr03p1s",
        'AGENT_ID': "1-d1d5bf70-cb5e-4701-94fa-bf274a7925e7",
        'USER': "yinshu"
    }
    
    # 生成参数
    TOTAL_SAMPLES = 100       # 生成30个示例
    MAX_WORKERS = 3          # 并发线程数
    output_dir = "generated_samples_heat"
    
    # 用户消息（触发生成）
    USER_MESSAGE = "请生成一个新的 ReAct 任务示例"
    # ========== 配置区域结束 ==========
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n配置信息:")
    print(f"  Agent ID: {config['AGENT_ID']}")
    print(f"  生成数量: {TOTAL_SAMPLES}")
    print(f"  并发线程: {MAX_WORKERS}")
    print(f"  输出目录: {output_dir}/")
    print(f"  触发消息: \"{USER_MESSAGE}\"")
    print("-" * 70)
    
    # 开始生成
    start_time = time.time()
    success_count = 0
    failed_count = 0
    
    print(f"\n开始生成...\n")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        futures = {
            executor.submit(generate_single_sample, i+1, config, USER_MESSAGE): i+1
            for i in range(TOTAL_SAMPLES)
        }
        
        # 处理完成的任务
        for future in as_completed(futures):
            try:
                sample_data = future.result()
                
                # 保存到文件
                if save_sample(sample_data, output_dir):
                    if sample_data['success']:
                        success_count += 1
                        print(f"✓ [{success_count + failed_count:2d}/{TOTAL_SAMPLES}] "
                              f"示例 {sample_data['sample_id']:03d} 生成成功")
                    else:
                        failed_count += 1
                        print(f"✗ [{success_count + failed_count:2d}/{TOTAL_SAMPLES}] "
                              f"示例 {sample_data['sample_id']:03d} 生成失败")
                    
            except Exception as e:
                failed_count += 1
                sample_id = futures[future]
                print(f"✗ 处理示例 {sample_id} 时出错: {e}")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # 生成汇总文件
    summary_file = os.path.join(output_dir, "summary.json")
    summary_data = {
        "total": TOTAL_SAMPLES,
        "success": success_count,
        "failed": failed_count,
        "success_rate": f"{success_count/TOTAL_SAMPLES*100:.1f}%",
        "elapsed_time_seconds": round(elapsed_time, 2),
        "average_time_seconds": round(elapsed_time/TOTAL_SAMPLES, 2),
        "generated_at": datetime.datetime.now().isoformat(),
        "config": {
            "max_workers": MAX_WORKERS,
            "agent_id": config['AGENT_ID'],
            "user_message": USER_MESSAGE
        }
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    
    # 打印统计
    print("\n" + "=" * 70)
    print("生成完成！")
    print("=" * 70)
    print(f"总数量: {TOTAL_SAMPLES}")
    print(f"成功数: {success_count} ✓")
    print(f"失败数: {failed_count} ✗")
    print(f"成功率: {success_count/TOTAL_SAMPLES*100:.1f}%")
    print(f"总耗时: {elapsed_time:.2f} 秒")
    print(f"平均值: {elapsed_time/TOTAL_SAMPLES:.2f} 秒/个")
    print(f"\n输出目录: {output_dir}/")
    print(f"汇总文件: {summary_file}")
    print("=" * 70)

