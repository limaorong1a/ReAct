"""
测试 API 连接和智能体配置
在批量生成前使用此脚本验证配置
"""

import datetime
import urllib.parse
import base64
import hmac
import hashlib
import requests
import json
import time


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
                       base_url: str = "https://api-bj.clink.cn"):
    """创建会话"""
    api_path = "/agent/v1/create-conversation"
    url = f"{base_url}{api_path}"
    
    request_body = {
        "agent_id": agent_id,
        "user": user,
        "inputs": inputs or {},
        "created_at": int(datetime.datetime.now().timestamp() * 1000)
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
    response = requests.post(url=request_url, headers=headers, json=request_body)
    
    return response.json()


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
    
    response = requests.post(url=request_url, headers=headers, json=request_body)
    return response.json()


if __name__ == "__main__":
    print("="*70)
    print("API 连接测试")
    print("="*70)
    
    # 配置参数（与 generate_samples.py 保持一致）
    config = {
        'ACCESS_KEY_ID': "79c536f734569ae28ab2328b1f6abf9d",
        'ACCESS_KEY_SECRET': "0b4nUUD0nd4l4hr03p1s",
        'AGENT_ID': "1-d1d5bf70-cb5e-4701-94fa-bf274a7925e7",
        'USER': "yinshu_test"
    }
    
    # 测试消息
    test_message = "请生成一个新的 ReAct 任务示例"
    
    print(f"\n配置信息:")
    print(f"  Agent ID: {config['AGENT_ID']}")
    print(f"  User: {config['USER']}")
    print(f"  Test Message: \"{test_message}\"")
    print("-"*70)
    
    try:
        # 步骤1：创建会话
        print("\n[步骤 1/3] 创建会话...")
        start_time = time.time()
        
        conversation_result = create_conversation(
            access_key_id=config['ACCESS_KEY_ID'],
            access_key_secret=config['ACCESS_KEY_SECRET'],
            agent_id=config['AGENT_ID'],
            user=config['USER']
        )
        
        create_time = time.time() - start_time
        
        # 打印完整返回结果用于调试
        print(f"  返回结果: {json.dumps(conversation_result, ensure_ascii=False, indent=2)}")
        
        if 'error' in conversation_result:
            print(f"✗ 会话创建失败")
            print(f"  错误详情: {json.dumps(conversation_result['error'], ensure_ascii=False, indent=2)}")
            exit(1)
        
        conversation_id = conversation_result.get('conversation_id')
        if not conversation_id:
            print("✗ 未获取到 conversation_id")
            print(f"  请检查返回结果中的字段名称")
            exit(1)
        
        print(f"✓ 会话创建成功")
        print(f"  耗时: {create_time:.2f}秒")
        print(f"  Conversation ID: {conversation_id}")
        
        # 步骤2：发送测试消息
        print("\n[步骤 2/3] 发送测试消息...")
        start_time = time.time()
        
        chat_result = send_chat_message(
            access_key_id=config['ACCESS_KEY_ID'],
            access_key_secret=config['ACCESS_KEY_SECRET'],
            agent_id=config['AGENT_ID'],
            conversation_id=conversation_id,
            message=test_message,
            user=config['USER']
        )
        
        chat_time = time.time() - start_time
        
        if 'error' in chat_result:
            print(f"✗ 消息发送失败")
            print(f"  错误: {json.dumps(chat_result['error'], ensure_ascii=False, indent=2)}")
            exit(1)
        
        print(f"✓ 消息发送成功")
        print(f"  耗时: {chat_time:.2f}秒")
        
        # 步骤3：检查回复内容
        print("\n[步骤 3/3] 检查回复内容...")
        
        if 'answer' not in chat_result:
            print(f"✗ 返回格式异常")
            print(f"  完整返回: {json.dumps(chat_result, ensure_ascii=False, indent=2)}")
            exit(1)
        
        answer_content = chat_result['answer'][0].get('content', '')
        
        if not answer_content:
            print(f"✗ 回复内容为空")
            exit(1)
        
        print(f"✓ 成功获取回复")
        print(f"  回复长度: {len(answer_content)} 字符")
        
        # 检查格式
        if answer_content.startswith("You are in the middle of a room"):
            print(f"✓ 格式检查通过（以正确开头）")
        else:
            print(f"⚠ 格式可能有问题（未以标准开头）")
        
        # 显示回复内容
        print("\n" + "-"*70)
        print("回复内容预览（前600字符）:")
        print("-"*70)
        print(answer_content[:600])
        if len(answer_content) > 600:
            print("...")
        print("-"*70)
        
        # 测试总结
        print("\n" + "="*70)
        print("测试结果")
        print("="*70)
        print(f"✓ API 连接正常")
        print(f"✓ 智能体响应正常")
        print(f"✓ 总耗时: {create_time + chat_time:.2f}秒")
        
        if answer_content.startswith("You are in the middle of a room"):
            print(f"✓ 输出格式正确")
            print(f"\n🎉 测试通过！可以开始批量生成了：")
            print(f"   python generate_samples.py")
        else:
            print(f"⚠ 输出格式需要检查")
            print(f"\n建议：")
            print(f"  1. 检查 Dify 智能体的系统提示词是否正确配置")
            print(f"  2. 确保使用了 dify_system_prompt.txt 中的内容")
            print(f"  3. 检查智能体的温度参数（建议 0.7-0.9）")
        
        print("="*70)
        
    except requests.exceptions.ConnectionError as e:
        print(f"\n✗ 网络连接错误: {e}")
        print("  请检查网络连接和 API 地址")
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        print("\n请检查：")
        print("  1. ACCESS_KEY_ID 和 ACCESS_KEY_SECRET 是否正确")
        print("  2. AGENT_ID 是否正确")
        print("  3. Dify 智能体是否配置了系统提示词")
        print("  4. 网络连接是否正常")

