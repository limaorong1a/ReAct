# ReAct 示例批量生成工具

通过 Dify 智能体 API 自动生成 ReAct (Reasoning + Acting) 任务示例。

---

## 🚀 快速开始

### 1. 配置 Dify 智能体

1. 登录 Dify 平台，创建"聊天助手"应用
2. 将 `dify_system_prompt.txt` 中的**完整内容**复制到**系统提示词**
3. 设置模型参数：
   - 模型：GPT-4 或 Claude（推荐）
   - 温度：0.8
   - 最大长度：3000 tokens
4. 保存并发布智能体
5. 获取 Agent ID 和 API 密钥

### 2. 修改配置

在 `generate_samples.py` 中修改配置：

```python
config = {
    'ACCESS_KEY_ID': "你的KEY",
    'ACCESS_KEY_SECRET': "你的SECRET",
    'AGENT_ID': "你的AGENT_ID",
    'USER': "用户标识"
}

TOTAL_SAMPLES = 30  # 生成数量
MAX_WORKERS = 3     # 并发线程数
```

### 3. 测试连接

```bash
python test_connection.py
```

**预期输出**：
```
✓ API 连接正常
✓ 智能体响应正常
✓ 输出格式正确
🎉 测试通过！可以开始批量生成了
```

### 4. 批量生成

```bash
python generate_samples.py
```

生成过程会实时显示进度：
```
✓ [1/30] 示例 001 生成成功
✓ [2/30] 示例 002 生成成功
...
```

### 5. 合并文件（可选）

```bash
python merge_generated_samples.py
```

---

## 📁 文件说明

| 文件 | 说明 |
|------|------|
| `dify_system_prompt.txt` | **系统提示词**（复制到 Dify 中） |
| `generate_samples.py` | 主程序：批量生成示例 |
| `test_connection.py` | 测试工具：验证配置 |
| `merge_generated_samples.py` | 合并工具：整合文件 |
| `generated_samples/` | 输出目录（自动创建） |

---

## ⚙️ 工作原理

### 系统提示词（配置在 Dify 中）
- 包含完整的任务说明
- 包含3个参考示例
- 包含格式要求和约束

### 用户消息（Python 发送）
- 简单的触发指令："请生成一个新的 ReAct 任务示例"
- 智能体根据系统提示词生成新示例

### 流程图
```
Python 脚本
  ↓ 发送: "请生成一个新的 ReAct 任务示例"
Dify 智能体（已配置系统提示词）
  ↓ 返回: 完整的 ReAct 示例
Python 脚本
  ↓ 保存到文件
generated_samples/sample_001.json
```

---

## 📊 输出结果

### 单个示例文件
```json
{
  "generated_react_put_1": "You are in the middle of a room...",
  "metadata": {
    "conversation_id": "xxx",
    "generated_at": "2025-10-22T12:00:00",
    "success": true
  }
}
```

### 汇总文件
```json
{
  "total": 30,
  "success": 28,
  "failed": 2,
  "success_rate": "93.3%",
  "elapsed_time_seconds": 65.23,
  "average_time_seconds": 2.17
}
```

---

## 🎯 配置参数

### 生成参数
```python
TOTAL_SAMPLES = 30   # 生成数量（建议先测试5-10个）
MAX_WORKERS = 3      # 并发线程数（2-5，避免API限流）
USER_MESSAGE = "请生成一个新的 ReAct 任务示例"  # 触发消息
```

### Dify 智能体参数
- **温度**：0.7-0.9（保证多样性）
- **最大长度**：2000-3000 tokens
- **模型**：GPT-4 或 Claude

---

## 💡 使用建议

### 1. 分批生成
```bash
# 第一批：测试5个
TOTAL_SAMPLES = 5
python generate_samples.py

# 检查质量后，生成30个
TOTAL_SAMPLES = 30
python generate_samples.py
```

### 2. 控制并发
- 如果频繁出现 "TooManyRequests"，减少 `MAX_WORKERS`
- 建议设置为 2-3

### 3. 质量检查
- 查看 `generated_samples/summary.json` 的成功率
- 随机抽查 5-10 个示例文件
- 确认格式是否正确

---

## 🐛 故障排查

### 问题1：测试失败
**检查清单**：
- [ ] API 配置是否正确
- [ ] Dify 智能体是否正常运行
- [ ] 系统提示词是否完整复制
- [ ] 网络连接是否正常

### 问题2：格式不对
**解决方案**：
- 确保使用 `dify_system_prompt.txt` 的完整内容
- 检查 Dify 中的系统提示词是否被截断
- 调整温度参数（过高会导致格式混乱）

### 问题3：内容重复
**解决方案**：
- 提高温度参数（0.8-0.9）
- 确保系统提示词中强调了"多样性"

### 问题4：API 超限
**解决方案**：
- 减少 `MAX_WORKERS` 到 1-2
- 在代码中添加 `time.sleep(1)` 延迟

---

## 📈 性能参考

| 指标 | 数值 |
|------|------|
| 生成速度 | 1-3 秒/个 |
| 并发推荐 | 2-3 线程 |
| 成功率 | 90-95% |
| 总耗时 | 1-2 分钟（30个） |

---

## 📞 技术支持

遇到问题时：
1. 运行 `python test_connection.py` 诊断
2. 查看终端输出的错误信息
3. 检查 `generated_samples/summary.json`
4. 确认 Dify 智能体配置

---

## 📝 更新日志

**v1.0** (2025-10-22)
- ✨ 初始版本
- ✅ 支持批量生成
- ✅ 多线程并发
- ✅ 自动重试机制
- ✅ 简化的用户接口

---

**最后更新**: 2025-10-22
