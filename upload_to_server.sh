#!/bin/bash
# 数据上传脚本 - 从本地上传到服务器

set -e

# ====================================
# 配置区域 - 请修改为你的服务器信息
# ====================================

SERVER_USER="your_username"        # 替换为你的用户名
SERVER_IP="your_server_ip"         # 替换为服务器 IP
SERVER_PATH="/data_nvme/react_training"  # 服务器目标路径
SSH_KEY=""                         # 可选：SSH 密钥路径

# ====================================

echo "🚀 ReAct 数据上传到服务器"
echo "================================"
echo ""

# 检查是否已准备数据
if [ ! -f "react_train_alpaca.json" ]; then
    echo "⚠️  未找到训练数据，正在生成..."
    python prepare_training_data.py
    echo "✅ 数据生成完成"
fi

# 构建 SSH 命令
if [ -n "$SSH_KEY" ]; then
    SSH_CMD="ssh -i $SSH_KEY"
    SCP_CMD="scp -i $SSH_KEY"
    RSYNC_CMD="rsync -e \"ssh -i $SSH_KEY\""
else
    SSH_CMD="ssh"
    SCP_CMD="scp"
    RSYNC_CMD="rsync"
fi

# 测试连接
echo "🔗 测试服务器连接..."
if $SSH_CMD ${SERVER_USER}@${SERVER_IP} "echo '连接成功'"; then
    echo "✅ 服务器连接正常"
else
    echo "❌ 无法连接到服务器，请检查配置"
    exit 1
fi

# 创建服务器目录
echo ""
echo "📁 创建服务器目录..."
$SSH_CMD ${SERVER_USER}@${SERVER_IP} "mkdir -p ${SERVER_PATH}/{data,models,output,logs,scripts}"

# 上传数据文件
echo ""
echo "📤 上传训练数据..."
$SCP_CMD -r react_*.json ${SERVER_USER}@${SERVER_IP}:${SERVER_PATH}/data/
echo "✅ 数据文件上传完成"

# 上传脚本文件
echo ""
echo "📤 上传脚本文件..."
$SCP_CMD prepare_training_data.py ${SERVER_USER}@${SERVER_IP}:${SERVER_PATH}/
$SCP_CMD llama_factory_train_config.yaml ${SERVER_USER}@${SERVER_IP}:${SERVER_PATH}/
echo "✅ 脚本文件上传完成"

# 上传文档
echo ""
echo "📤 上传文档..."
$SCP_CMD SERVER_TRAINING_GUIDE.md ${SERVER_USER}@${SERVER_IP}:${SERVER_PATH}/
$SCP_CMD TRAINING_GUIDE.md ${SERVER_USER}@${SERVER_IP}:${SERVER_PATH}/
echo "✅ 文档上传完成"

# 上传原始数据（可选）
read -p "是否上传原始数据文件夹？(y/n): " upload_raw
if [ "$upload_raw" = "y" ]; then
    echo ""
    echo "📤 上传原始数据（这可能需要一些时间）..."
    for dir in generated_samples_*; do
        if [ -d "$dir" ]; then
            echo "上传 $dir..."
            $SCP_CMD -r "$dir" ${SERVER_USER}@${SERVER_IP}:${SERVER_PATH}/
        fi
    done
    echo "✅ 原始数据上传完成"
fi

# 显示上传摘要
echo ""
echo "================================"
echo "✨ 上传完成！"
echo ""
echo "已上传到服务器："
echo "  路径: ${SERVER_PATH}"
echo "  数据: react_*.json"
echo "  配置: llama_factory_train_config.yaml"
echo ""
echo "下一步："
echo "1. SSH 连接到服务器:"
echo "   ssh ${SERVER_USER}@${SERVER_IP}"
echo ""
echo "2. 运行环境配置脚本:"
echo "   cd ${SERVER_PATH}"
echo "   bash quick_start.sh"
echo ""
echo "3. 或手动配置 LLaMA-Factory"
echo ""

