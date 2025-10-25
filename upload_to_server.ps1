# Windows PowerShell 上传脚本
# 使用方式: 在 PowerShell 中运行: .\upload_to_server.ps1

# ====================================
# 配置区域 - 请修改为你的服务器信息
# ====================================

$SERVER_USER = "your_username"        # 替换为你的用户名
$SERVER_IP = "your_server_ip"         # 替换为服务器 IP
$SERVER_PATH = "/data_nvme/react_training"  # 服务器目标路径
$SSH_KEY = ""                         # 可选：SSH 密钥路径

# ====================================

Write-Host "🚀 ReAct 数据上传到服务器" -ForegroundColor Green
Write-Host "================================" -ForegroundColor Green
Write-Host ""

# 检查是否已准备数据
if (-not (Test-Path "react_train_alpaca.json")) {
    Write-Host "⚠️  未找到训练数据，正在生成..." -ForegroundColor Yellow
    python prepare_training_data.py
    Write-Host "✅ 数据生成完成" -ForegroundColor Green
}

# 检查 SSH 工具
$scp_available = Get-Command scp -ErrorAction SilentlyContinue
if (-not $scp_available) {
    Write-Host "❌ 未找到 SCP 工具" -ForegroundColor Red
    Write-Host ""
    Write-Host "请安装以下工具之一：" -ForegroundColor Yellow
    Write-Host "1. OpenSSH Client (Windows 10/11 自带)" -ForegroundColor Yellow
    Write-Host "   设置 -> 应用 -> 可选功能 -> 添加功能 -> OpenSSH 客户端" -ForegroundColor Yellow
    Write-Host "2. Git for Windows (包含 SSH/SCP)" -ForegroundColor Yellow
    Write-Host "3. WinSCP (图形界面)" -ForegroundColor Yellow
    exit 1
}

# 构建 SSH/SCP 命令
$ssh_cmd = "ssh"
$scp_cmd = "scp"
if ($SSH_KEY) {
    $ssh_cmd = "ssh -i `"$SSH_KEY`""
    $scp_cmd = "scp -i `"$SSH_KEY`""
}

# 测试连接
Write-Host "🔗 测试服务器连接..." -ForegroundColor Cyan
$test_result = & ssh $SERVER_USER@$SERVER_IP "echo 'connected'" 2>&1
if ($test_result -like "*connected*") {
    Write-Host "✅ 服务器连接正常" -ForegroundColor Green
} else {
    Write-Host "❌ 无法连接到服务器，请检查配置" -ForegroundColor Red
    Write-Host $test_result
    exit 1
}

# 创建服务器目录
Write-Host ""
Write-Host "📁 创建服务器目录..." -ForegroundColor Cyan
& ssh $SERVER_USER@$SERVER_IP "mkdir -p $SERVER_PATH/{data,models,output,logs,scripts}"
Write-Host "✅ 目录创建完成" -ForegroundColor Green

# 上传数据文件
Write-Host ""
Write-Host "📤 上传训练数据..." -ForegroundColor Cyan
scp react_*.json ${SERVER_USER}@${SERVER_IP}:${SERVER_PATH}/data/
if ($LASTEXITCODE -eq 0) {
    Write-Host "✅ 数据文件上传完成" -ForegroundColor Green
} else {
    Write-Host "❌ 数据上传失败" -ForegroundColor Red
    exit 1
}

# 上传脚本文件
Write-Host ""
Write-Host "📤 上传脚本文件..." -ForegroundColor Cyan
scp prepare_training_data.py ${SERVER_USER}@${SERVER_IP}:${SERVER_PATH}/
scp llama_factory_train_config.yaml ${SERVER_USER}@${SERVER_IP}:${SERVER_PATH}/
scp server_auto_train.sh ${SERVER_USER}@${SERVER_IP}:${SERVER_PATH}/
scp monitor_training.sh ${SERVER_USER}@${SERVER_IP}:${SERVER_PATH}/
Write-Host "✅ 脚本文件上传完成" -ForegroundColor Green

# 上传文档
Write-Host ""
Write-Host "📤 上传文档..." -ForegroundColor Cyan
scp SERVER_TRAINING_GUIDE.md ${SERVER_USER}@${SERVER_IP}:${SERVER_PATH}/
scp RTX5090_QUICKSTART.md ${SERVER_USER}@${SERVER_IP}:${SERVER_PATH}/
Write-Host "✅ 文档上传完成" -ForegroundColor Green

# 设置脚本权限
Write-Host ""
Write-Host "🔧 设置脚本权限..." -ForegroundColor Cyan
& ssh $SERVER_USER@$SERVER_IP "chmod +x $SERVER_PATH/*.sh"
Write-Host "✅ 权限设置完成" -ForegroundColor Green

# 上传原始数据（可选）
Write-Host ""
$upload_raw = Read-Host "是否上传原始数据文件夹？(y/n)"
if ($upload_raw -eq "y") {
    Write-Host ""
    Write-Host "📤 上传原始数据（这可能需要一些时间）..." -ForegroundColor Cyan
    
    $dirs = Get-ChildItem -Directory -Filter "generated_samples_*"
    foreach ($dir in $dirs) {
        Write-Host "上传 $($dir.Name)..." -ForegroundColor Yellow
        scp -r $dir.FullName ${SERVER_USER}@${SERVER_IP}:${SERVER_PATH}/
    }
    Write-Host "✅ 原始数据上传完成" -ForegroundColor Green
}

# 显示摘要
Write-Host ""
Write-Host "================================" -ForegroundColor Green
Write-Host "✨ 上传完成！" -ForegroundColor Green
Write-Host ""
Write-Host "已上传到服务器：" -ForegroundColor Cyan
Write-Host "  路径: $SERVER_PATH" -ForegroundColor White
Write-Host "  数据: react_*.json" -ForegroundColor White
Write-Host "  配置: llama_factory_train_config.yaml" -ForegroundColor White
Write-Host "  脚本: server_auto_train.sh, monitor_training.sh" -ForegroundColor White
Write-Host ""
Write-Host "下一步：" -ForegroundColor Yellow
Write-Host "1. SSH 连接到服务器:" -ForegroundColor White
Write-Host "   ssh $SERVER_USER@$SERVER_IP" -ForegroundColor Gray
Write-Host ""
Write-Host "2. 进入项目目录:" -ForegroundColor White
Write-Host "   cd $SERVER_PATH" -ForegroundColor Gray
Write-Host ""
Write-Host "3. 运行自动训练脚本:" -ForegroundColor White
Write-Host "   bash server_auto_train.sh" -ForegroundColor Gray
Write-Host ""
Write-Host "4. 或查看快速开始指南:" -ForegroundColor White
Write-Host "   cat RTX5090_QUICKSTART.md" -ForegroundColor Gray
Write-Host ""

