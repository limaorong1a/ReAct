# RTX 5090 服务器训练快速开始

## 🎯 5 分钟快速开始

### 配置信息
- **GPU**: NVIDIA GeForce RTX 5090 D × 1
- **显存**: 24GB
- **CPU**: 16 vCPU
- **内存**: 47GB
- **存储**: /data_nvme (NVMe SSD)

---

## 📝 第一步：在本地准备数据（5 分钟）

```bash
# 1. 生成训练数据
python prepare_training_data.py

# 2. 修改上传脚本配置
# 编辑 upload_to_server.sh，修改：
#   SERVER_USER="your_username"
#   SERVER_IP="your_server_ip"

# 3. 上传到服务器
chmod +x upload_to_server.sh
./upload_to_server.sh
```

---

## 🚀 第二步：在服务器上训练（一键完成）

### 方法 A：自动化脚本（推荐）

```bash
# 1. SSH 连接服务器
ssh username@server-ip

# 2. 进入项目目录
cd /data_nvme/react_training

# 3. 运行自动训练脚本
chmod +x server_auto_train.sh
bash server_auto_train.sh

# 脚本会自动完成：
# ✅ 环境检查
# ✅ 安装依赖
# ✅ 配置数据集
# ✅ 启动训练
```

### 方法 B：使用 tmux（推荐，可断开连接）

```bash
# 1. 创建 tmux 会话
tmux new -s react_training

# 2. 运行训练脚本
cd /data_nvme/react_training
bash server_auto_train.sh

# 3. 分离会话（训练继续运行）
# 按 Ctrl+B，然后按 D

# 4. 重新连接（随时可以）
tmux attach -t react_training
```

---

## 📊 第三步：监控训练（实时查看）

### 使用监控脚本

```bash
cd /data_nvme/react_training
chmod +x monitor_training.sh
bash monitor_training.sh

# 然后选择操作：
# 1 - 查看实时日志
# 2 - 监控 GPU
# 3 - 查看进度
# 4 - 启动 TensorBoard
# 5 - 查看统计
```

### 常用命令

```bash
# 查看实时日志
tail -f /data_nvme/react_training/logs/train.log

# 监控 GPU
watch -n 1 nvidia-smi

# 查看进度
ls -lh /data_nvme/react_training/output/qwen2.5-7b-react/checkpoint-*

# 启动 TensorBoard
tensorboard --logdir=/data_nvme/react_training/logs --host=0.0.0.0 --port=6006

# 在本地电脑建立隧道访问
ssh -L 6006:localhost:6006 username@server-ip
# 然后打开: http://localhost:6006
```

---

## ⏱️ 训练时间预估

基于 RTX 5090 和 600 个样本：

| 模型 | 预计时间 | 显存占用 | 推荐度 |
|------|---------|---------|--------|
| Qwen2.5-3B | 1.5-2 小时 | ~12GB | ⭐⭐⭐ |
| **Qwen2.5-7B** | **3-4 小时** | **~18GB** | **⭐⭐⭐⭐⭐** |
| Qwen2.5-14B | 6-8 小时 | ~22GB | ⭐⭐⭐⭐ |

**推荐**: Qwen2.5-7B（性能与速度最平衡）

---

## 💾 第四步：下载模型到本地

### 训练完成后

```bash
# 在服务器上合并模型（如果需要）
cd /data_nvme/react_training/LLaMA-Factory

llamafactory-cli export \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --adapter_name_or_path /data_nvme/react_training/output/qwen2.5-7b-react \
    --template qwen \
    --finetuning_type lora \
    --export_dir /data_nvme/react_training/merged_model \
    --export_size 2 \
    --export_device cpu

# 在本地电脑下载（使用 rsync，支持断点续传）
rsync -avz --progress username@server-ip:/data_nvme/react_training/merged_model ./

# 或只下载 LoRA 适配器（更小，约 100MB）
rsync -avz --progress username@server-ip:/data_nvme/react_training/output/qwen2.5-7b-react ./
```

---

## 🔧 常见问题

### Q1: 如何断开连接后继续训练？

**A**: 使用 tmux 或 nohup（自动脚本已使用）

```bash
# tmux 方式
tmux new -s training
# 运行训练...
# 按 Ctrl+B, D 分离

# 重新连接
tmux attach -t training
```

### Q2: 如何修改训练参数？

**A**: 编辑 `server_auto_train.sh` 顶部的配置区域：

```bash
# 训练参数
BATCH_SIZE=4        # 批次大小（可改为 6 或 8）
GRAD_ACCUM=4        # 梯度累积
LORA_RANK=32        # LoRA 秩
EPOCHS=3            # 训练轮数
LEARNING_RATE=5e-5  # 学习率
```

### Q3: 训练失败怎么办？

**A**: 检查日志文件

```bash
# 查看错误信息
tail -n 50 /data_nvme/react_training/logs/train.log

# 常见问题：
# - 显存不足: 减小 BATCH_SIZE
# - 数据路径错误: 检查 dataset_info.json
# - 网络问题: 预先下载模型到本地
```

### Q4: 如何停止训练？

**A**: 使用监控脚本或直接 kill

```bash
# 方法 1: 使用监控脚本
bash monitor_training.sh
# 选择 6 - 停止训练

# 方法 2: 手动停止
cat /data_nvme/react_training/train.pid
kill <PID>
```

### Q5: 如何从检查点继续训练？

**A**: 添加 `--resume_from_checkpoint` 参数

```bash
--resume_from_checkpoint /data_nvme/react_training/output/qwen2.5-7b-react/checkpoint-500
```

---

## 📈 预期效果

基于 600 个高质量样本训练：

| 指标 | Qwen2.5-7B | Qwen2.5-14B |
|------|-----------|------------|
| **训练损失** | ~0.3 | ~0.25 |
| **验证损失** | ~0.4 | ~0.35 |
| **简单任务准确率** | 85%+ | 90%+ |
| **复杂任务准确率** | 55-65% | 65-75% |
| **平均准确率** | 70-75% | 75-80% |

---

## 🎓 完整文档

- **详细指南**: `SERVER_TRAINING_GUIDE.md`
- **训练对比**: `TRAINING_COMPARISON.md`
- **通用指南**: `TRAINING_GUIDE.md`

---

## 📞 脚本文件

| 文件 | 用途 |
|------|------|
| `upload_to_server.sh` | 上传数据到服务器 |
| `server_auto_train.sh` | 自动化训练脚本 |
| `monitor_training.sh` | 训练监控脚本 |
| `prepare_training_data.py` | 数据准备脚本 |

---

## ✅ 检查清单

在开始训练前，确保：

- [ ] 已在本地运行 `prepare_training_data.py`
- [ ] 已上传数据到服务器 `/data_nvme/react_training/data/`
- [ ] 已修改 `upload_to_server.sh` 中的服务器信息
- [ ] 可以 SSH 连接到服务器
- [ ] 服务器上 `nvidia-smi` 显示 RTX 5090
- [ ] 服务器上有足够的存储空间（至少 50GB）

---

## 🚀 一键命令总结

```bash
# === 本地操作 ===
# 1. 准备数据
python prepare_training_data.py

# 2. 上传到服务器（修改配置后）
chmod +x upload_to_server.sh
./upload_to_server.sh

# === 服务器操作 ===
# 3. SSH 连接
ssh username@server-ip

# 4. 开始训练（tmux 方式）
cd /data_nvme/react_training
tmux new -s react_training
bash server_auto_train.sh
# Ctrl+B, D 分离

# 5. 监控训练
bash monitor_training.sh

# 6. 下载模型（本地）
rsync -avz --progress username@server-ip:/data_nvme/react_training/merged_model ./
```

---

**祝训练顺利！** 🎉

有问题请查看详细文档或提 Issue。

