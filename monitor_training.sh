#!/bin/bash
# 训练监控脚本

PROJECT_DIR="/data_nvme/react_training"

echo "📊 ReAct 模型训练监控"
echo "================================"
echo ""

# 检查训练进程
if [ -f "${PROJECT_DIR}/train.pid" ]; then
    TRAIN_PID=$(cat ${PROJECT_DIR}/train.pid)
    if ps -p ${TRAIN_PID} > /dev/null 2>&1; then
        echo "✅ 训练进程运行中 (PID: ${TRAIN_PID})"
    else
        echo "⚠️  训练进程已停止 (PID: ${TRAIN_PID})"
    fi
else
    echo "❌ 未找到训练进程"
fi

echo ""
echo "================================"
echo ""

# 菜单选择
echo "请选择操作:"
echo "1) 查看实时日志"
echo "2) 查看 GPU 状态"
echo "3) 查看训练进度"
echo "4) 查看 TensorBoard"
echo "5) 显示训练统计"
echo "6) 停止训练"
echo "7) 清理检查点"
echo "0) 退出"
echo ""

read -p "输入选项 [0-7]: " choice

case $choice in
    1)
        echo ""
        echo "📄 实时日志 (Ctrl+C 退出)"
        echo "-------------------"
        tail -f ${PROJECT_DIR}/logs/train.log
        ;;
        
    2)
        echo ""
        echo "🖥️  GPU 监控 (Ctrl+C 退出)"
        echo "-------------------"
        watch -n 1 nvidia-smi
        ;;
        
    3)
        echo ""
        echo "📈 训练进度"
        echo "-------------------"
        
        # 检查输出目录
        OUTPUT_DIR="${PROJECT_DIR}/output"
        if [ -d "$OUTPUT_DIR" ]; then
            echo "检查点列表:"
            ls -lh ${OUTPUT_DIR}/*/checkpoint-* 2>/dev/null | tail -n 5 || echo "  未找到检查点"
            
            echo ""
            echo "最新日志:"
            tail -n 10 ${PROJECT_DIR}/logs/train.log | grep -E "(loss|epoch|step)" || echo "  未找到训练信息"
        else
            echo "  训练尚未开始或输出目录不存在"
        fi
        
        echo ""
        read -p "按 Enter 继续..."
        ;;
        
    4)
        echo ""
        echo "📊 启动 TensorBoard"
        echo "-------------------"
        echo "TensorBoard 将在后台启动..."
        
        nohup tensorboard --logdir=${PROJECT_DIR}/logs --host=0.0.0.0 --port=6006 > /dev/null 2>&1 &
        TB_PID=$!
        
        echo "✅ TensorBoard 已启动 (PID: ${TB_PID})"
        echo ""
        echo "访问方式:"
        echo "1. 如果在服务器上有桌面环境:"
        echo "   浏览器打开: http://localhost:6006"
        echo ""
        echo "2. 如果在远程连接:"
        echo "   在本地电脑运行 SSH 隧道:"
        echo "   ssh -L 6006:localhost:6006 username@server-ip"
        echo "   然后在本地浏览器打开: http://localhost:6006"
        echo ""
        read -p "按 Enter 继续..."
        ;;
        
    5)
        echo ""
        echo "📊 训练统计"
        echo "-------------------"
        
        if [ -f "${PROJECT_DIR}/train.pid" ]; then
            TRAIN_PID=$(cat ${PROJECT_DIR}/train.pid)
            
            # 训练时长
            if [ -f "${PROJECT_DIR}/logs/train_start.log" ]; then
                START_TIME=$(cat ${PROJECT_DIR}/logs/train_start.log | grep -oP '(?<=: ).*')
                echo "开始时间: ${START_TIME}"
                echo "当前时间: $(date)"
            fi
            
            echo ""
            
            # GPU 显存使用
            echo "GPU 显存:"
            nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
            
            echo ""
            
            # 日志文件大小
            if [ -f "${PROJECT_DIR}/logs/train.log" ]; then
                LOG_SIZE=$(du -h ${PROJECT_DIR}/logs/train.log | cut -f1)
                echo "日志大小: ${LOG_SIZE}"
            fi
            
            echo ""
            
            # 检查点数量和大小
            OUTPUT_DIR="${PROJECT_DIR}/output"
            if [ -d "$OUTPUT_DIR" ]; then
                CHECKPOINT_COUNT=$(find ${OUTPUT_DIR} -name "checkpoint-*" -type d 2>/dev/null | wc -l)
                CHECKPOINT_SIZE=$(du -sh ${OUTPUT_DIR} 2>/dev/null | cut -f1)
                echo "检查点数量: ${CHECKPOINT_COUNT}"
                echo "模型大小: ${CHECKPOINT_SIZE}"
            fi
            
            echo ""
            
            # 最新损失值
            echo "最新训练损失:"
            grep "{'loss':" ${PROJECT_DIR}/logs/train.log | tail -n 1 || echo "  未找到损失记录"
        else
            echo "  训练尚未开始"
        fi
        
        echo ""
        read -p "按 Enter 继续..."
        ;;
        
    6)
        echo ""
        echo "⚠️  停止训练"
        echo "-------------------"
        
        if [ -f "${PROJECT_DIR}/train.pid" ]; then
            TRAIN_PID=$(cat ${PROJECT_DIR}/train.pid)
            
            read -p "确认停止训练进程 ${TRAIN_PID}? (y/n): " confirm
            if [ "$confirm" = "y" ]; then
                kill ${TRAIN_PID}
                echo "✅ 训练进程已停止"
                
                # 停止 GPU 监控
                if [ -f "${PROJECT_DIR}/gpu_monitor.pid" ]; then
                    GPU_PID=$(cat ${PROJECT_DIR}/gpu_monitor.pid)
                    kill ${GPU_PID} 2>/dev/null
                    echo "✅ GPU 监控已停止"
                fi
            else
                echo "❌ 已取消"
            fi
        else
            echo "❌ 未找到训练进程"
        fi
        
        echo ""
        read -p "按 Enter 继续..."
        ;;
        
    7)
        echo ""
        echo "🧹 清理检查点"
        echo "-------------------"
        
        OUTPUT_DIR="${PROJECT_DIR}/output"
        if [ -d "$OUTPUT_DIR" ]; then
            echo "当前检查点:"
            ls -d ${OUTPUT_DIR}/*/checkpoint-* 2>/dev/null | tail -n 10
            
            echo ""
            echo "清理选项:"
            echo "1) 保留最新 3 个检查点"
            echo "2) 保留最新 1 个检查点"
            echo "3) 删除所有检查点（保留最终模型）"
            echo "0) 取消"
            
            read -p "选择: " clean_choice
            
            case $clean_choice in
                1)
                    find ${OUTPUT_DIR} -name "checkpoint-*" -type d | sort -V | head -n -3 | xargs rm -rf
                    echo "✅ 已清理，保留最新 3 个"
                    ;;
                2)
                    find ${OUTPUT_DIR} -name "checkpoint-*" -type d | sort -V | head -n -1 | xargs rm -rf
                    echo "✅ 已清理，保留最新 1 个"
                    ;;
                3)
                    rm -rf ${OUTPUT_DIR}/*/checkpoint-*
                    echo "✅ 所有检查点已删除"
                    ;;
                *)
                    echo "❌ 已取消"
                    ;;
            esac
        else
            echo "  未找到输出目录"
        fi
        
        echo ""
        read -p "按 Enter 继续..."
        ;;
        
    0)
        echo "👋 再见！"
        exit 0
        ;;
        
    *)
        echo "❌ 无效选项"
        ;;
esac

