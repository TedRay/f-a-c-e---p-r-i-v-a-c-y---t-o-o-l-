#!/bin/bash
# 幸福的照片隐私打码工具 - 启动脚本

cd "$(dirname "$0")"

echo "========================================"
echo "  幸福的照片隐私打码工具"
echo "========================================"
echo ""

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 Python3"
    exit 1
fi

# 创建必要目录
mkdir -p static/uploads static/results known_faces models

# 安装依赖
echo "检查依赖..."
pip3 install -q --break-system-packages -r requirements.txt 2>/dev/null || \
pip3 install -q -r requirements.txt 2>/dev/null

# 启动服务
echo "启动 Web 服务..."
echo "访问地址: http://localhost:5000"
echo ""

python3 app.py
