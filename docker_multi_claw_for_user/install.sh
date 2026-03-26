#!/bin/bash
# deploy.sh - 一键部署脚本

set -e

echo "🚀 开始部署CoPaw多用户系统..."

# 检查Docker和Docker Compose
if ! command -v docker &> /dev/null; then
    echo "❌ Docker未安装，请先安装Docker"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose未安装，请先安装Docker Compose"
    exit 1
fi

# 创建必要的目录
mkdir -p nginx/conf.d
mkdir -p backups
mkdir -p logs
mkdir -p ssl

# 复制配置文件
echo "📝 配置Nginx..."
cp nginx/nginx.conf nginx/
cp nginx/conf.d/copaw.conf nginx/conf.d/

# 构建并启动服务
echo "🔨 构建Docker镜像..."
docker-compose build

echo "🚀 启动服务..."
docker-compose up -d

# 等待服务启动
echo "⏳ 等待服务启动..."
sleep 30

# 健康检查
echo "🔍 检查服务状态..."
if curl -f http://localhost/health > /dev/null 2>&1; then
    echo "✅ 部署成功！"
    echo "🌐 管理界面: http://localhost/admin/"
    echo "📊 API文档: http://localhost/admin/docs"
else
    echo "❌ 部署失败，请检查日志"
    docker-compose logs
    exit 1
fi

echo "🎉 CoPaw多用户系统部署完成！"