# CoPaw 多用户实例管理系统 API 接口文档

> 文档生成时间：2026-03-30  
> 项目名称：CoPaw Multi-User Manager  
> 版本：1.0.0  
> 基础路径：`http://localhost:8080`

---

## 目录

1. [概述](#概述)
2. [系统架构](#系统架构)
3. [接口分类详解](#接口分类详解)
   - [用户管理接口](#1-用户管理接口)
   - [实例管理接口](#2-实例管理接口)
   - [备份管理接口](#3-备份管理接口)
   - [资源监控接口](#4-资源监控接口)
   - [系统接口](#5-系统接口)
4. [数据模型](#数据模型)
5. [错误处理](#错误处理)
6. [后端代码索引](#后端代码索引)

---

## 概述

CoPaw 多用户实例管理系统是一个基于 FastAPI 构建的后端服务，用于为每个用户创建和管理独立的 CoPaw（智能体框架）容器实例。

### 技术栈
- **后端框架**：FastAPI + Uvicorn
- **容器技术**：Docker
- **数据库**：PostgreSQL
- **缓存**：Redis（可选）

### 核心功能
- 用户注册/登录/登出/注销
- 为每个用户创建独立的 Docker 容器实例
- 实例生命周期管理（启动、停止、重建）
- 用户数据备份与恢复
- 实例资源监控（CPU、内存、磁盘）

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                     FastAPI Application                       │
│                        (main.py:8080)                         │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ InstanceManager │  │LifecycleManager │  │BackupManager │ │
│  │   (容器管理)     │  │  (生命周期管理)   │  │  (备份管理)  │ │
│  └────────┬────────┘  └────────┬────────┘  └──────┬───────┘ │
│           │                    │                   │         │
│           ▼                    ▼                   ▼         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ResourceMonitor  │  │ PostgresClient  │  │ Docker API   │ │
│  │  (资源监控)      │  │   (数据持久化)   │  │  (容器操作)  │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### 端口分配
- **管理服务**：8080
- **用户实例**：9001-9999（动态分配）

### Docker 容器规范
- **镜像**：`osclaw:v010`
- **命名规则**：`copaw-user-{user_id}`
- **数据卷**：`copaw-data-{user_id}`、`copaw-secrets-{user_id}`
- **资源限制**：内存 1GB，CPU 50%

---

## 接口分类详解

---

### 1. 用户管理接口

**路径前缀**：`/admin/users`  
**后端文件**：`main.py`  

#### 1.1 用户注册

```
POST /admin/users/{user_id}/register?user_name={user_name}
```

**代码位置**：`main.py:118`

**功能描述**：为新用户创建独立的 CoPaw 容器实例，用于用户首次使用系统。

**路径参数**：
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| user_id | string | 是 | 用户唯一标识符 |

**查询参数**：
| 参数 | 类型 | 必填 | 说明 |
|----------|--------|------|------|
| user_name | string | 是 | 用户显示名称 |

**请求示例**：
```bash
curl -X POST "http://localhost:8080/admin/users/user001/register?user_name=张三"
```

**成功响应**：
```json
{
  "success": true,
  "data": {
    "user_id": "user001",
    "user_name": "张三",
    "container_id": "abc123...",
    "container_name": "copaw-user-user001",
    "port": 9001,
    "url": "http://localhost:9001",
    "status": "running",
    "created_at": "2026-03-30T10:00:00"
  }
}
```

**执行流程**：
1. 保存用户信息到 PostgreSQL 数据库
2. 分配可用端口（9001-9999）
3. 创建 Docker 数据卷（data + secrets）
4. 启动 CoPaw 容器实例
5. 等待容器初始化完成
6. 保存实例信息到数据库

**应用场景**：用户首次注册使用系统

---

#### 1.2 用户登录

```
POST /admin/users/{user_id}/login
```

**代码位置**：`main.py:128`

**功能描述**：启动用户对应的 CoPaw 容器实例，返回访问地址。若容器不存在或启动失败，会自动重建。

**路径参数**：
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| user_id | string | 是 | 用户唯一标识符 |

**请求示例**：
```bash
curl -X POST "http://localhost:8080/admin/users/user001/login"
```

**成功响应**：
```json
{
  "success": true,
  "url": "http://localhost:9001"
}
```

**错误响应**：
```json
{
  "detail": "Instance not available"
}
```

**执行流程**：
1. 从数据库获取用户实例信息
2. 检查容器运行状态
3. 若容器未运行，尝试启动
4. 若启动失败，删除并重建容器
5. 更新用户活跃时间
6. 返回容器访问地址

**应用场景**：用户登录系统获取实例访问地址

---

#### 1.3 用户登出

```
POST /admin/users/{user_id}/logout
```

**代码位置**：`main.py:140`

**功能描述**：停止用户对应的 CoPaw 容器实例，释放资源。

**路径参数**：
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| user_id | string | 是 | 用户唯一标识符 |

**请求示例**：
```bash
curl -X POST "http://localhost:8080/admin/users/user001/logout"
```

**成功响应**：
```json
{
  "success": true
}
```

**执行流程**：
1. 获取用户容器信息
2. 发送停止命令
3. 更新实例状态为 stopped
4. 记录登出日志

**应用场景**：用户主动退出系统

---

#### 1.4 删除用户

```
DELETE /admin/users/{user_id}
```

**代码位置**：`main.py:150`

**功能描述**：完全删除用户及其所有相关资源（容器、数据卷、数据库记录）。

**路径参数**：
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| user_id | string | 是 | 用户唯一标识符 |

**请求示例**：
```bash
curl -X DELETE "http://localhost:8080/admin/users/user001"
```

**成功响应**：
```json
{
  "success": true
}
```

**执行流程**：
1. 停止并删除 Docker 容器
2. 删除数据卷（data + secrets）
3. 删除数据库中的用户记录
4. 删除实例记录
5. 记录删除日志

**应用场景**：用户注销账号或管理员清理用户

---

### 2. 实例管理接口

**路径前缀**：`/admin/instances`  
**后端文件**：`main.py`

#### 2.1 列出所有实例

```
GET /admin/instances
```

**代码位置**：`main.py:160`

**功能描述**：获取系统中所有 CoPaw 实例的列表信息。

**请求示例**：
```bash
curl "http://localhost:8080/admin/instances"
```

**成功响应**：
```json
{
  "success": true,
  "data": [
    {
      "id": "inst_001",
      "user_id": "user001",
      "user_name": "张三",
      "container_id": "abc123...",
      "container_name": "copaw-user-user001",
      "port": 9001,
      "url": "http://localhost:9001",
      "status": "running",
      "created_at": "2026-03-30T10:00:00",
      "last_active": "2026-03-30T15:30:00"
    },
    {
      "id": "inst_002",
      "user_id": "user002",
      "user_name": "李四",
      "container_name": "copaw-user-user002",
      "port": 9002,
      "url": "http://localhost:9002",
      "status": "stopped",
      "created_at": "2026-03-29T09:00:00"
    }
  ]
}
```

**应用场景**：运维人员查看所有用户实例状态

---

### 3. 备份管理接口

**路径前缀**：`/admin/backups`  
**后端文件**：`main.py`、`data_backup_recover.py`

#### 3.1 列出用户备份

```
GET /admin/backups/{user_id}
```

**代码位置**：`main.py:170`

**功能描述**：获取指定用户的所有备份记录列表。

**路径参数**：
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| user_id | string | 是 | 用户唯一标识符 |

**请求示例**：
```bash
curl "http://localhost:8080/admin/backups/user001"
```

**成功响应**：
```json
{
  "success": true,
  "data": [
    {
      "id": "backup_001",
      "user_id": "user001",
      "backup_path": "/backup/user001_backup_20260330_100000.tar.gz",
      "secrets_path": "/backup/secrets_user001_backup_20260330_100000.tar.gz",
      "backup_size": 15728640,
      "created_at": "2026-03-30T10:00:00"
    },
    {
      "id": "backup_002",
      "user_id": "user001",
      "backup_path": "/backup/user001_backup_20260329_180000.tar.gz",
      "backup_size": 14680064,
      "created_at": "2026-03-29T18:00:00"
    }
  ]
}
```

**应用场景**：查看用户历史备份记录

---

#### 3.2 创建用户备份

```
POST /admin/backups/{user_id}/create
```

**代码位置**：`main.py:180`

**功能描述**：为指定用户创建数据备份，包括工作数据和敏感数据。

**路径参数**：
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| user_id | string | 是 | 用户唯一标识符 |

**请求示例**：
```bash
curl -X POST "http://localhost:8080/admin/backups/user001/create"
```

**成功响应**：
```json
{
  "success": true,
  "backup_path": "/backup/user001_backup_20260330_150000.tar.gz"
}
```

**执行流程**：
1. 创建临时备份容器（alpine）
2. 挂载用户数据卷（data + secrets）
3. 打包数据为 tar.gz 格式
4. 创建备份元数据 JSON 文件
5. 记录备份信息到数据库

**备份文件命名**：
- 数据备份：`{user_id}_backup_{timestamp}.tar.gz`
- 敏感数据：`secrets_{user_id}_backup_{timestamp}.tar.gz`
- 元数据：`{user_id}_backup_{timestamp}.json`

**应用场景**：定期备份用户数据或用户主动备份

---

### 4. 资源监控接口

**路径前缀**：`/admin/metrics`  
**后端文件**：`main.py`、`resource_monitor.py`

#### 4.1 获取用户资源指标

```
GET /admin/metrics/{user_id}
```

**代码位置**：`main.py:190`

**功能描述**：获取指定用户 CoPaw 容器的实时资源使用情况。

**路径参数**：
| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| user_id | string | 是 | 用户唯一标识符 |

**请求示例**：
```bash
curl "http://localhost:8080/admin/metrics/user001"
```

**成功响应**：
```json
{
  "success": true,
  "data": {
    "user_id": "user001",
    "container_id": "abc123...",
    "cpu_usage": 15.5,
    "memory_usage": 45.2,
    "memory_used_mb": 463,
    "memory_limit_mb": 1024,
    "disk_usage": 32.8,
    "disk_used_gb": 3.28,
    "disk_limit_gb": 10,
    "uptime_minutes": 120,
    "status": "running",
    "last_collected": "2026-03-30T15:30:00"
  }
}
```

**响应字段说明**：
| 字段 | 类型 | 说明 |
|------|------|------|
| cpu_usage | float | CPU 使用百分比 |
| memory_usage | float | 内存使用百分比 |
| memory_used_mb | int | 已使用内存（MB） |
| memory_limit_mb | int | 内存限制（MB） |
| disk_usage | float | 磁盘使用百分比 |
| disk_used_gb | float | 已使用磁盘（GB） |
| disk_limit_gb | int | 磁盘限制（GB） |
| uptime_minutes | int | 运行时长（分钟） |
| status | string | 容器状态 |

**监控阈值**（在 resource_monitor.py 中配置）：
- 内存警告：1GB
- CPU 警告：80%
- 磁盘警告：90%

**应用场景**：运维监控、资源告警、性能分析

---

### 5. 系统接口

#### 5.1 健康检查

```
GET /health
```

**代码位置**：`main.py:200`

**功能描述**：检查服务运行状态，用于负载均衡器或监控系统探测。

**请求示例**：
```bash
curl "http://localhost:8080/health"
```

**成功响应**：
```json
{
  "status": "healthy",
  "service": "copaw-manager"
}
```

**应用场景**：
- 负载均衡器健康探测
- Kubernetes liveness/readiness 探针
- 监控系统状态检测

---

#### 5.2 静态页面

```
GET /
```

**代码位置**：`main.py:85`

**功能描述**：返回管理界面 HTML 页面。

**请求示例**：
```bash
curl "http://localhost:8080/"
```

**成功响应**：HTML 页面内容

**应用场景**：访问 Web 管理界面

---

## 数据模型

### 用户表 (users)

| 字段 | 类型 | 说明 |
|------|------|------|
| id | string | 用户唯一标识（主键） |
| name | string | 用户显示名称 |
| created_at | timestamp | 创建时间 |
| last_active | timestamp | 最后活跃时间 |

### 实例表 (instances)

| 字段 | 类型 | 说明 |
|------|------|------|
| id | string | 实例唯一标识（主键） |
| user_id | string | 关联用户ID（外键） |
| container_id | string | Docker 容器ID |
| container_name | string | 容器名称 |
| port | int | 分配端口 |
| url | string | 访问地址 |
| status | string | 状态（running/stopped） |
| created_at | timestamp | 创建时间 |
| last_active | timestamp | 最后活跃时间 |

### 备份表 (backups)

| 字段 | 类型 | 说明 |
|------|------|------|
| id | string | 备份唯一标识（主键） |
| user_id | string | 关联用户ID（外键） |
| backup_path | string | 备份文件路径 |
| backup_size | bigint | 备份文件大小（字节） |
| created_at | timestamp | 创建时间 |

---

## 错误处理

### 统一错误响应格式

```json
{
  "detail": "错误描述信息"
}
```

### HTTP 状态码

| 状态码 | 说明 |
|--------|------|
| 200 | 请求成功 |
| 404 | 资源不存在（实例不可用） |
| 500 | 服务器内部错误 |

### 常见错误场景

| 场景 | 状态码 | 错误信息 |
|------|--------|----------|
| 用户不存在 | 500 | "User not found" |
| 实例不可用 | 404 | "Instance not available" |
| 端口分配失败 | 500 | "No available ports in range" |
| 容器启动失败 | 500 | "Failed to start container" |
| 备份失败 | 500 | "Backup failed: {reason}" |

---

## 后端代码索引

| 文件路径 | 功能描述 |
|----------|----------|
| `main.py` | FastAPI 应用入口、API 路由定义 |
| `claw_instance_manager.py` | Docker 容器实例管理（创建、删除、端口分配） |
| `claw_lifespan_manager.py` | 实例生命周期管理（注册、登录、登出、删除） |
| `data_backup_recover.py` | 数据备份与恢复 |
| `resource_monitor.py` | 资源监控（CPU、内存、磁盘） |
| `postgres_client_pg8000.py` | PostgreSQL 数据库客户端 |
| `redis_client.py` | Redis 缓存客户端（可选） |
| `load_balancer.py` | 负载均衡器（预留） |
| `port_is_use.py` | 端口检测工具 |

---

## 部署说明

### 启动服务

```bash
# 开发模式
python main.py

# 生产模式
uvicorn main:app --host 0.0.0.0 --port 8080 --workers 4
```

### 环境依赖

- Python 3.8+
- Docker
- PostgreSQL
- Redis（可选）

### Docker 镜像要求

- 镜像名称：`osclaw:v010`
- 容器端口：8088
- 数据目录：`/app/working`
- 敏感数据目录：`/app/working.secret`

---

*文档版本: 1.0.0*  
*最后更新: 2026-03-30*
