# Semantic-Router Matrix 快速入门指南

本指南帮助你快速部署和配置 semantic-router 的 Matrix 通信功能。

## 快速开始

### 1. Docker Compose 部署 (推荐)

```bash
cd deploy/docker-compose

# 设置环境变量
export MATRIX_REGISTRATION_TOKEN=$(openssl rand -hex 16)
export MATRIX_DOMAIN="matrix.semantic-router.local"

# 启动服务
docker-compose -f docker-compose.matrix.yaml up -d

# 查看日志
docker-compose -f docker-compose.matrix.yaml logs -f
```

### 2. 访问服务

| 服务 | URL | 说明 |
|------|-----|------|
| Element Web | http://localhost:8088 | Matrix 客户端 UI |
| Dashboard | http://localhost:8700 | Semantic Router 控制台 |
| Matrix API | http://localhost:6167 | Matrix 服务器 API |

### 3. 登录 Element Web

1. 打开 http://localhost:8088
2. 点击 "Sign In"
3. 用户名: `admin`
4. 密码: `<MATRIX_REGISTRATION_TOKEN>`

## 配置说明

### 通信模式

在 `config/config.yaml` 中配置:

```yaml
matrix:
  mode: "hybrid"  # native | matrix | hybrid
```

| 模式 | 说明 |
|------|------|
| `native` | 仅使用内置 Room 系统 |
| `matrix` | 仅使用 Matrix 协议 |
| `hybrid` | 混合模式，按 Room 类型自动选择 |

### Room 模式映射

```yaml
matrix:
  room_mode_map:
    "team-*": "matrix"     # 团队 Room 使用 Matrix
    "worker-*": "matrix"   # Worker Room 使用 Matrix
    "*": "native"          # 其他使用内置系统
```

## 常用命令

### 创建 Worker

```bash
# 通过 Dashboard API
curl -X POST http://localhost:8700/api/openclaw/workers \
  -H "Content-Type: application/json" \
  -d '{
    "name": "alice",
    "teamId": "default",
    "identity": {
      "name": "Alice",
      "role": "Frontend Developer",
      "emoji": "👩‍💻"
    }
  }'
```

### 发送消息 (带 @mention)

```bash
# 通过 Dashboard API
curl -X POST http://localhost:8700/api/openclaw/rooms/team-default/messages \
  -H "Content-Type: application/json" \
  -d '{
    "content": "@alice 请实现登录页面",
    "senderType": "user",
    "senderName": "Admin"
  }'

# 直接通过 Matrix API
curl -X PUT "http://localhost:6167/_matrix/client/v3/rooms/!team-default:matrix.semantic-router.local/send/m.room.message/1" \
  -H "Authorization: Bearer ${ADMIN_TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "msgtype": "m.text",
    "body": "@alice 请实现登录页面",
    "m.mentions": {
      "user_ids": ["@alice:matrix.semantic-router.local"]
    }
  }'
```

## 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                      Dashboard UI                            │
│                  (ClawRoomChat.tsx)                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Communication Bridge                       │
│              (matrix_bridge.go)                              │
│                                                              │
│   ┌─────────────────┐        ┌─────────────────┐            │
│   │  Native Store   │◄──────►│  Matrix Client  │            │
│   │  (JSON Files)   │        │  (Tuwunel)      │            │
│   └─────────────────┘        └─────────────────┘            │
└─────────────────────────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
    ┌─────────┐    ┌─────────┐    ┌─────────┐
    │ Leader  │    │ Worker  │    │ Element │
    │(OpenClaw│    │(OpenClaw│    │  Web    │
    └─────────┘    └─────────┘    └─────────┘
```

## 故障排查

### Worker 不响应消息

**原因**: 消息缺少 `m.mentions` 字段

**解决**: 确保消息包含正确的 mentions:

```json
{
  "msgtype": "m.text",
  "body": "@alice 请实现功能",
  "m.mentions": {
    "user_ids": ["@alice:matrix.semantic-router.local"]
  }
}
```

### Matrix 连接失败

**检查**:
1. Tuwunel 服务是否运行: `curl http://localhost:6167/_matrix/client/versions`
2. 注册 Token 是否正确
3. 网络连接是否正常

### 消息不同步

**检查**:
1. `MATRIX_SYNC_TO_MATRIX` 和 `MATRIX_SYNC_FROM_MATRIX` 环境变量
2. MatrixSyncWorker 是否运行
3. Dashboard 日志中的错误信息

## 更多资源

- [完整技术文档](./README.md)
- [HiClaw 技术深度分析](../../docs/hiclaw-analysis.md)
- [Matrix 规范](https://spec.matrix.org/)
- [OpenClaw 文档](https://openclaw.io/docs)
