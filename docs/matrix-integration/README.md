# Semantic-Router Matrix 通信方案设计

本文档详细描述 semantic-router 的 Matrix 通信架构设计，实现 Agent 间的实时通信、人机协作（Human-in-the-Loop），并与现有 dashboard Room 系统完全兼容。

---

## 目录

0. [部署架构理解：vllm-sr serve vs Matrix](#0-部署架构理解vllm-sr-serve-vs-matrix)
1. [架构概览](#1-架构概览)
2. [双模式通信设计](#2-双模式通信设计)
3. [Matrix 协议层](#3-matrix-协议层)
4. [Dashboard 兼容层](#4-dashboard-兼容层)
5. [OpenClaw Matrix 插件配置](#5-openclaw-matrix-插件配置)
6. [部署方案](#6-部署方案)
7. [消息格式与 API](#7-消息格式与-api)
8. [安全模型](#8-安全模型)
9. [迁移指南](#9-迁移指南)

---

## 0. 部署架构理解：vllm-sr serve vs Matrix

### 0.1 vllm-sr serve 是什么？

`vllm-sr serve` 是 **单机一体化部署** 的 CLI 命令，它会启动一个 Docker 容器，内部通过 **Supervisord** 管理多个进程：

```
┌──────────────────────────────────────────────────────────────────────────┐
│                      vllm-sr-container (Docker)                           │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                        Supervisord (进程管理)                        │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                    │                                      │
│         ┌──────────────────────────┼──────────────────────────┐          │
│         │                          │                          │          │
│         ▼                          ▼                          ▼          │
│  ┌────────────────┐      ┌────────────────┐      ┌────────────────────┐ │
│  │   Router (Go)  │      │  Envoy Proxy   │      │ Dashboard Backend  │ │
│  │   :50051 gRPC  │      │ :8888 (user)   │      │     :8700 HTTP     │ │
│  │   :8080 API    │      │  智能路由入口  │      │   配置 UI + API    │ │
│  └────────────────┘      └────────────────┘      └────────────────────┘ │
│                                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐ │
│  │                     Optional Observability Stack                     │ │
│  │    Jaeger (:16686)  +  Prometheus (:9090)  +  Grafana (:3000)       │ │
│  └─────────────────────────────────────────────────────────────────────┘ │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

**启动流程**：
```bash
vllm-sr serve --config config.yaml
```

1. CLI 调用 `docker run` 启动 `vllm-sr-container`
2. 容器内 Supervisord 按优先级启动进程:
   - **Priority 1**: Router (start-router.sh → 生成配置 → /usr/local/bin/router)
   - **Priority 2**: Envoy (配置生成 → envoy -c /etc/envoy/envoy.yaml)
   - **Priority 3**: Access Logs Tail
   - **Priority 4**: Dashboard (start-dashboard.sh → /usr/local/bin/dashboard-backend)
   - **Priority 5**: Log Forwarder (聚合所有日志到 stdout)

### 0.2 vllm-sr serve vs Matrix 的关系

| 维度 | vllm-sr serve | Matrix 通信方案 |
|------|---------------|-----------------|
| **定位** | LLM 请求路由 + 管理控制台 | Agent 间实时通信 + Human-in-the-Loop |
| **用途** | 智能路由 LLM 请求到最优后端 | OpenClaw Agent 协作通信 |
| **通信对象** | 用户 → Envoy → vLLM 后端 | Agent ↔ Agent ↔ Human |
| **协议** | HTTP/gRPC (OpenAI API 兼容) | Matrix Protocol (IM) |
| **是否冲突** | **否，是互补关系** | **否，可以共存** |

### 0.3 完整部署架构（vllm-sr + Matrix + OpenClaw）

```
┌────────────────────────────────────────────────────────────────────────────┐
│                           Complete Architecture                             │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────────┐│
│  │                    Layer 1: LLM Routing (vllm-sr serve)                 ││
│  │                                                                          ││
│  │  User Request ──► Envoy ──► Router (智能选择) ──► vLLM Backend(s)       ││
│  │       │               :8888            │              │                  ││
│  │       │                                │              ├── vLLM-1 (GPT-4) ││
│  │       │                                │              ├── vLLM-2 (Claude)││
│  │       │                                │              └── vLLM-3 (Local) ││
│  │       │                                │                                 ││
│  │       │                         Dashboard :8700                          ││
│  │       │                         (配置 + 监控)                            ││
│  └───────┼──────────────────────────────────────────────────────────────────┘│
│          │                                                                   │
│  ┌───────┼──────────────────────────────────────────────────────────────────┐│
│  │       │         Layer 2: Agent Communication (Matrix)                    ││
│  │       │                                                                   ││
│  │       │    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       ││
│  │       │    │   Tuwunel   │     │ Element Web │     │   MinIO     │       ││
│  │       │    │Matrix Server│     │  Human UI   │     │File Storage │       ││
│  │       │    │   :6167     │     │   :8088     │     │ :9000/:9001 │       ││
│  │       │    └──────┬──────┘     └──────┬──────┘     └──────┬──────┘       ││
│  │       │           │                   │                   │              ││
│  │       │           └───────────────────┼───────────────────┘              ││
│  │       │                               │                                   ││
│  └───────┼───────────────────────────────┼──────────────────────────────────┘│
│          │                               │                                   │
│  ┌───────┼───────────────────────────────┼──────────────────────────────────┐│
│  │       │         Layer 3: Agent Runtime (OpenClaw)                        ││
│  │       │                               │                                   ││
│  │       ▼                               ▼                                   ││
│  │  ┌─────────────────┐          ┌─────────────────┐                        ││
│  │  │  Leader Agent   │◄────────►│  Worker Agent   │                        ││
│  │  │   (OpenClaw)    │  Matrix  │   (OpenClaw)    │                        ││
│  │  │                 │  Rooms   │                 │                        ││
│  │  │  Uses vllm-sr   │          │  Uses vllm-sr   │                        ││
│  │  │  for LLM calls  │          │  for LLM calls  │                        ││
│  │  └────────┬────────┘          └────────┬────────┘                        ││
│  │           │                            │                                  ││
│  │           └────────────┬───────────────┘                                  ││
│  │                        │                                                  ││
│  │              Dashboard Room System                                        ││
│  │              (Unified Room API)                                           ││
│  │                                                                           ││
│  └───────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 0.4 三种部署模式

#### 模式 A：仅 vllm-sr（无 Agent 协作）

```bash
# 适用于：单纯的 LLM 路由场景，无 Agent
vllm-sr serve --config config.yaml

# 端口:
#   8888 - Envoy (LLM API 入口)
#   8700 - Dashboard
#   16686 - Jaeger (可选)
```

#### 模式 B：vllm-sr + Native Room（内置通信）

```bash
# 适用于：需要 Agent 协作，但不需要人类实时介入
vllm-sr serve --config config.yaml

# Dashboard 内置 Room 系统自动启用
# Agent 通过 Dashboard API 通信
# 无需 Matrix 服务器
```

#### 模式 C：vllm-sr + Matrix（完整 Human-in-the-Loop）【推荐】

```yaml
# config.yaml 中启用 Matrix
matrix:
  enabled: true
  domain: "matrix.vllm-sr.local"
  admin_user: "admin"
```

```bash
# 一键启动全栈！Matrix 服务器自动启动
vllm-sr serve --config config.yaml

# 端口:
#   8888 - Envoy (LLM API 入口)
#   8700 - Dashboard (支持 Matrix 模式)
#   6167 - Tuwunel (Matrix Server) ← 自动启动
#   16686 - Jaeger (可选)

# 可选：启动 Element Web 人类客户端
cd deploy/docker-compose
docker-compose -f docker-compose.matrix.yaml up -d
# 访问 http://localhost:8088 登录 Matrix
```

**关键改进**: Matrix 服务器现在由 `vllm-sr serve` **自动启动和管理**，无需单独部署！

### 0.5 一键启动（新设计）

只需在 `config.yaml` 中启用 Matrix，`vllm-sr serve` 会自动：

1. **启动 Tuwunel Matrix 服务器** (端口 6167)
2. **创建系统用户** (@system, @admin, @leader)
3. **配置 OpenClaw 的 Matrix 通道**
4. **注入环境变量到 Dashboard**

```yaml
# config.yaml
version: v0.1

listeners:
  - name: "vllm-router"
    port: 8888
    # ... LLM 路由配置

# 启用 Matrix 通信
matrix:
  enabled: true
  domain: "matrix.vllm-sr.local"
  port: 6167
  admin_user: "admin"
```

```bash
# 一条命令启动全部！
vllm-sr serve --config config.yaml

# 输出包含:
# ✓ Matrix server is healthy
# ✓ Matrix user ready: @system:matrix.vllm-sr.local
# ✓ Matrix user ready: @admin:matrix.vllm-sr.local
# ✓ Matrix user ready: @leader:matrix.vllm-sr.local
#
# Matrix Communication:
#   • Matrix API: http://localhost:6167
#   • Domain: matrix.vllm-sr.local
#   • Admin: @admin:matrix.vllm-sr.local
#   • OpenClaw agents will auto-connect via Matrix
```

### 0.6 数据流对比

#### LLM 请求流（vllm-sr）
```
User App ──► Envoy:8888 ──► Router ──► vLLM Backend ──► Response
                │
                └── Dashboard 可见请求统计
```

#### Agent 通信流（Matrix）
```
Leader ──► Matrix Room ──► Worker
   │            │
   │            └── Human (Element Web) 可见并可介入
   │
   └── 调用 vllm-sr:8888 获取 LLM 响应
```

---

## 1. 架构概览

### 1.1 系统架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Semantic-Router + Matrix 架构                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                         Dashboard Layer                               │   │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐          │   │
│  │  │ ClawRoomChat   │  │  TeamManager   │  │ WorkerManager  │          │   │
│  │  │ (React TSX)    │  │  (React TSX)   │  │ (React TSX)    │          │   │
│  │  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘          │   │
│  │          │                   │                   │                    │   │
│  │          └───────────────────┼───────────────────┘                    │   │
│  │                              │                                        │   │
│  │                   ┌──────────▼──────────┐                             │   │
│  │                   │   Unified Room API   │                            │   │
│  │                   │   /api/openclaw/*    │                            │   │
│  │                   └──────────┬──────────┘                             │   │
│  └──────────────────────────────┼────────────────────────────────────────┘   │
│                                 │                                            │
│  ┌──────────────────────────────▼────────────────────────────────────────┐   │
│  │                      Communication Bridge                              │   │
│  │                                                                        │   │
│  │  ┌─────────────────┐     Mode Switch      ┌─────────────────┐         │   │
│  │  │   Native Room   │◄────────────────────►│  Matrix Bridge  │         │   │
│  │  │   (JSON File)   │                      │  (Tuwunel)      │         │   │
│  │  └────────┬────────┘                      └────────┬────────┘         │   │
│  │           │                                        │                  │   │
│  │           │   mode: "native"                       │  mode: "matrix"  │   │
│  │           │                                        │                  │   │
│  └───────────┼────────────────────────────────────────┼──────────────────┘   │
│              │                                        │                      │
│              ▼                                        ▼                      │
│  ┌──────────────────────┐              ┌────────────────────────────────┐   │
│  │  Local JSON Storage  │              │        Tuwunel Server          │   │
│  │  - rooms.json        │              │  (conduwuit fork)              │   │
│  │  - room-messages/    │              │                                │   │
│  │    *.json            │              │  ┌────────────────────────┐    │   │
│  └──────────────────────┘              │  │  Matrix Rooms          │    │   │
│                                        │  │  - Team Rooms          │    │   │
│                                        │  │  - Worker Rooms        │    │   │
│                                        │  │  - DM Channels         │    │   │
│                                        │  └────────────────────────┘    │   │
│                                        │                                │   │
│                                        │  Port: 6167 (Federation)      │   │
│                                        └────────────────────────────────┘   │
│                                                       │                      │
│                                                       │                      │
│  ┌────────────────────────────────────────────────────┼──────────────────┐   │
│  │                    Agent Runtime Layer             │                  │   │
│  │                                                    │                  │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌──────────────▼────┐             │   │
│  │  │   Leader    │  │   Worker    │  │    Element Web    │             │   │
│  │  │  (OpenClaw) │  │  (OpenClaw) │  │  (Human Client)   │             │   │
│  │  │             │  │             │  │                   │             │   │
│  │  │  matrix:    │  │  matrix:    │  │   Port: 8088      │             │   │
│  │  │  enabled    │  │  enabled    │  │                   │             │   │
│  │  └──────┬──────┘  └──────┬──────┘  └───────────────────┘             │   │
│  │         │                │                                           │   │
│  │         └────────────────┴──────────────────────────────────────────►│   │
│  │                      Matrix Protocol (m.room.message)                │   │
│  └───────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 组件角色表

| 组件 | 职责 | 技术栈 |
|------|------|--------|
| **Dashboard Frontend** | Room UI、消息展示、团队管理 | React + TypeScript |
| **Dashboard Backend** | Room API、消息持久化、WebSocket Hub | Go + Gorilla |
| **Communication Bridge** | 统一消息路由、模式切换 | Go (新增) |
| **Tuwunel Server** | Matrix 协议服务端 | Rust (conduwuit fork) |
| **Element Web** | 人类访问 Matrix 的客户端 | React |
| **OpenClaw Agent** | Agent 运行时 + Matrix 插件 | Node.js |

---

## 2. 双模式通信设计

### 2.1 模式选择策略

semantic-router 支持两种通信模式，可根据部署环境灵活选择：

| 模式 | 适用场景 | 优势 | 劣势 |
|------|----------|------|------|
| **Native (原生)** | 单机开发、简单部署 | 零依赖、快速启动 | 无联邦、无移动端 |
| **Matrix** | 生产环境、多 Agent 协作 | 联邦、审计、移动端 | 需要额外部署 |

### 2.2 配置文件结构

在 `config/config.yaml` 中新增 Matrix 配置节：

```yaml
# config/config.yaml (新增部分)

# Matrix 通信配置
matrix:
  # 通信模式: "native" | "matrix" | "hybrid"
  # - native: 仅使用内置 Room 系统 (JSON 文件存储)
  # - matrix: 仅使用 Matrix 协议
  # - hybrid: 同时支持两种模式，按 Room 类型自动选择
  mode: "hybrid"
  
  # Tuwunel 服务器配置 (mode 为 matrix 或 hybrid 时生效)
  server:
    # 服务器域名 (用于生成 Matrix User ID)
    domain: "matrix.semantic-router.local"
    # 内部服务地址 (Dashboard Backend 直连)
    internal_url: "http://tuwunel:6167"
    # 外部服务地址 (Agent 和 Element Web 访问)
    external_url: "http://localhost:8080/matrix"
    # 注册 Token (用于自动创建用户)
    registration_token: "${MATRIX_REGISTRATION_TOKEN}"
    
  # 用户配置
  users:
    # 系统管理员 (Matrix 消息白名单)
    admin: "admin"
    # 系统账户 (Dashboard Backend 使用)
    system: "system"
    
  # Room 模式映射 (hybrid 模式下使用)
  room_mode_map:
    # team-* 开头的 Room 使用 Matrix
    "team-*": "matrix"
    # worker-* 开头的 Room 使用 Matrix  
    "worker-*": "matrix"
    # 其他使用 native
    "*": "native"
    
  # 消息桥接配置
  bridge:
    # 是否将 native 消息同步到 Matrix
    sync_to_matrix: true
    # 是否将 Matrix 消息同步到 native
    sync_from_matrix: true
    # 消息去重 TTL (秒)
    dedup_ttl: 60
```

### 2.3 Communication Bridge 核心逻辑

```go
// dashboard/backend/handlers/matrix_bridge.go (新增文件)

package handlers

import (
    "context"
    "fmt"
    "regexp"
    "sync"
    "time"
)

// CommunicationMode 定义通信模式
type CommunicationMode string

const (
    ModeNative CommunicationMode = "native"
    ModeMatrix CommunicationMode = "matrix"
    ModeHybrid CommunicationMode = "hybrid"
)

// MatrixBridgeConfig 桥接配置
type MatrixBridgeConfig struct {
    Mode           CommunicationMode
    ServerDomain   string
    InternalURL    string
    ExternalURL    string
    RegToken       string
    AdminUser      string
    SystemUser     string
    RoomModeMap    map[string]CommunicationMode
    SyncToMatrix   bool
    SyncFromMatrix bool
    DedupTTL       time.Duration
}

// MatrixBridge 通信桥接器
type MatrixBridge struct {
    config       MatrixBridgeConfig
    matrixClient *MatrixClient
    nativeStore  *NativeRoomStore
    dedupCache   *DedupCache
    roomModes    []roomModeRule
    mu           sync.RWMutex
}

type roomModeRule struct {
    pattern *regexp.Regexp
    mode    CommunicationMode
}

// NewMatrixBridge 创建通信桥接器
func NewMatrixBridge(config MatrixBridgeConfig) (*MatrixBridge, error) {
    bridge := &MatrixBridge{
        config:     config,
        dedupCache: NewDedupCache(config.DedupTTL),
    }
    
    // 编译 Room 模式规则
    for pattern, mode := range config.RoomModeMap {
        re, err := compileGlobPattern(pattern)
        if err != nil {
            return nil, fmt.Errorf("invalid room pattern %q: %w", pattern, err)
        }
        bridge.roomModes = append(bridge.roomModes, roomModeRule{
            pattern: re,
            mode:    mode,
        })
    }
    
    // 初始化 Matrix 客户端 (如果启用)
    if config.Mode == ModeMatrix || config.Mode == ModeHybrid {
        client, err := NewMatrixClient(MatrixClientConfig{
            HomeserverURL: config.InternalURL,
            Domain:        config.ServerDomain,
            SystemUser:    config.SystemUser,
            RegToken:      config.RegToken,
        })
        if err != nil {
            return nil, fmt.Errorf("failed to init matrix client: %w", err)
        }
        bridge.matrixClient = client
    }
    
    return bridge, nil
}

// GetRoomMode 获取 Room 的通信模式
func (b *MatrixBridge) GetRoomMode(roomID string) CommunicationMode {
    if b.config.Mode != ModeHybrid {
        return b.config.Mode
    }
    
    // 按顺序匹配规则
    for _, rule := range b.roomModes {
        if rule.pattern.MatchString(roomID) {
            return rule.mode
        }
    }
    return ModeNative
}

// SendMessage 发送消息 (自动路由到正确的后端)
func (b *MatrixBridge) SendMessage(ctx context.Context, msg *ClawRoomMessage) error {
    mode := b.GetRoomMode(msg.RoomID)
    
    // 检查去重
    if b.dedupCache.IsDuplicate(msg.ID) {
        return nil
    }
    b.dedupCache.Mark(msg.ID)
    
    switch mode {
    case ModeNative:
        return b.sendNative(ctx, msg)
    case ModeMatrix:
        return b.sendMatrix(ctx, msg)
    default:
        return fmt.Errorf("unknown mode: %s", mode)
    }
}

// sendNative 发送到原生 Room 系统
func (b *MatrixBridge) sendNative(ctx context.Context, msg *ClawRoomMessage) error {
    if err := b.nativeStore.SaveMessage(msg); err != nil {
        return err
    }
    
    // 同步到 Matrix (如果启用)
    if b.config.SyncToMatrix && b.matrixClient != nil {
        go b.syncToMatrix(msg)
    }
    return nil
}

// sendMatrix 发送到 Matrix 服务器
func (b *MatrixBridge) sendMatrix(ctx context.Context, msg *ClawRoomMessage) error {
    matrixMsg := b.convertToMatrixMessage(msg)
    if err := b.matrixClient.SendMessage(ctx, matrixMsg); err != nil {
        return err
    }
    
    // 同步到 Native (如果启用)
    if b.config.SyncFromMatrix {
        go b.syncFromMatrix(msg)
    }
    return nil
}

// convertToMatrixMessage 转换消息格式
func (b *MatrixBridge) convertToMatrixMessage(msg *ClawRoomMessage) *MatrixMessage {
    matrixRoomID := b.mapRoomID(msg.RoomID)
    
    // 构建 m.mentions
    mentions := &MatrixMentions{}
    for _, mention := range msg.Mentions {
        userID := b.mapUserID(mention)
        mentions.UserIDs = append(mentions.UserIDs, userID)
    }
    
    return &MatrixMessage{
        RoomID:   matrixRoomID,
        MsgType:  "m.text",
        Body:     msg.Content,
        Mentions: mentions,
        Metadata: map[string]interface{}{
            "semantic_router.sender_type": msg.SenderType,
            "semantic_router.sender_id":   msg.SenderID,
            "semantic_router.sender_name": msg.SenderName,
            "semantic_router.room_id":     msg.RoomID,
            "semantic_router.team_id":     msg.TeamID,
        },
    }
}

// mapRoomID 映射 Room ID (native → Matrix)
func (b *MatrixBridge) mapRoomID(nativeID string) string {
    // 格式: !<room_id>:<domain>
    return fmt.Sprintf("!%s:%s", nativeID, b.config.ServerDomain)
}

// mapUserID 映射 User ID (native → Matrix)
func (b *MatrixBridge) mapUserID(nativeID string) string {
    // 格式: @<user_id>:<domain>
    return fmt.Sprintf("@%s:%s", nativeID, b.config.ServerDomain)
}

// 辅助函数: 编译 glob 模式为正则
func compileGlobPattern(pattern string) (*regexp.Regexp, error) {
    escaped := regexp.QuoteMeta(pattern)
    escaped = strings.ReplaceAll(escaped, `\*`, `.*`)
    escaped = strings.ReplaceAll(escaped, `\?`, `.`)
    return regexp.Compile("^" + escaped + "$")
}
```

---

## 3. Matrix 协议层

### 3.1 Matrix Client 实现

```go
// dashboard/backend/handlers/matrix_client.go (新增文件)

package handlers

import (
    "bytes"
    "context"
    "encoding/json"
    "fmt"
    "io"
    "net/http"
    "sync"
    "time"
)

// MatrixClientConfig 客户端配置
type MatrixClientConfig struct {
    HomeserverURL string
    Domain        string
    SystemUser    string
    RegToken      string
}

// MatrixClient Matrix 客户端
type MatrixClient struct {
    config      MatrixClientConfig
    httpClient  *http.Client
    accessToken string
    txnID       int64
    mu          sync.Mutex
}

// MatrixMessage Matrix 消息
type MatrixMessage struct {
    RoomID   string                 `json:"-"`
    MsgType  string                 `json:"msgtype"`
    Body     string                 `json:"body"`
    Mentions *MatrixMentions        `json:"m.mentions,omitempty"`
    Metadata map[string]interface{} `json:"-"`
}

// MatrixMentions 提及信息 (MSC3952)
type MatrixMentions struct {
    UserIDs []string `json:"user_ids,omitempty"`
    Room    bool     `json:"room,omitempty"`
}

// NewMatrixClient 创建客户端
func NewMatrixClient(config MatrixClientConfig) (*MatrixClient, error) {
    client := &MatrixClient{
        config: config,
        httpClient: &http.Client{
            Timeout: 30 * time.Second,
        },
    }
    
    // 登录系统账户
    if err := client.login(); err != nil {
        return nil, fmt.Errorf("matrix login failed: %w", err)
    }
    
    return client, nil
}

// login 登录 Matrix 服务器
func (c *MatrixClient) login() error {
    // 首先尝试注册 (幂等)
    if err := c.register(); err != nil {
        // 注册失败可能是已存在，继续尝试登录
    }
    
    loginReq := map[string]interface{}{
        "type": "m.login.password",
        "identifier": map[string]interface{}{
            "type": "m.id.user",
            "user": c.config.SystemUser,
        },
        "password":                      c.config.RegToken,
        "device_id":                     "semantic-router-dashboard",
        "initial_device_display_name":   "Semantic Router Dashboard",
    }
    
    resp, err := c.doRequest("POST", "/_matrix/client/v3/login", loginReq)
    if err != nil {
        return err
    }
    
    var loginResp struct {
        AccessToken string `json:"access_token"`
        UserID      string `json:"user_id"`
    }
    if err := json.Unmarshal(resp, &loginResp); err != nil {
        return err
    }
    
    c.accessToken = loginResp.AccessToken
    return nil
}

// register 注册系统账户
func (c *MatrixClient) register() error {
    regReq := map[string]interface{}{
        "username":            c.config.SystemUser,
        "password":            c.config.RegToken,
        "registration_token":  c.config.RegToken,
        "device_id":           "semantic-router-dashboard",
        "initial_device_display_name": "Semantic Router Dashboard",
    }
    
    _, err := c.doRequest("POST", "/_matrix/client/v3/register", regReq)
    return err
}

// SendMessage 发送消息
func (c *MatrixClient) SendMessage(ctx context.Context, msg *MatrixMessage) error {
    c.mu.Lock()
    c.txnID++
    txnID := c.txnID
    c.mu.Unlock()
    
    endpoint := fmt.Sprintf("/_matrix/client/v3/rooms/%s/send/m.room.message/%d",
        msg.RoomID, txnID)
    
    payload := map[string]interface{}{
        "msgtype": msg.MsgType,
        "body":    msg.Body,
    }
    
    // 添加 m.mentions (关键！Worker 只响应被正确 @ 的消息)
    if msg.Mentions != nil && len(msg.Mentions.UserIDs) > 0 {
        payload["m.mentions"] = msg.Mentions
    }
    
    // 添加自定义元数据
    for k, v := range msg.Metadata {
        payload[k] = v
    }
    
    _, err := c.doRequestWithAuth("PUT", endpoint, payload)
    return err
}

// CreateRoom 创建房间
func (c *MatrixClient) CreateRoom(ctx context.Context, req *CreateRoomRequest) (string, error) {
    payload := map[string]interface{}{
        "name":    req.Name,
        "topic":   req.Topic,
        "invite":  req.Invite,
        "preset":  "trusted_private_chat",
        "visibility": "private",
    }
    
    if req.IsDirect {
        payload["is_direct"] = true
    }
    
    resp, err := c.doRequestWithAuth("POST", "/_matrix/client/v3/createRoom", payload)
    if err != nil {
        return "", err
    }
    
    var createResp struct {
        RoomID string `json:"room_id"`
    }
    if err := json.Unmarshal(resp, &createResp); err != nil {
        return "", err
    }
    
    return createResp.RoomID, nil
}

// CreateRoomRequest 创建房间请求
type CreateRoomRequest struct {
    Name     string
    Topic    string
    Invite   []string
    IsDirect bool
}

// JoinRoom 加入房间
func (c *MatrixClient) JoinRoom(ctx context.Context, roomID string) error {
    _, err := c.doRequestWithAuth("POST", 
        fmt.Sprintf("/_matrix/client/v3/rooms/%s/join", roomID), nil)
    return err
}

// InviteUser 邀请用户
func (c *MatrixClient) InviteUser(ctx context.Context, roomID, userID string) error {
    payload := map[string]interface{}{
        "user_id": userID,
    }
    _, err := c.doRequestWithAuth("POST",
        fmt.Sprintf("/_matrix/client/v3/rooms/%s/invite", roomID), payload)
    return err
}

// doRequest 发送 HTTP 请求
func (c *MatrixClient) doRequest(method, endpoint string, body interface{}) ([]byte, error) {
    return c.doRequestWithToken(method, endpoint, body, "")
}

// doRequestWithAuth 发送带认证的请求
func (c *MatrixClient) doRequestWithAuth(method, endpoint string, body interface{}) ([]byte, error) {
    return c.doRequestWithToken(method, endpoint, body, c.accessToken)
}

func (c *MatrixClient) doRequestWithToken(method, endpoint string, body interface{}, token string) ([]byte, error) {
    url := c.config.HomeserverURL + endpoint
    
    var bodyReader io.Reader
    if body != nil {
        jsonBody, err := json.Marshal(body)
        if err != nil {
            return nil, err
        }
        bodyReader = bytes.NewReader(jsonBody)
    }
    
    req, err := http.NewRequest(method, url, bodyReader)
    if err != nil {
        return nil, err
    }
    
    req.Header.Set("Content-Type", "application/json")
    if token != "" {
        req.Header.Set("Authorization", "Bearer "+token)
    }
    
    resp, err := c.httpClient.Do(req)
    if err != nil {
        return nil, err
    }
    defer resp.Body.Close()
    
    respBody, err := io.ReadAll(resp.Body)
    if err != nil {
        return nil, err
    }
    
    if resp.StatusCode >= 400 {
        return nil, fmt.Errorf("matrix api error %d: %s", resp.StatusCode, string(respBody))
    }
    
    return respBody, nil
}
```

### 3.2 Matrix Sync Worker

```go
// dashboard/backend/handlers/matrix_sync.go (新增文件)

package handlers

import (
    "context"
    "encoding/json"
    "fmt"
    "log"
    "time"
)

// MatrixSyncWorker Matrix 消息同步 Worker
type MatrixSyncWorker struct {
    client       *MatrixClient
    bridge       *MatrixBridge
    nativeHub    *WebSocketHub
    stopCh       chan struct{}
    nextBatch    string
    pollInterval time.Duration
}

// NewMatrixSyncWorker 创建同步 Worker
func NewMatrixSyncWorker(client *MatrixClient, bridge *MatrixBridge, hub *WebSocketHub) *MatrixSyncWorker {
    return &MatrixSyncWorker{
        client:       client,
        bridge:       bridge,
        nativeHub:    hub,
        stopCh:       make(chan struct{}),
        pollInterval: 30 * time.Second, // long-polling timeout
    }
}

// Start 启动同步
func (w *MatrixSyncWorker) Start(ctx context.Context) {
    go w.syncLoop(ctx)
}

// Stop 停止同步
func (w *MatrixSyncWorker) Stop() {
    close(w.stopCh)
}

// syncLoop 同步循环
func (w *MatrixSyncWorker) syncLoop(ctx context.Context) {
    for {
        select {
        case <-w.stopCh:
            return
        case <-ctx.Done():
            return
        default:
            if err := w.doSync(ctx); err != nil {
                log.Printf("matrix sync error: %v", err)
                time.Sleep(5 * time.Second) // 错误后短暂等待
            }
        }
    }
}

// doSync 执行一次同步
func (w *MatrixSyncWorker) doSync(ctx context.Context) error {
    endpoint := fmt.Sprintf("/_matrix/client/v3/sync?timeout=%d", 
        int(w.pollInterval.Milliseconds()))
    if w.nextBatch != "" {
        endpoint += "&since=" + w.nextBatch
    }
    
    resp, err := w.client.doRequestWithAuth("GET", endpoint, nil)
    if err != nil {
        return err
    }
    
    var syncResp MatrixSyncResponse
    if err := json.Unmarshal(resp, &syncResp); err != nil {
        return err
    }
    
    w.nextBatch = syncResp.NextBatch
    
    // 处理 joined rooms 的新消息
    for roomID, roomData := range syncResp.Rooms.Join {
        for _, event := range roomData.Timeline.Events {
            if event.Type == "m.room.message" {
                w.handleRoomMessage(roomID, &event)
            }
        }
    }
    
    return nil
}

// handleRoomMessage 处理 Matrix 房间消息
func (w *MatrixSyncWorker) handleRoomMessage(roomID string, event *MatrixEvent) {
    // 忽略自己发送的消息
    if event.Sender == w.client.config.SystemUser {
        return
    }
    
    // 转换为 native 消息格式
    nativeMsg := w.convertToNativeMessage(roomID, event)
    
    // 同步到 native 系统
    if w.bridge.config.SyncFromMatrix {
        if err := w.bridge.nativeStore.SaveMessage(nativeMsg); err != nil {
            log.Printf("failed to sync matrix message to native: %v", err)
        }
    }
    
    // 广播到 WebSocket 客户端
    w.nativeHub.BroadcastToRoom(nativeMsg.RoomID, WSOutboundMessage{
        Type:    WSTypeNewMessage,
        RoomID:  nativeMsg.RoomID,
        Message: nativeMsg,
    })
}

// convertToNativeMessage 转换 Matrix 消息为 native 格式
func (w *MatrixSyncWorker) convertToNativeMessage(roomID string, event *MatrixEvent) *ClawRoomMessage {
    // 从元数据还原原始信息
    senderType := "user"
    senderID := ""
    senderName := event.Sender
    nativeRoomID := w.bridge.unmapRoomID(roomID)
    teamID := ""
    
    if meta, ok := event.Content["semantic_router.sender_type"].(string); ok {
        senderType = meta
    }
    if meta, ok := event.Content["semantic_router.sender_id"].(string); ok {
        senderID = meta
    }
    if meta, ok := event.Content["semantic_router.sender_name"].(string); ok {
        senderName = meta
    }
    if meta, ok := event.Content["semantic_router.room_id"].(string); ok {
        nativeRoomID = meta
    }
    if meta, ok := event.Content["semantic_router.team_id"].(string); ok {
        teamID = meta
    }
    
    // 提取 mentions
    var mentions []string
    if mentionsData, ok := event.Content["m.mentions"].(map[string]interface{}); ok {
        if userIDs, ok := mentionsData["user_ids"].([]interface{}); ok {
            for _, uid := range userIDs {
                if uidStr, ok := uid.(string); ok {
                    mentions = append(mentions, w.bridge.unmapUserID(uidStr))
                }
            }
        }
    }
    
    body := ""
    if bodyStr, ok := event.Content["body"].(string); ok {
        body = bodyStr
    }
    
    return &ClawRoomMessage{
        ID:         event.EventID,
        RoomID:     nativeRoomID,
        TeamID:     teamID,
        SenderType: senderType,
        SenderID:   senderID,
        SenderName: senderName,
        Content:    body,
        Mentions:   mentions,
        CreatedAt:  time.UnixMilli(event.OriginServerTS).Format(time.RFC3339),
    }
}

// MatrixSyncResponse Matrix sync 响应
type MatrixSyncResponse struct {
    NextBatch string `json:"next_batch"`
    Rooms     struct {
        Join map[string]struct {
            Timeline struct {
                Events []MatrixEvent `json:"events"`
            } `json:"timeline"`
        } `json:"join"`
    } `json:"rooms"`
}

// MatrixEvent Matrix 事件
type MatrixEvent struct {
    Type           string                 `json:"type"`
    EventID        string                 `json:"event_id"`
    Sender         string                 `json:"sender"`
    OriginServerTS int64                  `json:"origin_server_ts"`
    Content        map[string]interface{} `json:"content"`
}
```

---

## 4. Dashboard 兼容层

### 4.1 修改 Room Handler

在现有的 `openclaw_rooms.go` 中集成 Matrix Bridge：

```go
// dashboard/backend/handlers/openclaw_rooms.go (修改部分)

// OpenClawHandler 添加 Matrix Bridge 字段
type OpenClawHandler struct {
    // ... 现有字段 ...
    matrixBridge *MatrixBridge  // 新增
}

// NewOpenClawHandler 修改构造函数
func NewOpenClawHandler(cfg OpenClawConfig) (*OpenClawHandler, error) {
    h := &OpenClawHandler{
        // ... 现有初始化 ...
    }
    
    // 初始化 Matrix Bridge (如果配置启用)
    if cfg.MatrixEnabled {
        bridge, err := NewMatrixBridge(MatrixBridgeConfig{
            Mode:           cfg.MatrixMode,
            ServerDomain:   cfg.MatrixDomain,
            InternalURL:    cfg.MatrixInternalURL,
            ExternalURL:    cfg.MatrixExternalURL,
            RegToken:       cfg.MatrixRegToken,
            AdminUser:      cfg.MatrixAdminUser,
            SystemUser:     cfg.MatrixSystemUser,
            RoomModeMap:    cfg.MatrixRoomModeMap,
            SyncToMatrix:   cfg.MatrixSyncToMatrix,
            SyncFromMatrix: cfg.MatrixSyncFromMatrix,
            DedupTTL:       time.Duration(cfg.MatrixDedupTTL) * time.Second,
        })
        if err != nil {
            return nil, fmt.Errorf("failed to init matrix bridge: %w", err)
        }
        h.matrixBridge = bridge
    }
    
    return h, nil
}

// RoomMessagesHandler 修改消息处理
func (h *OpenClawHandler) RoomMessagesHandler() http.HandlerFunc {
    return func(w http.ResponseWriter, r *http.Request) {
        roomID := extractRoomID(r.URL.Path)
        
        switch r.Method {
        case http.MethodGet:
            // 根据模式获取消息
            mode := h.getRoomMode(roomID)
            var messages []ClawRoomMessage
            var err error
            
            if mode == ModeMatrix && h.matrixBridge != nil {
                messages, err = h.matrixBridge.GetMessages(r.Context(), roomID)
            } else {
                messages, err = h.loadRoomMessages(roomID)
            }
            
            if err != nil {
                writeJSONError(w, err.Error(), http.StatusInternalServerError)
                return
            }
            
            w.Header().Set("Content-Type", "application/json")
            json.NewEncoder(w).Encode(messages)
            
        case http.MethodPost:
            var payload clawRoomMessagePayload
            if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
                writeJSONError(w, err.Error(), http.StatusBadRequest)
                return
            }
            
            msg := h.buildMessage(roomID, payload)
            
            // 通过 Bridge 发送 (自动路由)
            if h.matrixBridge != nil {
                if err := h.matrixBridge.SendMessage(r.Context(), msg); err != nil {
                    writeJSONError(w, err.Error(), http.StatusInternalServerError)
                    return
                }
            } else {
                // 回退到原生模式
                if err := h.saveMessage(roomID, msg); err != nil {
                    writeJSONError(w, err.Error(), http.StatusInternalServerError)
                    return
                }
            }
            
            w.Header().Set("Content-Type", "application/json")
            w.WriteHeader(http.StatusCreated)
            json.NewEncoder(w).Encode(msg)
        }
    }
}

// getRoomMode 获取 Room 的通信模式
func (h *OpenClawHandler) getRoomMode(roomID string) CommunicationMode {
    if h.matrixBridge == nil {
        return ModeNative
    }
    return h.matrixBridge.GetRoomMode(roomID)
}
```

### 4.2 前端 ClawRoomChat 兼容

前端组件 `ClawRoomChat.tsx` 无需修改，因为 API 接口保持不变。但可以添加 Matrix 状态指示器：

```tsx
// dashboard/frontend/src/components/ClawRoomChat.tsx (可选增强)

// 新增 Matrix 状态类型
interface RoomMetadata {
  id: string;
  name: string;
  // 新增字段
  communicationMode?: 'native' | 'matrix' | 'hybrid';
  matrixRoomId?: string;
}

// 在 Room 头部显示通信模式
const RoomHeader: React.FC<{ room: RoomMetadata }> = ({ room }) => {
  return (
    <div className={styles.roomHeader}>
      <h3>{room.name}</h3>
      {room.communicationMode === 'matrix' && (
        <span className={styles.matrixBadge} title={`Matrix Room: ${room.matrixRoomId}`}>
          🔗 Matrix
        </span>
      )}
    </div>
  );
};
```

---

## 5. OpenClaw Matrix 插件配置

### 5.1 Leader Agent 配置

```json
{
  "$schema": "https://openclaw.io/schemas/config.json",
  "name": "leader",
  "version": "1.0.0",
  
  "gateway": {
    "mode": "local",
    "port": 18799,
    "auth": {
      "token": "${OPENCLAW_GATEWAY_TOKEN}"
    }
  },
  
  "channels": {
    "matrix": {
      "enabled": true,
      "homeserver": "${MATRIX_HOMESERVER_URL}",
      "accessToken": "${LEADER_MATRIX_ACCESS_TOKEN}",
      
      "dm": {
        "policy": "allowlist",
        "allowFrom": [
          "@admin:${MATRIX_DOMAIN}",
          "@system:${MATRIX_DOMAIN}"
        ]
      },
      
      "groupPolicy": "allowlist",
      "groupAllowFrom": [
        "@admin:${MATRIX_DOMAIN}",
        "@system:${MATRIX_DOMAIN}"
      ],
      
      "groups": {
        "*": {
          "allow": true,
          "requireMention": true
        }
      }
    }
  },
  
  "skills": {
    "autoLoad": true,
    "directory": "${OPENCLAW_SKILLS_DIR}"
  },
  
  "session": {
    "resetByType": {
      "dm": { "mode": "daily", "atHour": 4 },
      "group": { "mode": "idle", "idleMinutes": 2880 }
    }
  },
  
  "llm": {
    "provider": "openai-compatible",
    "baseUrl": "${LLM_API_BASE_URL}",
    "apiKey": "${LLM_API_KEY}",
    "model": "${LLM_MODEL}"
  }
}
```

### 5.2 Worker Agent 配置

```json
{
  "$schema": "https://openclaw.io/schemas/config.json",
  "name": "${WORKER_NAME}",
  "version": "1.0.0",
  
  "gateway": {
    "mode": "local",
    "port": 18800,
    "auth": {
      "token": "${OPENCLAW_GATEWAY_TOKEN}"
    }
  },
  
  "channels": {
    "matrix": {
      "enabled": true,
      "homeserver": "${MATRIX_HOMESERVER_URL}",
      "accessToken": "${WORKER_MATRIX_ACCESS_TOKEN}",
      
      "dm": {
        "policy": "allowlist",
        "allowFrom": []
      },
      
      "groupPolicy": "allowlist",
      "groupAllowFrom": [
        "@leader:${MATRIX_DOMAIN}",
        "@admin:${MATRIX_DOMAIN}",
        "@system:${MATRIX_DOMAIN}"
      ],
      
      "groups": {
        "*": {
          "allow": true,
          "requireMention": true
        }
      }
    }
  },
  
  "skills": {
    "autoLoad": true,
    "directory": "${OPENCLAW_SKILLS_DIR}"
  },
  
  "session": {
    "resetByType": {
      "dm": { "mode": "never" },
      "group": { "mode": "idle", "idleMinutes": 1440 }
    }
  },
  
  "llm": {
    "provider": "openai-compatible",
    "baseUrl": "${LLM_API_BASE_URL}",
    "apiKey": "${LLM_API_KEY}",
    "model": "${LLM_MODEL}"
  }
}
```

### 5.3 OpenClaw Matrix 插件安装脚本

```bash
#!/bin/bash
# deploy/scripts/setup-openclaw-matrix.sh

set -euo pipefail

# 配置变量
MATRIX_DOMAIN="${MATRIX_DOMAIN:-matrix.semantic-router.local}"
MATRIX_SERVER="${MATRIX_SERVER:-http://tuwunel:6167}"
REGISTRATION_TOKEN="${MATRIX_REGISTRATION_TOKEN:-$(openssl rand -hex 16)}"
ADMIN_USER="${MATRIX_ADMIN_USER:-admin}"

echo "🔧 Setting up OpenClaw Matrix integration..."

# 1. 创建系统用户
create_matrix_user() {
    local username="$1"
    local password="${REGISTRATION_TOKEN}"
    
    echo "Creating Matrix user: @${username}:${MATRIX_DOMAIN}"
    
    curl -s -X POST "${MATRIX_SERVER}/_matrix/client/v3/register" \
        -H "Content-Type: application/json" \
        -d "{
            \"username\": \"${username}\",
            \"password\": \"${password}\",
            \"registration_token\": \"${REGISTRATION_TOKEN}\",
            \"device_id\": \"semantic-router-${username}\",
            \"initial_device_display_name\": \"Semantic Router ${username}\"
        }" | jq .
}

# 2. 获取访问令牌
get_access_token() {
    local username="$1"
    local password="${REGISTRATION_TOKEN}"
    
    curl -s -X POST "${MATRIX_SERVER}/_matrix/client/v3/login" \
        -H "Content-Type: application/json" \
        -d "{
            \"type\": \"m.login.password\",
            \"identifier\": {
                \"type\": \"m.id.user\",
                \"user\": \"${username}\"
            },
            \"password\": \"${password}\",
            \"device_id\": \"semantic-router-${username}\",
            \"initial_device_display_name\": \"Semantic Router ${username}\"
        }" | jq -r '.access_token'
}

# 3. 创建系统用户
echo "📝 Creating system users..."
create_matrix_user "system"
create_matrix_user "admin"
create_matrix_user "leader"

# 4. 获取访问令牌
echo "🔑 Obtaining access tokens..."
SYSTEM_TOKEN=$(get_access_token "system")
ADMIN_TOKEN=$(get_access_token "admin")
LEADER_TOKEN=$(get_access_token "leader")

# 5. 生成配置文件
echo "📄 Generating configuration files..."

cat > /tmp/matrix-config.env << EOF
# Matrix Configuration
MATRIX_DOMAIN=${MATRIX_DOMAIN}
MATRIX_HOMESERVER_URL=${MATRIX_SERVER}
MATRIX_REGISTRATION_TOKEN=${REGISTRATION_TOKEN}

# System Tokens
SYSTEM_MATRIX_ACCESS_TOKEN=${SYSTEM_TOKEN}
ADMIN_MATRIX_ACCESS_TOKEN=${ADMIN_TOKEN}
LEADER_MATRIX_ACCESS_TOKEN=${LEADER_TOKEN}
EOF

echo "✅ Matrix setup complete!"
echo "Configuration saved to: /tmp/matrix-config.env"
echo ""
echo "Next steps:"
echo "1. Source the configuration: source /tmp/matrix-config.env"
echo "2. Update your deployment with the new environment variables"
echo "3. Restart the Dashboard and OpenClaw agents"
```

---

## 6. 部署方案

### 6.1 Docker Compose 部署

```yaml
# deploy/docker-compose/docker-compose.matrix.yaml

version: "3.8"

services:
  # Tuwunel Matrix 服务器
  tuwunel:
    image: ghcr.io/tuwunel/tuwunel:latest
    container_name: semantic-router-matrix
    restart: unless-stopped
    environment:
      # 服务器配置
      CONDUWUIT_SERVER_NAME: "${MATRIX_DOMAIN:-matrix.semantic-router.local}"
      CONDUWUIT_DATABASE_PATH: "/data/tuwunel"
      CONDUWUIT_PORT: "6167"
      
      # 注册配置
      CONDUWUIT_ALLOW_REGISTRATION: "true"
      CONDUWUIT_REGISTRATION_TOKEN: "${MATRIX_REGISTRATION_TOKEN}"
      
      # 性能配置
      CONDUWUIT_MAX_REQUEST_SIZE: "52428800"
      CONDUWUIT_MAX_CONCURRENT_REQUESTS: "500"
      
      # 日志配置
      CONDUWUIT_LOG: "info"
      
    volumes:
      - tuwunel-data:/data/tuwunel
    ports:
      - "6167:6167"
    networks:
      - semantic-router
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6167/_matrix/client/versions"]
      interval: 30s
      timeout: 10s
      retries: 3

  # Element Web 客户端 (可选)
  element-web:
    image: vectorim/element-web:latest
    container_name: semantic-router-element
    restart: unless-stopped
    volumes:
      - ./element-config.json:/app/config.json:ro
    ports:
      - "8088:80"
    networks:
      - semantic-router
    depends_on:
      - tuwunel

  # Dashboard 后端 (更新配置)
  dashboard:
    image: semantic-router-dashboard:latest
    container_name: semantic-router-dashboard
    restart: unless-stopped
    environment:
      DASHBOARD_PORT: "8700"
      
      # Matrix 配置
      MATRIX_ENABLED: "true"
      MATRIX_MODE: "hybrid"
      MATRIX_DOMAIN: "${MATRIX_DOMAIN:-matrix.semantic-router.local}"
      MATRIX_INTERNAL_URL: "http://tuwunel:6167"
      MATRIX_EXTERNAL_URL: "http://localhost:8080/matrix"
      MATRIX_REGISTRATION_TOKEN: "${MATRIX_REGISTRATION_TOKEN}"
      MATRIX_SYSTEM_USER: "system"
      MATRIX_ADMIN_USER: "admin"
      
      # 其他现有配置...
      TARGET_ROUTER_API_URL: "http://semantic-router:8080"
      
    volumes:
      - dashboard-data:/app/data
    ports:
      - "8700:8700"
    networks:
      - semantic-router
    depends_on:
      - tuwunel

  # Leader Agent (OpenClaw)
  leader:
    image: openclaw:latest
    container_name: semantic-router-leader
    restart: unless-stopped
    environment:
      OPENCLAW_CONFIG_PATH: "/config/leader-openclaw.json"
      MATRIX_HOMESERVER_URL: "http://tuwunel:6167"
      MATRIX_DOMAIN: "${MATRIX_DOMAIN:-matrix.semantic-router.local}"
      LEADER_MATRIX_ACCESS_TOKEN: "${LEADER_MATRIX_ACCESS_TOKEN}"
      LLM_API_BASE_URL: "${LLM_API_BASE_URL}"
      LLM_API_KEY: "${LLM_API_KEY}"
      LLM_MODEL: "${LLM_MODEL:-qwen2.5:7b}"
    volumes:
      - ./config/leader-openclaw.json:/config/leader-openclaw.json:ro
      - leader-workspace:/workspace
    ports:
      - "18799:18799"
    networks:
      - semantic-router
    depends_on:
      - tuwunel
      - dashboard

volumes:
  tuwunel-data:
  dashboard-data:
  leader-workspace:

networks:
  semantic-router:
    driver: bridge
```

### 6.2 Element Web 配置

```json
{
  "default_server_config": {
    "m.homeserver": {
      "base_url": "http://localhost:6167",
      "server_name": "matrix.semantic-router.local"
    }
  },
  "brand": "Semantic Router Matrix",
  "integrations_ui_url": "",
  "integrations_rest_url": "",
  "integrations_widgets_urls": [],
  "bug_report_endpoint_url": "",
  "show_labs_settings": false,
  "features": {
    "feature_thread": true,
    "feature_pinning": true
  },
  "default_country_code": "CN",
  "room_directory": {
    "servers": ["matrix.semantic-router.local"]
  },
  "setting_defaults": {
    "breadcrumbs": true
  }
}
```

### 6.3 Kubernetes 部署

```yaml
# deploy/kubernetes/matrix/tuwunel-deployment.yaml

apiVersion: v1
kind: Namespace
metadata:
  name: semantic-router-matrix

---
apiVersion: v1
kind: Secret
metadata:
  name: matrix-secrets
  namespace: semantic-router-matrix
type: Opaque
stringData:
  registration-token: "${MATRIX_REGISTRATION_TOKEN}"

---
apiVersion: v1
kind: ConfigMap
metadata:
  name: tuwunel-config
  namespace: semantic-router-matrix
data:
  CONDUWUIT_SERVER_NAME: "matrix.semantic-router.svc.cluster.local"
  CONDUWUIT_DATABASE_PATH: "/data/tuwunel"
  CONDUWUIT_PORT: "6167"
  CONDUWUIT_ALLOW_REGISTRATION: "true"
  CONDUWUIT_MAX_REQUEST_SIZE: "52428800"
  CONDUWUIT_MAX_CONCURRENT_REQUESTS: "500"
  CONDUWUIT_LOG: "info"

---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: tuwunel
  namespace: semantic-router-matrix
spec:
  serviceName: tuwunel
  replicas: 1
  selector:
    matchLabels:
      app: tuwunel
  template:
    metadata:
      labels:
        app: tuwunel
    spec:
      containers:
        - name: tuwunel
          image: ghcr.io/tuwunel/tuwunel:latest
          ports:
            - containerPort: 6167
              name: matrix
          envFrom:
            - configMapRef:
                name: tuwunel-config
          env:
            - name: CONDUWUIT_REGISTRATION_TOKEN
              valueFrom:
                secretKeyRef:
                  name: matrix-secrets
                  key: registration-token
          volumeMounts:
            - name: data
              mountPath: /data/tuwunel
          resources:
            requests:
              memory: "256Mi"
              cpu: "100m"
            limits:
              memory: "1Gi"
              cpu: "500m"
          livenessProbe:
            httpGet:
              path: /_matrix/client/versions
              port: 6167
            initialDelaySeconds: 10
            periodSeconds: 30
          readinessProbe:
            httpGet:
              path: /_matrix/client/versions
              port: 6167
            initialDelaySeconds: 5
            periodSeconds: 10
  volumeClaimTemplates:
    - metadata:
        name: data
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: 10Gi

---
apiVersion: v1
kind: Service
metadata:
  name: tuwunel
  namespace: semantic-router-matrix
spec:
  selector:
    app: tuwunel
  ports:
    - port: 6167
      targetPort: 6167
      name: matrix
  type: ClusterIP

---
# Element Web Deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: element-web
  namespace: semantic-router-matrix
spec:
  replicas: 1
  selector:
    matchLabels:
      app: element-web
  template:
    metadata:
      labels:
        app: element-web
    spec:
      containers:
        - name: element-web
          image: vectorim/element-web:latest
          ports:
            - containerPort: 80
          volumeMounts:
            - name: config
              mountPath: /app/config.json
              subPath: config.json
      volumes:
        - name: config
          configMap:
            name: element-config

---
apiVersion: v1
kind: Service
metadata:
  name: element-web
  namespace: semantic-router-matrix
spec:
  selector:
    app: element-web
  ports:
    - port: 80
      targetPort: 80
  type: ClusterIP
```

### 6.4 Helm Values 扩展

```yaml
# deploy/helm/semantic-router/values-matrix.yaml (新增文件)

# Matrix 集成配置
matrix:
  enabled: true
  
  # 通信模式
  mode: "hybrid"  # native | matrix | hybrid
  
  # Tuwunel 服务器
  tuwunel:
    enabled: true
    image:
      repository: ghcr.io/tuwunel/tuwunel
      tag: latest
      pullPolicy: IfNotPresent
    
    # 服务器域名
    domain: "matrix.semantic-router.local"
    
    # 资源配置
    resources:
      requests:
        memory: "256Mi"
        cpu: "100m"
      limits:
        memory: "1Gi"
        cpu: "500m"
    
    # 持久化存储
    persistence:
      enabled: true
      size: 10Gi
      storageClass: ""
    
    # 服务配置
    service:
      type: ClusterIP
      port: 6167
  
  # Element Web 客户端
  element:
    enabled: true
    image:
      repository: vectorim/element-web
      tag: latest
    
    service:
      type: ClusterIP
      port: 80
    
    ingress:
      enabled: false
      # ... ingress 配置
  
  # 系统用户配置
  users:
    admin: "admin"
    system: "system"
    leader: "leader"
  
  # Bridge 配置
  bridge:
    syncToMatrix: true
    syncFromMatrix: true
    dedupTTL: 60

# Dashboard Matrix 集成
dashboard:
  matrix:
    enabled: true
    internalUrl: "http://tuwunel:6167"
    externalUrl: "/matrix"
    
  # Room 模式映射
  roomModeMap:
    "team-*": "matrix"
    "worker-*": "matrix"
    "*": "native"
```

---

## 7. 消息格式与 API

### 7.1 统一消息格式

```typescript
// 统一消息格式 (前后端通用)
interface UnifiedMessage {
  // 基础字段
  id: string;
  roomId: string;
  teamId?: string;
  
  // 发送者信息
  senderType: 'user' | 'leader' | 'worker' | 'system';
  senderId?: string;
  senderName: string;
  
  // 消息内容
  content: string;
  mentions?: string[];  // @提及的用户/角色
  
  // 时间戳
  createdAt: string;
  
  // 元数据
  metadata?: Record<string, string>;
  
  // Matrix 扩展字段 (仅 Matrix 模式)
  matrixEventId?: string;
  matrixRoomId?: string;
}

// API 响应格式
interface RoomApiResponse {
  room: {
    id: string;
    name: string;
    teamId: string;
    communicationMode: 'native' | 'matrix';
    matrixRoomId?: string;
    createdAt: string;
    updatedAt: string;
  };
  messages: UnifiedMessage[];
  pagination?: {
    total: number;
    offset: number;
    limit: number;
    hasMore: boolean;
  };
}
```

### 7.2 @Mention 规范

```typescript
// @mention 类型
type MentionType = 
  | '@all'           // 所有成员
  | '@leader'        // Leader Agent
  | '@workers'       // 所有 Workers
  | `@${string}`;    // 特定用户/Worker

// @mention 解析
function parseMentions(content: string): string[] {
  const mentionRegex = /@([a-zA-Z0-9_.-]+)/g;
  const mentions: string[] = [];
  let match;
  
  while ((match = mentionRegex.exec(content)) !== null) {
    mentions.push(match[1]);
  }
  
  return mentions;
}

// Matrix 格式转换
function toMatrixMentions(mentions: string[], domain: string): MatrixMentions {
  const userIds = mentions
    .filter(m => !['all', 'leader', 'workers'].includes(m))
    .map(m => `@${m}:${domain}`);
  
  const room = mentions.includes('all');
  
  return { user_ids: userIds, room };
}
```

### 7.3 API 端点对照

| 操作 | Native API | Matrix API |
|------|------------|------------|
| 获取房间列表 | `GET /api/openclaw/rooms` | 自动代理 |
| 创建房间 | `POST /api/openclaw/rooms` | `POST /_matrix/client/v3/createRoom` |
| 发送消息 | `POST /api/openclaw/rooms/{id}/messages` | `PUT /_matrix/client/v3/rooms/{id}/send/m.room.message/{txnId}` |
| 获取消息 | `GET /api/openclaw/rooms/{id}/messages` | `GET /_matrix/client/v3/rooms/{id}/messages` |
| WebSocket | `GET /api/openclaw/rooms/{id}/ws` | Matrix Sync API |
| SSE 流 | `GET /api/openclaw/rooms/{id}/stream` | 通过 Bridge 转换 |

---

## 8. 安全模型

### 8.1 认证流程

```
┌─────────────────────────────────────────────────────────────────┐
│                      Authentication Flow                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. 用户/Agent 注册                                              │
│  ┌─────────────┐    Registration Token    ┌─────────────────┐   │
│  │ OpenClaw    │ ─────────────────────────► │  Tuwunel       │   │
│  │ / Dashboard │                            │  Matrix Server │   │
│  └─────────────┘                            └─────────────────┘   │
│                                                                  │
│  2. 登录获取 Access Token                                        │
│  ┌─────────────┐    m.login.password      ┌─────────────────┐   │
│  │ OpenClaw    │ ─────────────────────────► │  Tuwunel       │   │
│  │ / Dashboard │ ◄───────────────────────── │  Matrix Server │   │
│  └─────────────┘    { access_token }       └─────────────────┘   │
│                                                                  │
│  3. 发送消息 (带 Access Token)                                   │
│  ┌─────────────┐    Bearer {token}        ┌─────────────────┐   │
│  │ OpenClaw    │ ─────────────────────────► │  Tuwunel       │   │
│  └─────────────┘                            └─────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 8.2 权限矩阵

| 角色 | 创建房间 | 发送消息 | 邀请用户 | 管理 Worker |
|------|---------|---------|---------|------------|
| Admin | ✅ | ✅ | ✅ | ✅ |
| Leader | ✅ | ✅ | ✅ | ✅ |
| Worker | ❌ | ✅ (仅被 @时) | ❌ | ❌ |
| System | ✅ | ✅ | ✅ | ❌ |

### 8.3 Token 管理

```go
// dashboard/backend/handlers/matrix_token_manager.go

package handlers

import (
    "sync"
    "time"
)

// TokenManager Matrix Token 管理器
type TokenManager struct {
    tokens    map[string]*TokenEntry
    mu        sync.RWMutex
    refresher *TokenRefresher
}

type TokenEntry struct {
    UserID      string
    AccessToken string
    DeviceID    string
    ExpiresAt   time.Time
    RefreshAt   time.Time
}

// GetOrRefresh 获取或刷新 Token
func (m *TokenManager) GetOrRefresh(userID string) (string, error) {
    m.mu.RLock()
    entry, exists := m.tokens[userID]
    m.mu.RUnlock()
    
    if !exists || time.Now().After(entry.ExpiresAt) {
        return m.refresh(userID)
    }
    
    // 预刷新 (提前 5 分钟)
    if time.Now().After(entry.RefreshAt) {
        go m.refresh(userID)
    }
    
    return entry.AccessToken, nil
}

// refresh 刷新 Token
func (m *TokenManager) refresh(userID string) (string, error) {
    m.mu.Lock()
    defer m.mu.Unlock()
    
    // 重新登录获取新 Token
    token, err := m.refresher.Login(userID)
    if err != nil {
        return "", err
    }
    
    m.tokens[userID] = &TokenEntry{
        UserID:      userID,
        AccessToken: token.AccessToken,
        DeviceID:    token.DeviceID,
        ExpiresAt:   time.Now().Add(24 * time.Hour),
        RefreshAt:   time.Now().Add(23 * time.Hour),
    }
    
    return token.AccessToken, nil
}
```

---

## 9. 迁移指南

### 9.1 从 Native 迁移到 Hybrid

```bash
#!/bin/bash
# deploy/scripts/migrate-to-hybrid.sh

set -euo pipefail

echo "🔄 Migrating from Native to Hybrid mode..."

# 1. 部署 Tuwunel
echo "Step 1: Deploying Tuwunel Matrix server..."
docker-compose -f docker-compose.matrix.yaml up -d tuwunel

# 等待服务就绪
echo "Waiting for Tuwunel to be ready..."
until curl -s http://localhost:6167/_matrix/client/versions > /dev/null; do
    sleep 2
done

# 2. 创建系统用户
echo "Step 2: Creating system users..."
./setup-openclaw-matrix.sh

# 3. 同步现有房间到 Matrix
echo "Step 3: Syncing existing rooms to Matrix..."
./sync-rooms-to-matrix.sh

# 4. 更新 Dashboard 配置
echo "Step 4: Updating Dashboard configuration..."
cat >> .env << EOF
MATRIX_ENABLED=true
MATRIX_MODE=hybrid
EOF

# 5. 重启服务
echo "Step 5: Restarting services..."
docker-compose restart dashboard

echo "✅ Migration complete!"
echo ""
echo "Next steps:"
echo "1. Access Element Web at http://localhost:8088"
echo "2. Test room messaging in both modes"
echo "3. Verify Worker @mention responses"
```

### 9.2 Room 同步脚本

```bash
#!/bin/bash
# deploy/scripts/sync-rooms-to-matrix.sh

set -euo pipefail

DASHBOARD_API="${DASHBOARD_API:-http://localhost:8700}"
MATRIX_SERVER="${MATRIX_SERVER:-http://localhost:6167}"
SYSTEM_TOKEN="${SYSTEM_MATRIX_ACCESS_TOKEN}"

# 获取现有房间列表
rooms=$(curl -s "${DASHBOARD_API}/api/openclaw/rooms" | jq -c '.[]')

for room in $rooms; do
    room_id=$(echo "$room" | jq -r '.id')
    room_name=$(echo "$room" | jq -r '.name')
    team_id=$(echo "$room" | jq -r '.teamId')
    
    echo "Syncing room: ${room_name} (${room_id})"
    
    # 创建对应的 Matrix 房间
    matrix_room_id=$(curl -s -X POST "${MATRIX_SERVER}/_matrix/client/v3/createRoom" \
        -H "Authorization: Bearer ${SYSTEM_TOKEN}" \
        -H "Content-Type: application/json" \
        -d "{
            \"name\": \"${room_name}\",
            \"topic\": \"Semantic Router Room: ${room_id}\",
            \"preset\": \"trusted_private_chat\",
            \"creation_content\": {
                \"semantic_router.room_id\": \"${room_id}\",
                \"semantic_router.team_id\": \"${team_id}\"
            }
        }" | jq -r '.room_id')
    
    echo "Created Matrix room: ${matrix_room_id}"
    
    # 同步消息历史 (可选)
    # ./sync-room-messages.sh "${room_id}" "${matrix_room_id}"
done

echo "✅ Room sync complete!"
```

---

## 附录

### A. 环境变量参考

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `MATRIX_ENABLED` | 是否启用 Matrix | `false` |
| `MATRIX_MODE` | 通信模式 | `native` |
| `MATRIX_DOMAIN` | Matrix 服务器域名 | `matrix.semantic-router.local` |
| `MATRIX_INTERNAL_URL` | 内部服务地址 | `http://tuwunel:6167` |
| `MATRIX_EXTERNAL_URL` | 外部访问地址 | `http://localhost:8080/matrix` |
| `MATRIX_REGISTRATION_TOKEN` | 注册 Token | - |
| `MATRIX_SYSTEM_USER` | 系统账户 | `system` |
| `MATRIX_ADMIN_USER` | 管理员账户 | `admin` |
| `MATRIX_SYNC_TO_MATRIX` | 同步到 Matrix | `true` |
| `MATRIX_SYNC_FROM_MATRIX` | 从 Matrix 同步 | `true` |
| `MATRIX_DEDUP_TTL` | 消息去重 TTL (秒) | `60` |

### B. 故障排查

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| Worker 不响应 | 缺少 `m.mentions` | 确保消息包含正确的 `m.mentions.user_ids` |
| 注册失败 | Token 不正确 | 检查 `CONDUWUIT_REGISTRATION_TOKEN` |
| 消息不同步 | Bridge 配置错误 | 检查 `MATRIX_SYNC_*` 配置 |
| 连接超时 | 网络问题 | 检查 Tuwunel 服务是否正常运行 |

### C. 性能调优

```yaml
# Tuwunel 性能配置
CONDUWUIT_MAX_REQUEST_SIZE: "52428800"      # 50MB
CONDUWUIT_MAX_CONCURRENT_REQUESTS: "500"    # 并发请求数
CONDUWUIT_ROCKSDB_CACHE_CAPACITY_MB: "256"  # 缓存大小

# Dashboard Bridge 配置
MATRIX_BRIDGE_WORKERS: "4"                   # Bridge worker 数量
MATRIX_SYNC_BATCH_SIZE: "100"                # 同步批量大小
MATRIX_SYNC_INTERVAL: "1000"                 # 同步间隔 (ms)
```

---

*文档版本：1.0.0*
*最后更新：2026-03-06*
*作者：Semantic Router Team*
