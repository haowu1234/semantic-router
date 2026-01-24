# Dashboard MCP 支持设计方案（v3.0 - 官方规范对齐版）

## 📋 文档信息

| 项目 | 内容 |
|-----|------|
| 版本 | 3.0 |
| 状态 | 设计中 |
| 更新日期 | 2026-01-24 |
| 协议版本 | MCP 2025-06-18 |

---

## 1. 概述

### 1.1 背景

MCP（Model Context Protocol）是由 Anthropic 推出的开放协议，用于标准化 AI 模型与外部工具、资源之间的通信方式。本方案严格遵循 **MCP 2025-06-18** 官方规范。

### 1.2 官方规范关键变化（2025 更新）

| 变化项 | 旧版本 | 新版本 (2025) | 影响 |
|-------|-------|--------------|------|
| 传输协议 | Stdio + HTTP+SSE + WebSocket | **Stdio + Streamable HTTP** | 简化为 2 种官方传输 |
| HTTP 传输 | SSE 独立实现 | **Streamable HTTP 统一** | SSE 已废弃 |
| 认证方式 | Bearer Token | **OAuth 2.1 + PKCE** | 安全性增强 |
| 工具定义 | inputSchema | **inputSchema + outputSchema** | 结构化输出 |
| 新增能力 | - | **Elicitation** | 服务器请求用户输入 |

### 1.3 架构设计原则

```
┌──────────────────────────────────────────────────────────────────┐
│                      MCP 官方架构模型                            │
│                                                                  │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │                      Host (Dashboard)                     │  │
│   │                                                           │  │
│   │  ┌─────────────────────────────────────────────────────┐ │  │
│   │  │              MCP Client (多个)                       │ │  │
│   │  │                                                      │ │  │
│   │  │   每个 Client 维护与一个 Server 的 1:1 连接         │ │  │
│   │  └─────────────────────────────────────────────────────┘ │  │
│   └──────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │              MCP Server (外部进程/服务)                   │  │
│   │                                                           │  │
│   │   提供: Tools + Resources + Prompts                      │  │
│   └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 2. 传输协议设计（官方规范对齐）

### 2.1 官方支持的传输类型

| 协议 | 官方状态 | 适用场景 | 说明 |
|-----|---------|---------|------|
| **Stdio** | ✅ 官方支持 | 本地工具 | 子进程 stdin/stdout 通信 |
| **Streamable HTTP** | ✅ **官方推荐** | 远程服务 | 统一端点，支持立即/流式响应 |
| ~~SSE~~ | ⚠️ **已废弃** | - | 2025-03-26 起用 Streamable HTTP 替代 |
| ~~WebSocket~~ | ❌ **非官方** | - | 官方规范未定义 |

### 2.2 Streamable HTTP 传输（核心）

官方规范定义的单一 HTTP 端点模式：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Streamable HTTP 传输规范                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   单一端点: POST /mcp (或服务器自定义路径)                              │
│                                                                         │
│   ┌───────────────────────────────────────────────────────────────────┐ │
│   │  请求 Headers                                                     │ │
│   │  ├─ Content-Type: application/json                                │ │
│   │  ├─ Accept: application/json, text/event-stream                   │ │
│   │  ├─ Mcp-Session-Id: {session_id} (可选，用于会话保持)             │ │
│   │  └─ Authorization: Bearer {token} (OAuth 2.1)                     │ │
│   └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│   ┌───────────────────────────────────────────────────────────────────┐ │
│   │  响应模式（服务器根据请求决定）                                    │ │
│   │                                                                    │ │
│   │  模式 A: 立即响应                    模式 B: 流式响应              │ │
│   │  ┌────────────────────────┐         ┌────────────────────────┐   │ │
│   │  │ HTTP/1.1 200 OK        │         │ HTTP/1.1 200 OK        │   │ │
│   │  │ Content-Type:          │         │ Content-Type:          │   │ │
│   │  │   application/json     │         │   text/event-stream    │   │ │
│   │  │                        │         │                        │   │ │
│   │  │ {                      │         │ event: message         │   │ │
│   │  │   "jsonrpc": "2.0",    │         │ data: {"progress":30}  │   │ │
│   │  │   "id": 1,             │         │                        │   │ │
│   │  │   "result": {...}      │         │ event: message         │   │ │
│   │  │ }                      │         │ data: {"result":...}   │   │ │
│   │  └────────────────────────┘         └────────────────────────┘   │ │
│   └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│   ┌───────────────────────────────────────────────────────────────────┐ │
│   │  GET /mcp (可选 - 服务器主动推送)                                  │ │
│   │  ├─ 用于接收服务器主动发送的通知/请求                             │ │
│   │  ├─ 需要 Mcp-Session-Id header                                    │ │
│   │  └─ 响应: text/event-stream                                       │ │
│   └───────────────────────────────────────────────────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 系统架构

### 3.1 整体架构图

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          Dashboard Frontend (React)                           │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌─────────────────────┐  ┌────────────────────┐  ┌────────────────────────┐ │
│  │   ChatComponent     │  │   MCPConfigPanel   │  │     Tool Registry      │ │
│  │                     │  │                    │  │                        │ │
│  │  • 工具调用展示     │◄─│  • 服务器管理       │─►│  • Built-in Tools      │ │
│  │  • 流式进度         │  │  • 传输类型:        │  │  • MCP Tools (动态)    │ │
│  │  • Elicitation UI   │  │    ✓ Stdio         │  │                        │ │
│  │                     │  │    ✓ Streamable    │  │  统一调用接口          │ │
│  └─────────────────────┘  └────────┬───────────┘  └───────────┬────────────┘ │
│                                    │                          │              │
│                    ┌───────────────┴──────────────────────────┴───────┐     │
│                    │              useMCPServers Hook                   │     │
│                    │  • 状态管理  • OAuth 处理  • 工具同步             │     │
│                    └───────────────────────┬──────────────────────────┘     │
│                                            │                                 │
└────────────────────────────────────────────┼─────────────────────────────────┘
                                             │
                                             ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          Dashboard Backend (Go)                               │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌────────────────────────────┐    ┌──────────────────────────────────────┐  │
│  │    MCP Config Handler      │    │        MCP Tool Handler              │  │
│  │                            │    │                                      │  │
│  │  /api/mcp/servers          │    │  GET  /api/mcp/tools                 │  │
│  │  /api/mcp/servers/:id      │    │  POST /api/mcp/tools/execute         │  │
│  │  /api/mcp/oauth/callback   │    │  POST /api/mcp/tools/execute/stream  │  │
│  └─────────────┬──────────────┘    └──────────────┬───────────────────────┘  │
│                │                                   │                          │
│                ▼                                   ▼                          │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │                        MCP Client Manager                               │  │
│  │                                                                         │  │
│  │  ┌───────────────────────────────────────────────────────────────────┐ │  │
│  │  │                      Transport Interface                          │ │  │
│  │  │  • Connect(ctx)                    • Disconnect()                 │ │  │
│  │  │  • Call(ctx, method, params)       • IsConnected()                │ │  │
│  │  │  • CallStreaming(ctx, method, params, callback)                   │ │  │
│  │  └───────────────────────────────────────────────────────────────────┘ │  │
│  │                              │                                          │  │
│  │              ┌───────────────┴───────────────┐                         │  │
│  │              │                               │                         │  │
│  │              ▼                               ▼                         │  │
│  │  ┌─────────────────────────┐    ┌─────────────────────────────────┐   │  │
│  │  │    StdioTransport       │    │   StreamableHTTPTransport       │   │  │
│  │  │                         │    │                                 │   │  │
│  │  │  • 子进程管理            │    │  • 单一端点 POST               │   │  │
│  │  │  • stdin/stdout 通信    │    │  • 支持立即/流式响应            │   │  │
│  │  │  • 本地命令执行         │    │  • Session 管理                 │   │  │
│  │  │                         │    │  • OAuth 2.1 认证               │   │  │
│  │  └─────────────────────────┘    └─────────────────────────────────┘   │  │
│  │                                                                         │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘
                                        │
                        ┌───────────────┴───────────────┐
                        │                               │
                        ▼                               ▼
                ┌───────────────┐               ┌───────────────┐
                │  MCP Server   │               │  MCP Server   │
                │   (Stdio)     │               │ (Streamable)  │
                │               │               │               │
                │  filesystem   │               │  Remote API   │
                │  git, shell   │               │  AI Services  │
                └───────────────┘               └───────────────┘
```

---

## 4. 数据模型设计（官方规范对齐）

### 4.1 MCP 服务器配置

```typescript
// frontend/src/tools/mcp/types.ts

/**
 * MCP 传输类型 (官方规范 2025-06-18)
 */
export type MCPTransportType = 
  | 'stdio'           // 本地命令行 - 官方支持
  | 'streamable-http' // 流式 HTTP - 官方推荐

/**
 * MCP 服务器配置
 */
export interface MCPServerConfig {
  /** 唯一标识符 (UUID) */
  id: string
  
  /** 显示名称 */
  name: string
  
  /** 服务器描述 */
  description?: string
  
  /** 传输协议类型 (仅官方支持的 2 种) */
  transport: MCPTransportType
  
  /** 连接配置 */
  connection: MCPConnectionConfig
  
  /** 是否启用 */
  enabled: boolean
  
  /** 安全配置 (新增 - OAuth 2.1) */
  security?: MCPSecurityConfig
  
  /** 高级选项 */
  options?: MCPServerOptions
}

/**
 * 连接配置
 */
export interface MCPConnectionConfig {
  // ═══════════════════════════════════════════
  // Stdio 传输配置
  // ═══════════════════════════════════════════
  /** 可执行命令 */
  command?: string
  /** 命令参数 */
  args?: string[]
  /** 环境变量 */
  env?: Record<string, string>
  /** 工作目录 */
  cwd?: string
  
  // ═══════════════════════════════════════════
  // Streamable HTTP 传输配置
  // ═══════════════════════════════════════════
  /** 服务器 URL (单一端点) */
  url?: string
  /** 自定义请求头 */
  headers?: Record<string, string>
}

/**
 * 安全配置 (新增 - 官方 2025-06-18 规范)
 */
export interface MCPSecurityConfig {
  /** OAuth 2.1 认证配置 */
  oauth?: {
    /** OAuth 客户端 ID */
    clientId: string
    /** OAuth 客户端密钥 (仅机密客户端) */
    clientSecret?: string
    /** 授权端点 */
    authorizationUrl: string
    /** Token 端点 */
    tokenUrl: string
    /** 请求的权限范围 */
    scopes?: string[]
    /** 是否使用 PKCE (公共客户端强制要求) */
    usePKCE?: boolean
  }
  
  /** 允许的 Origin (防止 DNS 重绑定攻击) */
  allowedOrigins?: string[]
  
  /** 是否仅限本地访问 */
  localOnly?: boolean
}

/**
 * 高级选项
 */
export interface MCPServerOptions {
  /** 自动重连 */
  autoReconnect?: boolean
  /** 重连间隔 (ms) */
  reconnectInterval?: number
  /** 请求超时 (ms) */
  timeout?: number
  /** 最大重试次数 */
  maxRetries?: number
}
```

### 4.2 MCP 工具定义（新增 outputSchema）

```typescript
/**
 * MCP 工具定义 (官方 2025-06-18 规范)
 */
export interface MCPToolDefinition {
  /** 工具名称 */
  name: string
  
  /** 工具描述 */
  description?: string
  
  /** 输入参数 Schema (JSON Schema) */
  inputSchema: JSONSchema
  
  /** 🆕 输出结果 Schema (JSON Schema) - 官方新增 */
  outputSchema?: JSONSchema
  
  /** 所属 MCP 服务器 ID */
  serverId: string
  
  /** 所属 MCP 服务器名称 */
  serverName: string
}

/**
 * JSON Schema 定义
 */
export interface JSONSchema {
  type: 'object' | 'string' | 'number' | 'integer' | 'boolean' | 'array'
  properties?: Record<string, JSONSchemaProperty>
  required?: string[]
  description?: string
}

interface JSONSchemaProperty {
  type: 'string' | 'number' | 'integer' | 'boolean' | 'array' | 'object'
  description?: string
  enum?: unknown[]
  default?: unknown
  items?: JSONSchemaProperty
  properties?: Record<string, JSONSchemaProperty>
}
```

### 4.3 工具执行结果

```typescript
/**
 * 工具执行结果
 */
export interface MCPToolResult {
  /** 是否为流式响应 */
  isStreaming: boolean
  
  /** 执行是否成功 */
  success: boolean
  
  /** 执行结果 */
  result?: unknown
  
  /** 🆕 结构化内容 (如果工具定义了 outputSchema) */
  structuredContent?: unknown
  
  /** 错误信息 */
  error?: string
  
  /** 执行耗时 (ms) */
  executionTime?: number
}

/**
 * 流式响应数据块
 */
export interface MCPStreamChunk {
  /** 块类型 */
  type: 'progress' | 'partial' | 'complete' | 'error'
  
  /** 数据内容 */
  data: unknown
  
  /** 进度 (0-100) */
  progress?: number
  
  /** 时间戳 */
  timestamp?: number
}
```

### 4.4 🆕 Elicitation 请求（官方新增）

```typescript
/**
 * Elicitation 请求 (服务器请求用户输入)
 * 官方 2025-06-18 新增能力
 */
export interface ElicitationRequest {
  /** 请求 ID */
  id: string
  
  /** 提示消息 */
  message: string
  
  /** 期望的输入格式 (JSON Schema) */
  schema: JSONSchema
  
  /** 请求的权限 (可选) */
  requestedPermission?: string
  
  /** 来源服务器 */
  serverId: string
}

/**
 * Elicitation 响应
 */
export interface ElicitationResponse {
  /** 请求 ID */
  requestId: string
  
  /** 用户操作 */
  action: 'approve' | 'deny'
  
  /** 用户输入数据 */
  data?: unknown
}
```

---

## 5. 核心组件实现

### 5.1 后端：Streamable HTTP 传输（官方规范实现）

```go
// backend/mcp/transport_streamable_http.go

package mcp

import (
    "bufio"
    "bytes"
    "context"
    "encoding/json"
    "fmt"
    "io"
    "net/http"
    "strings"
    "sync"
    "time"
)

// StreamableHTTPTransport 官方规范的 Streamable HTTP 传输
type StreamableHTTPTransport struct {
    config    *StreamableHTTPConfig
    client    *http.Client
    
    mu        sync.RWMutex
    connected bool
    sessionID string
    
    // OAuth 2.1 token
    accessToken string
    tokenExpiry time.Time
}

type StreamableHTTPConfig struct {
    // 单一端点 URL
    URL     string
    Headers map[string]string
    Timeout time.Duration
    
    // OAuth 2.1 配置
    OAuth *OAuthConfig
}

type OAuthConfig struct {
    ClientID        string
    ClientSecret    string
    AuthorizationURL string
    TokenURL        string
    Scopes          []string
    UsePKCE         bool
}

func NewStreamableHTTPTransport(config *StreamableHTTPConfig) *StreamableHTTPTransport {
    timeout := config.Timeout
    if timeout == 0 {
        timeout = 30 * time.Second
    }
    
    return &StreamableHTTPTransport{
        config: config,
        client: &http.Client{Timeout: timeout},
    }
}

// Connect 初始化连接 (发送 initialize 请求)
func (t *StreamableHTTPTransport) Connect(ctx context.Context) error {
    // 如果配置了 OAuth，先获取 token
    if t.config.OAuth != nil {
        if err := t.refreshOAuthToken(ctx); err != nil {
            return fmt.Errorf("OAuth authentication failed: %w", err)
        }
    }
    
    // 发送 initialize 请求
    result, err := t.Call(ctx, "initialize", map[string]interface{}{
        "protocolVersion": "2025-06-18",  // 使用最新协议版本
        "capabilities": map[string]interface{}{
            "tools": map[string]interface{}{
                "listChanged": true,  // 支持工具列表变更通知
            },
        },
        "clientInfo": map[string]interface{}{
            "name":    "dashboard-mcp-client",
            "version": "1.0.0",
        },
    })
    if err != nil {
        return fmt.Errorf("initialize failed: %w", err)
    }
    
    // 解析响应，保存 session ID
    if resp, ok := result.(map[string]interface{}); ok {
        // 检查服务器返回的协议版本
        if version, ok := resp["protocolVersion"].(string); ok {
            fmt.Printf("Connected to MCP server with protocol version: %s\n", version)
        }
    }
    
    t.mu.Lock()
    t.connected = true
    t.mu.Unlock()
    
    return nil
}

// Call 执行 JSON-RPC 调用 (官方规范实现)
func (t *StreamableHTTPTransport) Call(ctx context.Context, method string, params interface{}) (interface{}, error) {
    reqBody := map[string]interface{}{
        "jsonrpc": "2.0",
        "id":      generateRequestID(),
        "method":  method,
        "params":  params,
    }
    
    bodyBytes, err := json.Marshal(reqBody)
    if err != nil {
        return nil, fmt.Errorf("failed to marshal request: %w", err)
    }
    
    req, err := http.NewRequestWithContext(ctx, "POST", t.config.URL, bytes.NewReader(bodyBytes))
    if err != nil {
        return nil, fmt.Errorf("failed to create request: %w", err)
    }
    
    // 设置官方规范要求的 Headers
    req.Header.Set("Content-Type", "application/json")
    req.Header.Set("Accept", "application/json, text/event-stream")  // 同时接受两种响应
    
    // Session ID (如果存在)
    if t.sessionID != "" {
        req.Header.Set("Mcp-Session-Id", t.sessionID)
    }
    
    // OAuth Token (如果配置)
    if t.accessToken != "" {
        req.Header.Set("Authorization", "Bearer "+t.accessToken)
    }
    
    // 自定义 Headers
    for k, v := range t.config.Headers {
        req.Header.Set(k, v)
    }
    
    resp, err := t.client.Do(req)
    if err != nil {
        return nil, fmt.Errorf("request failed: %w", err)
    }
    defer resp.Body.Close()
    
    // 保存服务器返回的 Session ID
    if sid := resp.Header.Get("Mcp-Session-Id"); sid != "" {
        t.sessionID = sid
    }
    
    // 根据 Content-Type 处理响应
    contentType := resp.Header.Get("Content-Type")
    if strings.Contains(contentType, "text/event-stream") {
        // 流式响应 - 收集所有事件返回最终结果
        return t.collectStreamResponse(resp.Body)
    }
    
    // 立即响应
    return t.parseJSONResponse(resp.Body)
}

// CallStreaming 执行流式调用
func (t *StreamableHTTPTransport) CallStreaming(
    ctx context.Context,
    method string,
    params interface{},
    onChunk func(chunk StreamChunk) error,
) error {
    reqBody := map[string]interface{}{
        "jsonrpc": "2.0",
        "id":      generateRequestID(),
        "method":  method,
        "params":  params,
    }
    
    bodyBytes, _ := json.Marshal(reqBody)
    req, _ := http.NewRequestWithContext(ctx, "POST", t.config.URL, bytes.NewReader(bodyBytes))
    
    req.Header.Set("Content-Type", "application/json")
    req.Header.Set("Accept", "text/event-stream")  // 明确请求流式响应
    
    if t.sessionID != "" {
        req.Header.Set("Mcp-Session-Id", t.sessionID)
    }
    if t.accessToken != "" {
        req.Header.Set("Authorization", "Bearer "+t.accessToken)
    }
    for k, v := range t.config.Headers {
        req.Header.Set(k, v)
    }
    
    resp, err := t.client.Do(req)
    if err != nil {
        return err
    }
    defer resp.Body.Close()
    
    // 解析 SSE 流
    return t.parseSSEStream(resp.Body, onChunk)
}

// parseSSEStream 解析 SSE 事件流
func (t *StreamableHTTPTransport) parseSSEStream(body io.Reader, onChunk func(StreamChunk) error) error {
    scanner := bufio.NewScanner(body)
    var eventType string
    var eventData strings.Builder
    
    for scanner.Scan() {
        line := scanner.Text()
        
        switch {
        case strings.HasPrefix(line, "event:"):
            eventType = strings.TrimSpace(line[6:])
        case strings.HasPrefix(line, "data:"):
            eventData.WriteString(strings.TrimSpace(line[5:]))
        case line == "" && eventData.Len() > 0:
            // 事件完成
            chunk := StreamChunk{Type: eventType}
            if err := json.Unmarshal([]byte(eventData.String()), &chunk.Data); err != nil {
                chunk.Data = eventData.String()  // 非 JSON 数据
            }
            
            if err := onChunk(chunk); err != nil {
                return err
            }
            
            eventType = ""
            eventData.Reset()
        }
    }
    
    return scanner.Err()
}

// refreshOAuthToken OAuth 2.1 token 刷新
func (t *StreamableHTTPTransport) refreshOAuthToken(ctx context.Context) error {
    if t.config.OAuth == nil {
        return nil
    }
    
    // 检查 token 是否仍有效
    if t.accessToken != "" && time.Now().Before(t.tokenExpiry) {
        return nil
    }
    
    // 构建 token 请求
    data := url.Values{}
    data.Set("grant_type", "client_credentials")
    data.Set("client_id", t.config.OAuth.ClientID)
    if t.config.OAuth.ClientSecret != "" {
        data.Set("client_secret", t.config.OAuth.ClientSecret)
    }
    if len(t.config.OAuth.Scopes) > 0 {
        data.Set("scope", strings.Join(t.config.OAuth.Scopes, " "))
    }
    
    req, _ := http.NewRequestWithContext(ctx, "POST", t.config.OAuth.TokenURL, 
        strings.NewReader(data.Encode()))
    req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
    
    resp, err := t.client.Do(req)
    if err != nil {
        return err
    }
    defer resp.Body.Close()
    
    var tokenResp struct {
        AccessToken string `json:"access_token"`
        ExpiresIn   int    `json:"expires_in"`
    }
    if err := json.NewDecoder(resp.Body).Decode(&tokenResp); err != nil {
        return err
    }
    
    t.accessToken = tokenResp.AccessToken
    t.tokenExpiry = time.Now().Add(time.Duration(tokenResp.ExpiresIn-60) * time.Second)
    
    return nil
}

func (t *StreamableHTTPTransport) Disconnect() error {
    t.mu.Lock()
    t.connected = false
    t.sessionID = ""
    t.mu.Unlock()
    return nil
}

func (t *StreamableHTTPTransport) IsConnected() bool {
    t.mu.RLock()
    defer t.mu.RUnlock()
    return t.connected
}
```

### 5.2 前端：流式工具执行器

```typescript
// frontend/src/tools/executors/mcpTool.ts

import type { ToolExecutionContext } from '../types'
import type { MCPToolResult, MCPStreamChunk } from '../mcp/types'

/**
 * 执行 MCP 工具 (支持普通和流式响应)
 */
export async function executeMCPTool(
  serverId: string,
  toolName: string,
  args: unknown,
  context: ToolExecutionContext
): Promise<MCPToolResult> {
  const response = await fetch('/api/mcp/tools/execute', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...context.headers,
    },
    body: JSON.stringify({
      server_id: serverId,
      tool_name: toolName,
      arguments: args,
    }),
    signal: context.signal,
  })
  
  if (!response.ok) {
    return {
      isStreaming: false,
      success: false,
      error: `MCP tool execution failed: ${response.statusText}`,
    }
  }
  
  const result = await response.json()
  return {
    isStreaming: false,
    success: true,
    result: result.result,
    structuredContent: result.structuredContent,
    executionTime: result.execution_time_ms,
  }
}

/**
 * 流式执行 MCP 工具 (AsyncGenerator)
 */
export async function* executeMCPToolStreaming(
  serverId: string,
  toolName: string,
  args: unknown,
  context: ToolExecutionContext
): AsyncGenerator<MCPStreamChunk, MCPToolResult, unknown> {
  const response = await fetch('/api/mcp/tools/execute/stream', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Accept': 'text/event-stream',
      ...context.headers,
    },
    body: JSON.stringify({
      server_id: serverId,
      tool_name: toolName,
      arguments: args,
    }),
    signal: context.signal,
  })
  
  if (!response.ok) {
    return {
      isStreaming: true,
      success: false,
      error: `Streaming execution failed: ${response.statusText}`,
    }
  }
  
  if (!response.body) {
    throw new Error('Response body is null')
  }
  
  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let finalResult: unknown = null
  
  try {
    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      
      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() || ''
      
      let eventType = ''
      let eventData = ''
      
      for (const line of lines) {
        if (line.startsWith('event:')) {
          eventType = line.slice(6).trim()
        } else if (line.startsWith('data:')) {
          eventData = line.slice(5).trim()
        } else if (line === '' && eventData) {
          try {
            const chunk: MCPStreamChunk = {
              type: eventType as MCPStreamChunk['type'],
              data: JSON.parse(eventData),
              timestamp: Date.now(),
            }
            
            if (chunk.type === 'complete') {
              finalResult = chunk.data
            }
            
            yield chunk
          } catch {
            yield { type: 'partial', data: eventData, timestamp: Date.now() }
          }
          eventType = ''
          eventData = ''
        }
      }
    }
  } finally {
    reader.releaseLock()
  }
  
  return {
    isStreaming: true,
    success: true,
    result: finalResult,
  }
}
```

### 5.3 🆕 Elicitation 处理 Hook

```typescript
// frontend/src/tools/mcp/useElicitation.ts

import { useState, useCallback } from 'react'
import type { ElicitationRequest, ElicitationResponse } from './types'

export interface UseElicitationReturn {
  /** 当前待处理的 Elicitation 请求 */
  pendingRequest: ElicitationRequest | null
  
  /** 处理用户响应 */
  respond: (response: ElicitationResponse) => Promise<void>
  
  /** 取消当前请求 */
  cancel: () => void
}

/**
 * Elicitation Hook - 处理 MCP 服务器请求用户输入
 */
export function useElicitation(): UseElicitationReturn {
  const [pendingRequest, setPendingRequest] = useState<ElicitationRequest | null>(null)
  
  const respond = useCallback(async (response: ElicitationResponse) => {
    if (!pendingRequest) return
    
    await fetch('/api/mcp/elicitation/respond', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(response),
    })
    
    setPendingRequest(null)
  }, [pendingRequest])
  
  const cancel = useCallback(() => {
    if (pendingRequest) {
      respond({
        requestId: pendingRequest.id,
        action: 'deny',
      })
    }
  }, [pendingRequest, respond])
  
  return {
    pendingRequest,
    respond,
    cancel,
  }
}
```

---

## 6. API 设计

### 6.1 完整 API 端点

| 方法 | 端点 | 描述 |
|-----|------|-----|
| **配置管理** | | |
| `GET` | `/api/mcp/servers` | 获取所有服务器配置 |
| `POST` | `/api/mcp/servers` | 创建服务器配置 |
| `PUT` | `/api/mcp/servers/:id` | 更新服务器配置 |
| `DELETE` | `/api/mcp/servers/:id` | 删除服务器配置 |
| **连接管理** | | |
| `POST` | `/api/mcp/servers/:id/connect` | 连接服务器 |
| `POST` | `/api/mcp/servers/:id/disconnect` | 断开连接 |
| `GET` | `/api/mcp/servers/:id/status` | 获取服务器状态 |
| `POST` | `/api/mcp/servers/:id/test` | 测试连接 |
| **OAuth** | | |
| `GET` | `/api/mcp/oauth/authorize/:id` | 获取授权 URL |
| `GET` | `/api/mcp/oauth/callback` | OAuth 回调 |
| **工具** | | |
| `GET` | `/api/mcp/tools` | 获取所有工具 |
| `POST` | `/api/mcp/tools/execute` | 执行工具 (普通) |
| `POST` | `/api/mcp/tools/execute/stream` | 执行工具 (流式) |
| **Elicitation** | | |
| `POST` | `/api/mcp/elicitation/respond` | 响应 Elicitation |

---

## 7. UI 设计

### 7.1 配置面板

```
┌─────────────────────────────────────────────────────────────┐
│ 🔌 MCP Servers                                          [×] │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ 📁 Filesystem MCP                              [Stdio]  │ │
│ │    ● Connected  •  5 tools                         [⚙]  │ │
│ │                                                         │ │
│ │    工具: read_file, write_file, list_dir...            │ │
│ ├─────────────────────────────────────────────────────────┤ │
│ │ 🌐 Analytics API                    [Streamable HTTP]   │ │
│ │    ● Connected (OAuth)  •  3 tools                 [⚙]  │ │
│ │                                                         │ │
│ │    工具: analyze, visualize, export                     │ │
│ ├─────────────────────────────────────────────────────────┤ │
│ │ 🐙 GitHub MCP                              [Stdio]      │ │
│ │    ○ Disabled                                      [⚙]  │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ [+ 添加 MCP 服务器]                                         │
│                                                             │
│ ─────────────────────────────────────────────────────────── │
│ ℹ️ 仅支持官方传输协议: Stdio 和 Streamable HTTP             │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 添加服务器对话框

```
┌─────────────────────────────────────────────────────────────┐
│ 🔌 添加 MCP 服务器                                      [×] │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ 名称 *                                                      │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ My MCP Server                                           │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ 传输协议 * (官方支持)                                        │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ ◉ Stdio         本地命令行，适用于 filesystem/git 等     │ │
│ │ ○ Streamable HTTP  远程服务，支持流式响应和 OAuth        │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ ─────────────── Stdio 配置 ───────────────────             │
│                                                             │
│ 命令 *                                                      │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ npx                                                     │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ 参数                                                        │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ -y                                                      │ │
│ │ @modelcontextprotocol/server-filesystem                 │ │
│ │ /Users/workspace                                        │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ ─────────────── 高级选项 ───────────────────               │
│                                                             │
│ ☑ 自动重连     超时: [30] 秒                                │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                    [取消]  [测试连接]  [保存]               │
└─────────────────────────────────────────────────────────────┘
```

### 7.3 Streamable HTTP + OAuth 配置

```
┌─────────────────────────────────────────────────────────────┐
│ ─────────────── Streamable HTTP 配置 ──────────────────    │
│                                                             │
│ URL *                                                       │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ https://api.example.com/mcp                             │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ ☑ 启用 OAuth 2.1 认证                                       │
│                                                             │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ Client ID *                                             │ │
│ │ ┌─────────────────────────────────────────────────────┐ │ │
│ │ │ my-client-id                                        │ │ │
│ │ └─────────────────────────────────────────────────────┘ │ │
│ │                                                         │ │
│ │ Authorization URL *                                     │ │
│ │ ┌─────────────────────────────────────────────────────┐ │ │
│ │ │ https://auth.example.com/authorize                  │ │ │
│ │ └─────────────────────────────────────────────────────┘ │ │
│ │                                                         │ │
│ │ Token URL *                                             │ │
│ │ ┌─────────────────────────────────────────────────────┐ │ │
│ │ │ https://auth.example.com/token                      │ │ │
│ │ └─────────────────────────────────────────────────────┘ │ │
│ │                                                         │ │
│ │ ☑ 使用 PKCE (推荐)                                      │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 7.4 🆕 Elicitation 对话框

```
┌─────────────────────────────────────────────────────────────┐
│ 📝 MCP 服务器请求输入                                   [×] │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ 来自: Analytics MCP                                         │
│                                                             │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ 请提供数据库连接信息以继续分析任务:                       │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ 数据库主机 *                                                │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ localhost                                               │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ 端口                                                        │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │ 5432                                                    │ │
│ └─────────────────────────────────────────────────────────┘ │
│                                                             │
│ ⚠️ 注意: 此信息将发送至 MCP 服务器                          │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                              [拒绝]  [确认并发送]           │
└─────────────────────────────────────────────────────────────┘
```

---

## 8. 配置文件格式

```yaml
# .vllm-sr/mcp-servers.yaml
version: "1.0"
protocol_version: "2025-06-18"  # 协议版本

servers:
  # Stdio 传输示例
  - id: "fs-mcp-001"
    name: "Filesystem MCP"
    description: "本地文件系统访问"
    transport: "stdio"  # 官方支持
    enabled: true
    connection:
      command: "npx"
      args:
        - "-y"
        - "@modelcontextprotocol/server-filesystem"
        - "/Users/workspace"
    options:
      autoReconnect: true
      timeout: 30000

  # Streamable HTTP + OAuth 示例
  - id: "analytics-mcp-002"
    name: "AI Analytics"
    description: "AI 驱动的数据分析服务"
    transport: "streamable-http"  # 官方推荐
    enabled: true
    connection:
      url: "https://analytics.example.com/mcp"
    security:
      oauth:
        clientId: "dashboard-client"
        authorizationUrl: "https://auth.example.com/authorize"
        tokenUrl: "https://auth.example.com/token"
        scopes:
          - "read"
          - "analyze"
        usePKCE: true
    options:
      timeout: 60000
```

---

## 9. 文件结构

```
dashboard/
├── backend/
│   ├── handlers/
│   │   ├── mcp_config.go          # 配置 CRUD
│   │   ├── mcp_tools.go           # 工具执行
│   │   ├── mcp_oauth.go           # 🆕 OAuth 处理
│   │   └── mcp_elicitation.go     # 🆕 Elicitation 处理
│   │
│   ├── mcp/
│   │   ├── manager.go             # 客户端管理器
│   │   ├── client.go              # MCP 客户端
│   │   ├── transport.go           # 传输接口
│   │   ├── transport_stdio.go     # Stdio 实现
│   │   ├── transport_streamable.go # Streamable HTTP 实现
│   │   ├── oauth.go               # 🆕 OAuth 2.1 客户端
│   │   └── types.go               # 类型定义
│   │
│   └── router/
│       └── router.go              # 路由配置
│
└── frontend/
    └── src/
        ├── tools/
        │   ├── mcp/
        │   │   ├── types.ts       # 类型定义
        │   │   ├── useMCPServers.ts
        │   │   ├── useElicitation.ts  # 🆕 Elicitation Hook
        │   │   ├── mcpToolBridge.ts
        │   │   └── index.ts
        │   │
        │   └── executors/
        │       └── mcpTool.ts     # MCP 执行器
        │
        └── components/
            ├── MCPConfigPanel.tsx
            ├── MCPServerDialog.tsx
            ├── ElicitationDialog.tsx  # 🆕 Elicitation UI
            └── ChatComponent.tsx
```

---

## 10. 实现路线图

| 阶段 | 内容 | 预估时间 |
|-----|------|---------|
| **Phase 1** | 基础架构 + 类型定义 | 2-3 天 |
| **Phase 2** | Stdio 传输实现 | 2-3 天 |
| **Phase 3** | Streamable HTTP 实现 | 3-4 天 |
| **Phase 4** | OAuth 2.1 集成 | 2-3 天 |
| **Phase 5** | 工具执行 + 流式支持 | 2-3 天 |
| **Phase 6** | UI 组件开发 | 3-4 天 |
| **Phase 7** | Elicitation 支持 | 1-2 天 |
| **Phase 8** | 测试 + 优化 | 2-3 天 |
| **总计** | | **17-25 天** |

---

## 11. 关键变更总结

相较于 v2.0 方案，本方案的核心更新：

| 项目 | v2.0 | v3.0 (官方对齐) |
|-----|------|----------------|
| 传输协议 | 4 种 | **2 种** (Stdio + Streamable HTTP) |
| HTTP 传输 | SSE + Streamable HTTP | **仅 Streamable HTTP** |
| 认证 | Bearer Token | **OAuth 2.1 + PKCE** |
| 工具输出 | inputSchema | **inputSchema + outputSchema** |
| 新能力 | - | **Elicitation** |
| 协议版本 | 2024-11-05 | **2025-06-18** |

---

## 12. 参考资料

- [MCP 官方规范](https://modelcontextprotocol.io/specification)
- [MCP TypeScript SDK](https://github.com/modelcontextprotocol/typescript-sdk)
- [MCP Go SDK](https://github.com/mark3labs/mcp-go)
- [OAuth 2.1 草案](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-v2-1-09)