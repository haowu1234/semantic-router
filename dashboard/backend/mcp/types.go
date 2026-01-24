// Package mcp provides MCP (Model Context Protocol) client implementation
// following the official MCP 2025-06-18 specification.
package mcp

import (
	"encoding/json"
	"time"
)

// TransportType 传输协议类型 (官方规范 2025-06-18)
type TransportType string

const (
	// TransportStdio Stdio 传输 - 本地命令行
	TransportStdio TransportType = "stdio"
	// TransportStreamableHTTP Streamable HTTP 传输 - 官方推荐
	TransportStreamableHTTP TransportType = "streamable-http"
)

// ServerStatus 服务器连接状态
type ServerStatus string

const (
	StatusDisconnected ServerStatus = "disconnected"
	StatusConnecting   ServerStatus = "connecting"
	StatusConnected    ServerStatus = "connected"
	StatusError        ServerStatus = "error"
)

// ServerConfig MCP 服务器配置
type ServerConfig struct {
	// 唯一标识符 (UUID)
	ID string `json:"id" yaml:"id"`

	// 显示名称
	Name string `json:"name" yaml:"name"`

	// 服务器描述
	Description string `json:"description,omitempty" yaml:"description,omitempty"`

	// 传输协议类型
	Transport TransportType `json:"transport" yaml:"transport"`

	// 连接配置
	Connection ConnectionConfig `json:"connection" yaml:"connection"`

	// 是否启用
	Enabled bool `json:"enabled" yaml:"enabled"`

	// 安全配置 (OAuth 2.1)
	Security *SecurityConfig `json:"security,omitempty" yaml:"security,omitempty"`

	// 高级选项
	Options *ServerOptions `json:"options,omitempty" yaml:"options,omitempty"`
}

// ConnectionConfig 连接配置
type ConnectionConfig struct {
	// === Stdio 传输配置 ===
	// 可执行命令
	Command string `json:"command,omitempty" yaml:"command,omitempty"`
	// 命令参数
	Args []string `json:"args,omitempty" yaml:"args,omitempty"`
	// 环境变量
	Env map[string]string `json:"env,omitempty" yaml:"env,omitempty"`
	// 工作目录
	Cwd string `json:"cwd,omitempty" yaml:"cwd,omitempty"`

	// === Streamable HTTP 传输配置 ===
	// 服务器 URL (单一端点)
	URL string `json:"url,omitempty" yaml:"url,omitempty"`
	// 自定义请求头
	Headers map[string]string `json:"headers,omitempty" yaml:"headers,omitempty"`
}

// SecurityConfig 安全配置 (官方 2025-06-18 规范)
type SecurityConfig struct {
	// OAuth 2.1 认证配置
	OAuth *OAuthConfig `json:"oauth,omitempty" yaml:"oauth,omitempty"`

	// 允许的 Origin (防止 DNS 重绑定攻击)
	AllowedOrigins []string `json:"allowed_origins,omitempty" yaml:"allowed_origins,omitempty"`

	// 是否仅限本地访问
	LocalOnly bool `json:"local_only,omitempty" yaml:"local_only,omitempty"`
}

// OAuthConfig OAuth 2.1 认证配置
type OAuthConfig struct {
	// OAuth 客户端 ID
	ClientID string `json:"client_id" yaml:"client_id"`
	// OAuth 客户端密钥 (仅机密客户端)
	ClientSecret string `json:"client_secret,omitempty" yaml:"client_secret,omitempty"`
	// 授权端点
	AuthorizationURL string `json:"authorization_url" yaml:"authorization_url"`
	// Token 端点
	TokenURL string `json:"token_url" yaml:"token_url"`
	// 请求的权限范围
	Scopes []string `json:"scopes,omitempty" yaml:"scopes,omitempty"`
	// 是否使用 PKCE (公共客户端强制要求)
	UsePKCE bool `json:"use_pkce,omitempty" yaml:"use_pkce,omitempty"`
}

// ServerOptions 高级选项
type ServerOptions struct {
	// 自动重连
	AutoReconnect bool `json:"auto_reconnect,omitempty" yaml:"auto_reconnect,omitempty"`
	// 重连间隔 (ms)
	ReconnectInterval int `json:"reconnect_interval,omitempty" yaml:"reconnect_interval,omitempty"`
	// 请求超时 (ms)
	Timeout int `json:"timeout,omitempty" yaml:"timeout,omitempty"`
	// 最大重试次数
	MaxRetries int `json:"max_retries,omitempty" yaml:"max_retries,omitempty"`
}

// ToolDefinition MCP 工具定义 (官方 2025-06-18 规范)
type ToolDefinition struct {
	// 工具名称
	Name string `json:"name"`
	// 工具描述
	Description string `json:"description,omitempty"`
	// 输入参数 Schema (JSON Schema)
	InputSchema json.RawMessage `json:"inputSchema"`
	// 输出结果 Schema (JSON Schema) - 官方新增
	OutputSchema json.RawMessage `json:"outputSchema,omitempty"`
}

// Tool 完整的工具信息 (包含来源服务器)
type Tool struct {
	ToolDefinition
	// 所属 MCP 服务器 ID
	ServerID string `json:"serverId"`
	// 所属 MCP 服务器名称
	ServerName string `json:"serverName"`
}

// ToolExecuteRequest 工具执行请求
type ToolExecuteRequest struct {
	ServerID  string          `json:"server_id"`
	ToolName  string          `json:"tool_name"`
	Arguments json.RawMessage `json:"arguments"`
}

// ToolResult 工具执行结果
type ToolResult struct {
	// 是否为流式响应
	IsStreaming bool `json:"is_streaming"`
	// 执行是否成功
	Success bool `json:"success"`
	// 执行结果
	Result interface{} `json:"result,omitempty"`
	// 结构化内容 (如果工具定义了 outputSchema)
	StructuredContent interface{} `json:"structured_content,omitempty"`
	// 错误信息
	Error string `json:"error,omitempty"`
	// 执行耗时 (ms)
	ExecutionTimeMs int64 `json:"execution_time_ms,omitempty"`
}

// StreamChunk 流式响应数据块
type StreamChunk struct {
	// 块类型: progress | partial | complete | error
	Type string `json:"type"`
	// 数据内容
	Data interface{} `json:"data"`
	// 进度 (0-100)
	Progress int `json:"progress,omitempty"`
}

// ServerState 服务器运行时状态
type ServerState struct {
	Config      *ServerConfig    `json:"config"`
	Status      ServerStatus     `json:"status"`
	Error       string           `json:"error,omitempty"`
	Tools       []ToolDefinition `json:"tools,omitempty"`
	ConnectedAt *time.Time       `json:"connected_at,omitempty"`
}

// ========== JSON-RPC Types ==========

// JSONRPCRequest JSON-RPC 2.0 请求
type JSONRPCRequest struct {
	JSONRPC string      `json:"jsonrpc"`
	ID      interface{} `json:"id"`
	Method  string      `json:"method"`
	Params  interface{} `json:"params,omitempty"`
}

// JSONRPCResponse JSON-RPC 2.0 响应
type JSONRPCResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      interface{}     `json:"id"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *JSONRPCError   `json:"error,omitempty"`
}

// JSONRPCError JSON-RPC 错误
type JSONRPCError struct {
	Code    int         `json:"code"`
	Message string      `json:"message"`
	Data    interface{} `json:"data,omitempty"`
}

// ========== MCP Protocol Types ==========

// InitializeParams initialize 请求参数
type InitializeParams struct {
	ProtocolVersion string                 `json:"protocolVersion"`
	Capabilities    map[string]interface{} `json:"capabilities"`
	ClientInfo      ClientInfo             `json:"clientInfo"`
}

// ClientInfo 客户端信息
type ClientInfo struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

// InitializeResult initialize 响应结果
type InitializeResult struct {
	ProtocolVersion string                 `json:"protocolVersion"`
	Capabilities    map[string]interface{} `json:"capabilities"`
	ServerInfo      ServerInfo             `json:"serverInfo"`
}

// ServerInfo 服务器信息
type ServerInfo struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

// ListToolsResult tools/list 响应结果
type ListToolsResult struct {
	Tools []ToolDefinition `json:"tools"`
}

// CallToolParams tools/call 请求参数
type CallToolParams struct {
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments,omitempty"`
}

// CallToolResult tools/call 响应结果
type CallToolResult struct {
	Content []ContentItem `json:"content"`
	IsError bool          `json:"isError,omitempty"`
}

// ContentItem 内容项
type ContentItem struct {
	Type string `json:"type"` // "text" | "image" | "resource"
	Text string `json:"text,omitempty"`
	// 其他类型的字段...
}

// ========== Elicitation Types (官方新增) ==========

// ElicitationRequest 服务器请求用户输入
type ElicitationRequest struct {
	ID                  string          `json:"id"`
	Message             string          `json:"message"`
	Schema              json.RawMessage `json:"schema"`
	RequestedPermission string          `json:"requested_permission,omitempty"`
	ServerID            string          `json:"server_id"`
}

// ElicitationResponse 用户响应
type ElicitationResponse struct {
	RequestID string      `json:"request_id"`
	Action    string      `json:"action"` // "approve" | "deny"
	Data      interface{} `json:"data,omitempty"`
}

// ========== Config File Types ==========

// ServersConfigFile MCP 服务器配置文件结构
type ServersConfigFile struct {
	Version         string         `yaml:"version"`
	ProtocolVersion string         `yaml:"protocol_version"`
	Servers         []ServerConfig `yaml:"servers"`
}
