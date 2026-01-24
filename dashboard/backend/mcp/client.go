package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"
)

// Client MCP 客户端
type Client struct {
	config    *ServerConfig
	transport Transport

	mu     sync.RWMutex
	status ServerStatus
	err    error
	tools  []ToolDefinition

	connectedAt *time.Time
}

// NewClient 创建 MCP 客户端
func NewClient(config *ServerConfig) (*Client, error) {
	var transport Transport

	switch config.Transport {
	case TransportStdio:
		transport = NewStdioTransport(&StdioConfig{
			Command: config.Connection.Command,
			Args:    config.Connection.Args,
			Env:     config.Connection.Env,
			Cwd:     config.Connection.Cwd,
		})

	case TransportStreamableHTTP:
		var timeout time.Duration
		if config.Options != nil && config.Options.Timeout > 0 {
			timeout = time.Duration(config.Options.Timeout) * time.Millisecond
		}

		var oauth *OAuthConfig
		if config.Security != nil {
			oauth = config.Security.OAuth
		}

		transport = NewStreamableHTTPTransport(&StreamableHTTPConfig{
			URL:     config.Connection.URL,
			Headers: config.Connection.Headers,
			Timeout: timeout,
			OAuth:   oauth,
		})

	default:
		return nil, fmt.Errorf("unsupported transport type: %s", config.Transport)
	}

	return &Client{
		config:    config,
		transport: transport,
		status:    StatusDisconnected,
	}, nil
}

// Connect 建立连接
func (c *Client) Connect(ctx context.Context) error {
	c.mu.Lock()
	c.status = StatusConnecting
	c.mu.Unlock()

	// 连接传输层
	if err := c.transport.Connect(ctx); err != nil {
		c.mu.Lock()
		c.status = StatusError
		c.err = err
		c.mu.Unlock()
		return err
	}

	// 获取工具列表
	tools, err := c.ListTools(ctx)
	if err != nil {
		// 工具列表获取失败不影响连接状态
		fmt.Printf("Warning: failed to list tools: %v\n", err)
	}

	now := time.Now()
	c.mu.Lock()
	c.status = StatusConnected
	c.err = nil
	c.tools = tools
	c.connectedAt = &now
	c.mu.Unlock()

	return nil
}

// Disconnect 断开连接
func (c *Client) Disconnect() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if err := c.transport.Disconnect(); err != nil {
		return err
	}

	c.status = StatusDisconnected
	c.tools = nil
	c.connectedAt = nil

	return nil
}

// ListTools 获取工具列表
func (c *Client) ListTools(ctx context.Context) ([]ToolDefinition, error) {
	result, err := c.transport.Call(ctx, "tools/list", nil)
	if err != nil {
		return nil, err
	}

	// 解析结果
	resultMap, ok := result.(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("unexpected result type: %T", result)
	}

	toolsRaw, ok := resultMap["tools"]
	if !ok {
		return nil, fmt.Errorf("tools field not found in response")
	}

	// 转换为 JSON 再解析
	toolsBytes, err := json.Marshal(toolsRaw)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal tools: %w", err)
	}

	var tools []ToolDefinition
	if err := json.Unmarshal(toolsBytes, &tools); err != nil {
		return nil, fmt.Errorf("failed to unmarshal tools: %w", err)
	}

	return tools, nil
}

// CallTool 调用工具
func (c *Client) CallTool(ctx context.Context, name string, arguments json.RawMessage) (*CallToolResult, error) {
	params := CallToolParams{
		Name:      name,
		Arguments: arguments,
	}

	result, err := c.transport.Call(ctx, "tools/call", params)
	if err != nil {
		return nil, err
	}

	// 转换结果
	resultBytes, err := json.Marshal(result)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal result: %w", err)
	}

	var callResult CallToolResult
	if err := json.Unmarshal(resultBytes, &callResult); err != nil {
		return nil, fmt.Errorf("failed to unmarshal result: %w", err)
	}

	return &callResult, nil
}

// CallToolStreaming 流式调用工具
func (c *Client) CallToolStreaming(ctx context.Context, name string, arguments json.RawMessage, onChunk func(StreamChunk) error) error {
	params := CallToolParams{
		Name:      name,
		Arguments: arguments,
	}

	return c.transport.CallStreaming(ctx, "tools/call", params, onChunk)
}

// GetStatus 获取状态
func (c *Client) GetStatus() ServerStatus {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.status
}

// GetState 获取完整状态
func (c *Client) GetState() *ServerState {
	c.mu.RLock()
	defer c.mu.RUnlock()

	errMsg := ""
	if c.err != nil {
		errMsg = c.err.Error()
	}

	return &ServerState{
		Config:      c.config,
		Status:      c.status,
		Error:       errMsg,
		Tools:       c.tools,
		ConnectedAt: c.connectedAt,
	}
}

// GetTools 获取缓存的工具列表
func (c *Client) GetTools() []ToolDefinition {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.tools
}

// GetConfig 获取配置
func (c *Client) GetConfig() *ServerConfig {
	return c.config
}
