package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
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
	log.Printf("[MCP-Client] Connect() called for server: %s (transport: %s)", c.config.Name, c.config.Transport)

	c.mu.Lock()
	c.status = StatusConnecting
	c.mu.Unlock()
	log.Printf("[MCP-Client] Status set to: connecting")

	// 连接传输层
	log.Printf("[MCP-Client] Connecting transport layer...")
	if err := c.transport.Connect(ctx); err != nil {
		log.Printf("[MCP-Client] Transport connect failed: %v", err)
		c.mu.Lock()
		c.status = StatusError
		c.err = err
		c.mu.Unlock()
		return err
	}
	log.Printf("[MCP-Client] Transport connected successfully")

	// 发送 MCP initialize 请求 (协议必需)
	log.Printf("[MCP-Client] Sending initialize request...")
	initResult, err := c.transport.Call(ctx, "initialize", InitializeParams{
		ProtocolVersion: "2024-11-05",
		Capabilities: map[string]interface{}{
			"tools": map[string]interface{}{
				"listChanged": true,
			},
		},
		ClientInfo: ClientInfo{
			Name:    "semantic-router-mcp-client",
			Version: "1.0.0",
		},
	})
	if err != nil {
		log.Printf("[MCP-Client] Initialize request failed: %v", err)
		c.mu.Lock()
		c.status = StatusError
		c.err = fmt.Errorf("initialize failed: %w", err)
		c.mu.Unlock()
		_ = c.transport.Disconnect()
		return c.err
	}
	log.Printf("[MCP-Client] Initialize request succeeded")

	// 打印服务器信息
	if resp, ok := initResult.(map[string]interface{}); ok {
		if version, ok := resp["protocolVersion"].(string); ok {
			fmt.Printf("MCP server protocol version: %s\n", version)
		}
		if serverInfo, ok := resp["serverInfo"].(map[string]interface{}); ok {
			if name, ok := serverInfo["name"].(string); ok {
				fmt.Printf("MCP server name: %s\n", name)
			}
		}
	}

	// 发送 initialized 通知 (告知服务器客户端已准备好)
	// 注意：这是一个通知，不是请求，没有返回值
	log.Printf("[MCP-Client] Sending initialized notification...")
	_, _ = c.transport.Call(ctx, "notifications/initialized", nil)
	log.Printf("[MCP-Client] Initialized notification sent")

	// 获取工具列表
	log.Printf("[MCP-Client] Calling ListTools()...")
	tools, err := c.ListTools(ctx)
	if err != nil {
		// 工具列表获取失败不影响连接状态
		log.Printf("[MCP-Client] Warning: failed to list tools: %v", err)
		fmt.Printf("Warning: failed to list tools: %v\n", err)
	} else {
		log.Printf("[MCP-Client] ListTools() succeeded, got %d tools", len(tools))
	}

	now := time.Now()
	c.mu.Lock()
	c.status = StatusConnected
	c.err = nil
	c.tools = tools
	c.connectedAt = &now
	c.mu.Unlock()

	log.Printf("[MCP-Client] Connect() completed, status: connected, tools: %d", len(tools))
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
	log.Printf("[MCP-Client] ListTools() calling tools/list...")
	result, err := c.transport.Call(ctx, "tools/list", nil)
	if err != nil {
		log.Printf("[MCP-Client] tools/list failed: %v", err)
		return nil, err
	}
	log.Printf("[MCP-Client] tools/list succeeded, parsing result...")

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
