package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
	"sync"
	"time"

	"github.com/mark3labs/mcp-go/client"
	"github.com/mark3labs/mcp-go/client/transport"
	"github.com/mark3labs/mcp-go/mcp"
)

// Client MCP 客户端 (基于官方 SDK)
type Client struct {
	config *ServerConfig

	mu     sync.RWMutex
	status ServerStatus
	err    error
	tools  []ToolDefinition

	connectedAt *time.Time

	// SDK 客户端
	mcpClient client.MCPClient
}

// NewClient 创建 MCP 客户端
func NewClient(config *ServerConfig) (*Client, error) {
	return &Client{
		config: config,
		status: StatusDisconnected,
	}, nil
}

// Connect 建立连接
func (c *Client) Connect(ctx context.Context) error {
	log.Printf("[MCP-Client] Connect() called for server: %s (transport: %s)", c.config.Name, c.config.Transport)

	c.mu.Lock()
	c.status = StatusConnecting
	c.mu.Unlock()

	var mcpClient client.MCPClient
	var err error

	switch c.config.Transport {
	case TransportStdio:
		mcpClient, err = c.createStdioClient(ctx)
	case TransportStreamableHTTP:
		mcpClient, err = c.createStreamableHTTPClient(ctx)
	default:
		return fmt.Errorf("unsupported transport type: %s", c.config.Transport)
	}

	if err != nil {
		log.Printf("[MCP-Client] Failed to create client: %v", err)
		c.mu.Lock()
		c.status = StatusError
		c.err = err
		c.mu.Unlock()
		return err
	}

	c.mu.Lock()
	c.mcpClient = mcpClient
	c.mu.Unlock()

	// 初始化连接
	log.Printf("[MCP-Client] Initializing connection...")
	initReq := mcp.InitializeRequest{}
	initReq.Params.ProtocolVersion = mcp.LATEST_PROTOCOL_VERSION
	initReq.Params.ClientInfo = mcp.Implementation{
		Name:    "semantic-router-mcp-client",
		Version: "1.0.0",
	}

	_, err = mcpClient.Initialize(ctx, initReq)
	if err != nil {
		log.Printf("[MCP-Client] Initialize failed: %v", err)
		mcpClient.Close()
		c.mu.Lock()
		c.status = StatusError
		c.err = err
		c.mcpClient = nil
		c.mu.Unlock()
		return fmt.Errorf("initialize failed: %w", err)
	}
	log.Printf("[MCP-Client] Initialization complete")

	// 获取工具列表
	log.Printf("[MCP-Client] Listing tools...")
	tools, err := c.ListTools(ctx)
	if err != nil {
		log.Printf("[MCP-Client] Warning: failed to list tools: %v", err)
	} else {
		log.Printf("[MCP-Client] Got %d tools", len(tools))
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

// createStdioClient 创建 Stdio 客户端
func (c *Client) createStdioClient(ctx context.Context) (client.MCPClient, error) {
	log.Printf("[MCP-Client] Creating Stdio client: command=%s, args=%v", c.config.Connection.Command, c.config.Connection.Args)

	// 构建环境变量
	env := os.Environ()
	for k, v := range c.config.Connection.Env {
		env = append(env, fmt.Sprintf("%s=%s", k, v))
	}

	// 准备选项
	opts := []transport.StdioOption{}

	// 如果需要设置工作目录，使用自定义命令函数
	if c.config.Connection.Cwd != "" {
		opts = append(opts, transport.WithCommandFunc(func(ctx context.Context, command string, env []string, args []string) (*exec.Cmd, error) {
			cmd := exec.CommandContext(ctx, command, args...)
			cmd.Env = env
			cmd.Dir = c.config.Connection.Cwd
			return cmd, nil
		}))
	}

	// 使用 SDK 创建 Stdio 客户端
	// NewStdioMCPClient 会自动启动子进程
	mcpClient, err := client.NewStdioMCPClientWithOptions(
		c.config.Connection.Command,
		env,
		c.config.Connection.Args,
		opts...,
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create stdio client: %w", err)
	}

	return mcpClient, nil
}

// createStreamableHTTPClient 创建 Streamable HTTP 客户端
func (c *Client) createStreamableHTTPClient(ctx context.Context) (client.MCPClient, error) {
	log.Printf("[MCP-Client] Creating Streamable HTTP client: url=%s", c.config.Connection.URL)

	opts := []transport.StreamableHTTPCOption{}

	// 设置超时
	timeout := 30 * time.Second
	if c.config.Options != nil && c.config.Options.Timeout > 0 {
		timeout = time.Duration(c.config.Options.Timeout) * time.Millisecond
	}
	opts = append(opts, transport.WithHTTPTimeout(timeout))

	// 设置自定义 Headers
	if len(c.config.Connection.Headers) > 0 {
		opts = append(opts, transport.WithHTTPHeaders(c.config.Connection.Headers))
	}

	// 如果需要自定义 HTTP Client（如添加 OAuth Token）
	if c.config.Security != nil && c.config.Security.OAuth != nil {
		customClient := &http.Client{
			Transport: &oauthTransport{
				base:  http.DefaultTransport,
				oauth: c.config.Security.OAuth,
			},
			Timeout: timeout,
		}
		opts = append(opts, transport.WithHTTPBasicClient(customClient))
	}

	mcpClient, err := client.NewStreamableHttpClient(c.config.Connection.URL, opts...)
	if err != nil {
		return nil, fmt.Errorf("failed to create streamable http client: %w", err)
	}

	return mcpClient, nil
}

// oauthTransport 自定义 HTTP Transport，用于添加 OAuth Token
type oauthTransport struct {
	base  http.RoundTripper
	oauth *OAuthConfig

	// TODO: 实现 token 缓存和刷新
	mu          sync.RWMutex
	accessToken string
	tokenExpiry time.Time
}

func (t *oauthTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	// TODO: 实现 OAuth 2.1 token 获取和刷新逻辑
	// 这里只是占位符，实际需要实现 client_credentials 流程
	t.mu.RLock()
	token := t.accessToken
	t.mu.RUnlock()

	if token != "" {
		req.Header.Set("Authorization", "Bearer "+token)
	}

	return t.base.RoundTrip(req)
}

// Disconnect 断开连接
func (c *Client) Disconnect() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.mcpClient != nil {
		if err := c.mcpClient.Close(); err != nil {
			log.Printf("[MCP-Client] Error closing client: %v", err)
		}
		c.mcpClient = nil
	}

	c.status = StatusDisconnected
	c.tools = nil
	c.connectedAt = nil

	return nil
}

// ListTools 获取工具列表
func (c *Client) ListTools(ctx context.Context) ([]ToolDefinition, error) {
	c.mu.RLock()
	mcpClient := c.mcpClient
	c.mu.RUnlock()

	if mcpClient == nil {
		return nil, fmt.Errorf("not connected")
	}

	log.Printf("[MCP-Client] Calling tools/list...")
	result, err := mcpClient.ListTools(ctx, mcp.ListToolsRequest{})
	if err != nil {
		log.Printf("[MCP-Client] tools/list failed: %v", err)
		return nil, err
	}

	// 转换为我们的类型
	tools := make([]ToolDefinition, 0, len(result.Tools))
	for _, t := range result.Tools {
		inputSchema, _ := json.Marshal(t.InputSchema)

		// 打印工具详细信息，包括完整的 InputSchema
		log.Printf("[MCP-Client] Tool discovered: name=%s", t.Name)
		log.Printf("[MCP-Client]   description: %s", t.Description)
		log.Printf("[MCP-Client]   inputSchema: %s", string(inputSchema))

		// 从序列化后的 JSON 解析，打印 required 字段和 properties
		var schemaMap map[string]interface{}
		if err := json.Unmarshal(inputSchema, &schemaMap); err == nil {
			if required, ok := schemaMap["required"]; ok {
				log.Printf("[MCP-Client]   required params: %v", required)
			}
			if properties, ok := schemaMap["properties"]; ok {
				if propsMap, ok := properties.(map[string]interface{}); ok {
					log.Printf("[MCP-Client]   properties count: %d", len(propsMap))
					for propName, propSchema := range propsMap {
						propJSON, _ := json.Marshal(propSchema)
						log.Printf("[MCP-Client]     - %s: %s", propName, string(propJSON))
					}
				}
			}
		}

		tools = append(tools, ToolDefinition{
			Name:        t.Name,
			Description: t.Description,
			InputSchema: inputSchema,
		})
	}

	log.Printf("[MCP-Client] Got %d tools", len(tools))
	return tools, nil
}

// CallTool 调用工具
func (c *Client) CallTool(ctx context.Context, name string, arguments json.RawMessage) (*CallToolResult, error) {
	log.Printf("[MCP-Client] CallTool() called: tool=%s, server=%s", name, c.config.Name)
	log.Printf("[MCP-Client] CallTool() arguments: %s", string(arguments))

	c.mu.RLock()
	mcpClient := c.mcpClient
	c.mu.RUnlock()

	if mcpClient == nil {
		log.Printf("[MCP-Client] CallTool() error: not connected")
		return nil, fmt.Errorf("not connected")
	}

	// 解析参数
	var args map[string]interface{}
	if len(arguments) > 0 {
		if err := json.Unmarshal(arguments, &args); err != nil {
			log.Printf("[MCP-Client] CallTool() error: failed to parse arguments: %v", err)
			return nil, fmt.Errorf("failed to parse arguments: %w", err)
		}
	}
	log.Printf("[MCP-Client] CallTool() parsed args: %+v", args)

	// 构建请求
	req := mcp.CallToolRequest{}
	req.Params.Name = name
	req.Params.Arguments = args

	log.Printf("[MCP-Client] CallTool() sending request to MCP server...")
	result, err := mcpClient.CallTool(ctx, req)
	if err != nil {
		log.Printf("[MCP-Client] CallTool() MCP server error: %v", err)
		return nil, err
	}
	log.Printf("[MCP-Client] CallTool() success, content items: %d, isError: %v", len(result.Content), result.IsError)

	// 转换结果
	content := make([]ContentItem, 0, len(result.Content))
	for _, item := range result.Content {
		contentItem := ContentItem{Type: "text"}
		switch v := item.(type) {
		case mcp.TextContent:
			contentItem.Text = v.Text
		case *mcp.TextContent:
			contentItem.Text = v.Text
		default:
			// 其他类型转为 JSON
			data, _ := json.Marshal(item)
			contentItem.Text = string(data)
		}
		content = append(content, contentItem)
	}

	return &CallToolResult{
		Content: content,
		IsError: result.IsError,
	}, nil
}

// CallToolStreaming 流式调用工具
// 注意：SDK 目前可能不完全支持流式，这里提供兼容实现
func (c *Client) CallToolStreaming(ctx context.Context, name string, arguments json.RawMessage, onChunk func(StreamChunk) error) error {
	// SDK 当前版本可能不支持真正的流式
	// 使用同步调用模拟
	result, err := c.CallTool(ctx, name, arguments)
	if err != nil {
		return onChunk(StreamChunk{Type: "error", Data: err.Error()})
	}

	// 发送完成事件
	var data interface{}
	if len(result.Content) > 0 && result.Content[0].Type == "text" {
		data = result.Content[0].Text
	} else {
		data = result.Content
	}

	return onChunk(StreamChunk{Type: "complete", Data: data, Progress: 100})
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

// ========== 兼容类型 ==========

// CallToolResult tools/call 响应结果
type CallToolResult struct {
	Content []ContentItem `json:"content"`
	IsError bool          `json:"isError,omitempty"`
}

// ContentItem 内容项
type ContentItem struct {
	Type string `json:"type"` // "text" | "image" | "resource"
	Text string `json:"text,omitempty"`
}
