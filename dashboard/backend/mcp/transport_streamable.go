package mcp

import (
	"bufio"
	"bytes"
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// StreamableHTTPTransport 官方规范的 Streamable HTTP 传输
type StreamableHTTPTransport struct {
	config *StreamableHTTPConfig
	client *http.Client

	mu        sync.RWMutex
	connected bool
	sessionID string

	// OAuth 2.1 token
	accessToken string
	tokenExpiry time.Time

	// PKCE
	codeVerifier string

	// 请求 ID 计数器
	requestID atomic.Int64
}

// StreamableHTTPConfig Streamable HTTP 传输配置
type StreamableHTTPConfig struct {
	// 单一端点 URL
	URL     string
	Headers map[string]string
	Timeout time.Duration

	// OAuth 2.1 配置
	OAuth *OAuthConfig
}

// NewStreamableHTTPTransport 创建 Streamable HTTP 传输
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
	result, err := t.Call(ctx, "initialize", InitializeParams{
		ProtocolVersion: "2025-06-18", // 使用最新协议版本
		Capabilities: map[string]interface{}{
			"tools": map[string]interface{}{
				"listChanged": true, // 支持工具列表变更通知
			},
		},
		ClientInfo: ClientInfo{
			Name:    "dashboard-mcp-client",
			Version: "1.0.0",
		},
	})
	if err != nil {
		return fmt.Errorf("initialize failed: %w", err)
	}

	// 解析响应
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
	id := t.requestID.Add(1)

	reqBody := JSONRPCRequest{
		JSONRPC: "2.0",
		ID:      id,
		Method:  method,
		Params:  params,
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
	req.Header.Set("Accept", "application/json, text/event-stream") // 同时接受两种响应

	// Session ID (如果存在)
	if t.sessionID != "" {
		req.Header.Set("Mcp-Session-Id", t.sessionID)
	}

	// OAuth Token (如果配置)
	t.mu.RLock()
	if t.accessToken != "" {
		req.Header.Set("Authorization", "Bearer "+t.accessToken)
	}
	t.mu.RUnlock()

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

	// 检查 HTTP 状态码
	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusAccepted {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("HTTP error %d: %s", resp.StatusCode, string(body))
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
	id := t.requestID.Add(1)

	reqBody := JSONRPCRequest{
		JSONRPC: "2.0",
		ID:      id,
		Method:  method,
		Params:  params,
	}

	bodyBytes, _ := json.Marshal(reqBody)
	req, _ := http.NewRequestWithContext(ctx, "POST", t.config.URL, bytes.NewReader(bodyBytes))

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream") // 明确请求流式响应

	if t.sessionID != "" {
		req.Header.Set("Mcp-Session-Id", t.sessionID)
	}

	t.mu.RLock()
	if t.accessToken != "" {
		req.Header.Set("Authorization", "Bearer "+t.accessToken)
	}
	t.mu.RUnlock()

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

// parseJSONResponse 解析 JSON 响应
func (t *StreamableHTTPTransport) parseJSONResponse(body io.Reader) (interface{}, error) {
	var resp JSONRPCResponse
	if err := json.NewDecoder(body).Decode(&resp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	if resp.Error != nil {
		return nil, fmt.Errorf("RPC error %d: %s", resp.Error.Code, resp.Error.Message)
	}

	var result interface{}
	if len(resp.Result) > 0 {
		if err := json.Unmarshal(resp.Result, &result); err != nil {
			return nil, fmt.Errorf("failed to unmarshal result: %w", err)
		}
	}

	return result, nil
}

// collectStreamResponse 收集流式响应返回最终结果
func (t *StreamableHTTPTransport) collectStreamResponse(body io.Reader) (interface{}, error) {
	var finalResult interface{}

	err := t.parseSSEStream(body, func(chunk StreamChunk) error {
		if chunk.Type == "complete" || chunk.Type == "message" {
			finalResult = chunk.Data
		}
		return nil
	})

	if err != nil {
		return nil, err
	}

	return finalResult, nil
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

			// 尝试解析为 JSON-RPC 响应
			var resp JSONRPCResponse
			if err := json.Unmarshal([]byte(eventData.String()), &resp); err == nil {
				if resp.Error != nil {
					chunk.Type = "error"
					chunk.Data = resp.Error.Message
				} else if len(resp.Result) > 0 {
					var result interface{}
					json.Unmarshal(resp.Result, &result)
					chunk.Data = result
					if chunk.Type == "" {
						chunk.Type = "complete"
					}
				}
			} else {
				// 非 JSON-RPC 格式，直接使用原始数据
				chunk.Data = eventData.String()
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

	t.mu.RLock()
	// 检查 token 是否仍有效
	if t.accessToken != "" && time.Now().Before(t.tokenExpiry) {
		t.mu.RUnlock()
		return nil
	}
	t.mu.RUnlock()

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

	// PKCE 支持
	if t.config.OAuth.UsePKCE {
		// 生成 code verifier
		verifier := make([]byte, 32)
		rand.Read(verifier)
		t.codeVerifier = base64.RawURLEncoding.EncodeToString(verifier)

		// 生成 code challenge
		hash := sha256.Sum256([]byte(t.codeVerifier))
		codeChallenge := base64.RawURLEncoding.EncodeToString(hash[:])
		data.Set("code_challenge", codeChallenge)
		data.Set("code_challenge_method", "S256")
	}

	req, err := http.NewRequestWithContext(ctx, "POST", t.config.OAuth.TokenURL,
		strings.NewReader(data.Encode()))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")

	resp, err := t.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("OAuth token request failed: %s", string(body))
	}

	var tokenResp struct {
		AccessToken string `json:"access_token"`
		ExpiresIn   int    `json:"expires_in"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&tokenResp); err != nil {
		return err
	}

	t.mu.Lock()
	t.accessToken = tokenResp.AccessToken
	t.tokenExpiry = time.Now().Add(time.Duration(tokenResp.ExpiresIn-60) * time.Second)
	t.mu.Unlock()

	return nil
}

// Disconnect 断开连接
func (t *StreamableHTTPTransport) Disconnect() error {
	t.mu.Lock()
	t.connected = false
	t.sessionID = ""
	t.accessToken = ""
	t.mu.Unlock()
	return nil
}

// IsConnected 检查连接状态
func (t *StreamableHTTPTransport) IsConnected() bool {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.connected
}
