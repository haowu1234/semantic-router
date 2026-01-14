package mcp

import (
	"context"
	"encoding/json"
)

// JSONRPCRequest represents a JSON-RPC 2.0 request
type JSONRPCRequest struct {
	JSONRPC string      `json:"jsonrpc"`
	ID      int64       `json:"id"`
	Method  string      `json:"method"`
	Params  interface{} `json:"params,omitempty"`
}

// JSONRPCResponse represents a JSON-RPC 2.0 response
type JSONRPCResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      int64           `json:"id"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *JSONRPCError   `json:"error,omitempty"`
}

// JSONRPCError represents a JSON-RPC 2.0 error
type JSONRPCError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
	Data    any    `json:"data,omitempty"`
}

// MCPToolDefinition represents a tool discovered from MCP
type MCPToolDefinition struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	InputSchema map[string]interface{} `json:"inputSchema"`
}

// MCPToolsListResult represents the result of tools/list
type MCPToolsListResult struct {
	Tools []MCPToolDefinition `json:"tools"`
}

// MCPToolCallResult represents the result of tools/call
type MCPToolCallResult struct {
	Content []MCPContent `json:"content"`
	IsError bool         `json:"isError,omitempty"`
}

// MCPContent represents content in MCP responses
type MCPContent struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
}

// Transport defines the interface for MCP transport
type Transport interface {
	Send(ctx context.Context, req *JSONRPCRequest) (*JSONRPCResponse, error)
	Close() error
}

// Client is an MCP protocol client
type Client struct {
	transport Transport
	nextID    int64
}

// NewClient creates a new MCP client
func NewClient(transport Transport) *Client {
	return &Client{
		transport: transport,
		nextID:    1,
	}
}

// Initialize initializes the MCP connection
func (c *Client) Initialize(ctx context.Context) error {
	req := &JSONRPCRequest{
		JSONRPC: "2.0",
		ID:      c.nextID,
		Method:  "initialize",
		Params: map[string]interface{}{
			"protocolVersion": "2024-11-05",
			"capabilities":    map[string]interface{}{},
			"clientInfo": map[string]interface{}{
				"name":    "vllm-sr-dashboard",
				"version": "1.0.0",
			},
		},
	}
	c.nextID++

	resp, err := c.transport.Send(ctx, req)
	if err != nil {
		return err
	}

	if resp.Error != nil {
		return &RPCError{Code: resp.Error.Code, Message: resp.Error.Message}
	}

	// Send initialized notification
	notif := &JSONRPCRequest{
		JSONRPC: "2.0",
		Method:  "notifications/initialized",
	}
	_, _ = c.transport.Send(ctx, notif)

	return nil
}

// ListTools lists available tools from the MCP server
func (c *Client) ListTools(ctx context.Context) ([]MCPToolDefinition, error) {
	req := &JSONRPCRequest{
		JSONRPC: "2.0",
		ID:      c.nextID,
		Method:  "tools/list",
	}
	c.nextID++

	resp, err := c.transport.Send(ctx, req)
	if err != nil {
		return nil, err
	}

	if resp.Error != nil {
		return nil, &RPCError{Code: resp.Error.Code, Message: resp.Error.Message}
	}

	var result MCPToolsListResult
	if err := json.Unmarshal(resp.Result, &result); err != nil {
		return nil, err
	}

	return result.Tools, nil
}

// CallTool calls a tool on the MCP server
func (c *Client) CallTool(ctx context.Context, name string, args map[string]interface{}) (*MCPToolCallResult, error) {
	req := &JSONRPCRequest{
		JSONRPC: "2.0",
		ID:      c.nextID,
		Method:  "tools/call",
		Params: map[string]interface{}{
			"name":      name,
			"arguments": args,
		},
	}
	c.nextID++

	resp, err := c.transport.Send(ctx, req)
	if err != nil {
		return nil, err
	}

	if resp.Error != nil {
		return nil, &RPCError{Code: resp.Error.Code, Message: resp.Error.Message}
	}

	var result MCPToolCallResult
	if err := json.Unmarshal(resp.Result, &result); err != nil {
		return nil, err
	}

	return &result, nil
}

// Close closes the MCP client
func (c *Client) Close() error {
	return c.transport.Close()
}

// RPCError represents an RPC error
type RPCError struct {
	Code    int
	Message string
}

func (e *RPCError) Error() string {
	return e.Message
}
