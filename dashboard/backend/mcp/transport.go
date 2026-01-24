package mcp

import (
	"context"
)

// Transport MCP 传输层接口
type Transport interface {
	// Connect 建立连接
	Connect(ctx context.Context) error

	// Disconnect 断开连接
	Disconnect() error

	// IsConnected 检查连接状态
	IsConnected() bool

	// Call 执行同步调用
	Call(ctx context.Context, method string, params interface{}) (interface{}, error)

	// CallStreaming 执行流式调用
	CallStreaming(ctx context.Context, method string, params interface{}, onChunk func(StreamChunk) error) error
}
