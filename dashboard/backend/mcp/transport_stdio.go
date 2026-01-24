package mcp

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"sync"
	"sync/atomic"
)

// StdioTransport Stdio 传输实现
type StdioTransport struct {
	config *StdioConfig

	mu        sync.RWMutex
	cmd       *exec.Cmd
	stdin     io.WriteCloser
	stdout    io.ReadCloser
	stderr    io.ReadCloser
	connected bool

	// 请求 ID 计数器
	requestID atomic.Int64

	// 响应通道映射
	pendingMu sync.RWMutex
	pending   map[int64]chan *JSONRPCResponse

	// 通知处理器
	notifyHandler func(method string, params json.RawMessage)
}

// StdioConfig Stdio 传输配置
type StdioConfig struct {
	Command string
	Args    []string
	Env     map[string]string
	Cwd     string
}

// NewStdioTransport 创建 Stdio 传输
func NewStdioTransport(config *StdioConfig) *StdioTransport {
	return &StdioTransport{
		config:  config,
		pending: make(map[int64]chan *JSONRPCResponse),
	}
}

// Connect 启动子进程并建立连接
func (t *StdioTransport) Connect(ctx context.Context) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	if t.connected {
		return nil
	}

	// 创建命令
	t.cmd = exec.CommandContext(ctx, t.config.Command, t.config.Args...)

	// 设置工作目录
	if t.config.Cwd != "" {
		t.cmd.Dir = t.config.Cwd
	}

	// 设置环境变量
	t.cmd.Env = os.Environ()
	for k, v := range t.config.Env {
		t.cmd.Env = append(t.cmd.Env, fmt.Sprintf("%s=%s", k, v))
	}

	// 获取 stdin/stdout/stderr
	var err error
	t.stdin, err = t.cmd.StdinPipe()
	if err != nil {
		return fmt.Errorf("failed to get stdin pipe: %w", err)
	}

	t.stdout, err = t.cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("failed to get stdout pipe: %w", err)
	}

	t.stderr, err = t.cmd.StderrPipe()
	if err != nil {
		return fmt.Errorf("failed to get stderr pipe: %w", err)
	}

	// 启动进程
	if err := t.cmd.Start(); err != nil {
		return fmt.Errorf("failed to start process: %w", err)
	}

	t.connected = true

	// 启动响应读取协程
	go t.readResponses()

	// 启动 stderr 读取协程 (用于调试)
	go t.readStderr()

	return nil
}

// Disconnect 停止子进程
func (t *StdioTransport) Disconnect() error {
	t.mu.Lock()
	defer t.mu.Unlock()

	if !t.connected {
		return nil
	}

	t.connected = false

	// 关闭 stdin 以通知子进程退出
	if t.stdin != nil {
		_ = t.stdin.Close()
	}

	// 等待进程退出
	if t.cmd != nil && t.cmd.Process != nil {
		_ = t.cmd.Process.Kill()
		_ = t.cmd.Wait()
	}

	// 清理 pending 请求
	t.pendingMu.Lock()
	for id, ch := range t.pending {
		close(ch)
		delete(t.pending, id)
	}
	t.pendingMu.Unlock()

	return nil
}

// IsConnected 检查连接状态
func (t *StdioTransport) IsConnected() bool {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.connected
}

// Call 执行同步调用
func (t *StdioTransport) Call(ctx context.Context, method string, params interface{}) (interface{}, error) {
	if !t.IsConnected() {
		return nil, fmt.Errorf("not connected")
	}

	// 生成请求 ID
	id := t.requestID.Add(1)

	// 创建请求
	req := JSONRPCRequest{
		JSONRPC: "2.0",
		ID:      id,
		Method:  method,
		Params:  params,
	}

	// 创建响应通道
	respCh := make(chan *JSONRPCResponse, 1)
	t.pendingMu.Lock()
	t.pending[id] = respCh
	t.pendingMu.Unlock()

	defer func() {
		t.pendingMu.Lock()
		delete(t.pending, id)
		t.pendingMu.Unlock()
	}()

	// 发送请求
	reqBytes, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	t.mu.Lock()
	_, err = t.stdin.Write(append(reqBytes, '\n'))
	t.mu.Unlock()

	if err != nil {
		return nil, fmt.Errorf("failed to write request: %w", err)
	}

	// 等待响应
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case resp, ok := <-respCh:
		if !ok {
			return nil, fmt.Errorf("connection closed")
		}
		if resp.Error != nil {
			return nil, fmt.Errorf("RPC error %d: %s", resp.Error.Code, resp.Error.Message)
		}

		// 解析结果
		var result interface{}
		if len(resp.Result) > 0 {
			if err := json.Unmarshal(resp.Result, &result); err != nil {
				return nil, fmt.Errorf("failed to unmarshal result: %w", err)
			}
		}
		return result, nil
	}
}

// CallStreaming Stdio 不支持真正的流式，模拟实现
func (t *StdioTransport) CallStreaming(ctx context.Context, method string, params interface{}, onChunk func(StreamChunk) error) error {
	// Stdio 传输通常不支持流式响应，直接调用同步方法
	result, err := t.Call(ctx, method, params)
	if err != nil {
		return onChunk(StreamChunk{Type: "error", Data: err.Error()})
	}
	return onChunk(StreamChunk{Type: "complete", Data: result, Progress: 100})
}

// readResponses 读取子进程响应
func (t *StdioTransport) readResponses() {
	scanner := bufio.NewScanner(t.stdout)
	// 增大缓冲区以处理大响应
	scanner.Buffer(make([]byte, 1024*1024), 10*1024*1024)

	for scanner.Scan() {
		line := scanner.Bytes()
		if len(line) == 0 {
			continue
		}

		// 尝试解析为 JSON-RPC 响应
		var resp JSONRPCResponse
		if err := json.Unmarshal(line, &resp); err != nil {
			// 可能是通知或其他格式
			continue
		}

		// 如果有 ID，是响应
		if resp.ID != nil {
			var id int64
			switch v := resp.ID.(type) {
			case float64:
				id = int64(v)
			case int64:
				id = v
			default:
				continue
			}

			t.pendingMu.RLock()
			ch, ok := t.pending[id]
			t.pendingMu.RUnlock()

			if ok {
				ch <- &resp
			}
		}
	}

	// 连接断开
	t.mu.Lock()
	t.connected = false
	t.mu.Unlock()
}

// readStderr 读取 stderr (用于调试)
func (t *StdioTransport) readStderr() {
	scanner := bufio.NewScanner(t.stderr)
	for scanner.Scan() {
		// 可以记录日志或处理错误信息
		_ = scanner.Text()
	}
}

// SetNotifyHandler 设置通知处理器
func (t *StdioTransport) SetNotifyHandler(handler func(method string, params json.RawMessage)) {
	t.notifyHandler = handler
}
