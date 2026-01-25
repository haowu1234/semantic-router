package mcp

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"strings"
	"sync"
	"sync/atomic"
	"time"
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
// 注意：ctx 仅用于控制连接初始化超时，不会传递给子进程
// 子进程的生命周期由 Disconnect() 方法管理
func (t *StdioTransport) Connect(ctx context.Context) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	log.Printf("[MCP-Stdio] Connect() called, command: %s, args: %v", t.config.Command, t.config.Args)

	if t.connected {
		log.Printf("[MCP-Stdio] Already connected, skipping")
		return nil
	}

	// 创建命令 - 不使用 ctx，因为我们希望子进程在 HTTP 请求结束后继续运行
	// 子进程的生命周期由 Disconnect() 方法管理
	t.cmd = exec.Command(t.config.Command, t.config.Args...)
	log.Printf("[MCP-Stdio] Created command (without context): %s %v", t.config.Command, t.config.Args)

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
	log.Printf("[MCP-Stdio] Starting process...")
	if err := t.cmd.Start(); err != nil {
		log.Printf("[MCP-Stdio] Failed to start process: %v", err)
		// 提供更详细的错误信息
		if os.IsNotExist(err) || (err.Error() != "" && (strings.Contains(err.Error(), "not found") || strings.Contains(err.Error(), "no such file"))) {
			return fmt.Errorf("command not found: '%s' - please ensure the command is installed and in PATH", t.config.Command)
		}
		if os.IsPermission(err) || (err.Error() != "" && strings.Contains(err.Error(), "permission denied")) {
			return fmt.Errorf("permission denied: cannot execute '%s' - please check file permissions", t.config.Command)
		}
		return fmt.Errorf("failed to start process '%s': %w", t.config.Command, err)
	}
	log.Printf("[MCP-Stdio] Process started successfully, PID: %d", t.cmd.Process.Pid)

	// 创建用于收集 stderr 的 buffer 和同步通道
	stderrBuf := &strings.Builder{}
	processExited := make(chan error, 1)

	// 启动进程监控协程，监控进程退出状态
	go func() {
		err := t.cmd.Wait()
		processExited <- err
	}()

	// 启动 stderr 读取协程 (用于调试和捕获错误)
	log.Printf("[MCP-Stdio] Starting stderr reader goroutine")
	go t.readStderrWithBuffer(stderrBuf)

	// 等待一小段时间，检查进程是否立即退出
	// 这对于捕获 "command not found"、"file not found" 等启动错误很重要
	log.Printf("[MCP-Stdio] Waiting briefly to check if process exits immediately...")

	select {
	case err := <-processExited:
		// 进程立即退出了，这通常意味着启动失败
		// 多等一小会儿让 stderr 读取完成
		time.Sleep(50 * time.Millisecond)
		stderrOutput := stderrBuf.String()
		log.Printf("[MCP-Stdio] Process exited immediately with error: %v, stderr: %s", err, stderrOutput)

		errMsg := "process exited immediately"
		if err != nil {
			errMsg = fmt.Sprintf("process exited immediately with: %v", err)
		}
		if stderrOutput != "" {
			errMsg = fmt.Sprintf("%s, stderr: %s", errMsg, strings.TrimSpace(stderrOutput))
		}
		return fmt.Errorf("failed to start MCP server: %s", errMsg)

	case <-time.After(200 * time.Millisecond):
		// 进程在 200ms 后仍在运行，认为启动成功
		log.Printf("[MCP-Stdio] Process still running after 200ms, assuming successful start")
	}

	t.connected = true

	// 启动后台协程监控进程退出
	go func() {
		err := <-processExited
		log.Printf("[MCP-Stdio] Process exited: %v", err)
		t.mu.Lock()
		t.connected = false
		t.mu.Unlock()
	}()

	// 启动响应读取协程
	log.Printf("[MCP-Stdio] Starting response reader goroutine")
	go t.readResponses()

	log.Printf("[MCP-Stdio] Connect() completed successfully")
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

	// 强制杀死进程（Wait() 已在后台协程中处理）
	if t.cmd != nil && t.cmd.Process != nil {
		_ = t.cmd.Process.Kill()
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

	log.Printf("[MCP-Stdio] Sending request (id=%d, method=%s): %s", id, method, string(reqBytes))

	t.mu.Lock()
	_, err = t.stdin.Write(append(reqBytes, '\n'))
	t.mu.Unlock()

	if err != nil {
		log.Printf("[MCP-Stdio] Failed to write request: %v", err)
		return nil, fmt.Errorf("failed to write request: %w", err)
	}
	log.Printf("[MCP-Stdio] Request sent successfully, waiting for response...")

	// 等待响应
	log.Printf("[MCP-Stdio] Waiting for response (id=%d)...", id)
	select {
	case <-ctx.Done():
		log.Printf("[MCP-Stdio] Context cancelled while waiting for response (id=%d): %v", id, ctx.Err())
		return nil, ctx.Err()
	case resp, ok := <-respCh:
		if !ok {
			log.Printf("[MCP-Stdio] Response channel closed (id=%d)", id)
			return nil, fmt.Errorf("connection closed")
		}
		log.Printf("[MCP-Stdio] Received response (id=%d): result_len=%d, error=%v", id, len(resp.Result), resp.Error)
		if resp.Error != nil {
			log.Printf("[MCP-Stdio] RPC error (id=%d): code=%d, message=%s", id, resp.Error.Code, resp.Error.Message)
			return nil, fmt.Errorf("RPC error %d: %s", resp.Error.Code, resp.Error.Message)
		}

		// 解析结果
		var result interface{}
		if len(resp.Result) > 0 {
			if err := json.Unmarshal(resp.Result, &result); err != nil {
				return nil, fmt.Errorf("failed to unmarshal result: %w", err)
			}
		}
		log.Printf("[MCP-Stdio] Request completed successfully (id=%d)", id)
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
	log.Printf("[MCP-Stdio] readResponses() goroutine started")

	// 使用 bufio.Reader 而不是 Scanner，更可靠地处理行输入
	reader := bufio.NewReader(t.stdout)
	log.Printf("[MCP-Stdio] Reader created, starting read loop...")

	for {
		log.Printf("[MCP-Stdio] Waiting to read line from stdout...")
		line, err := reader.ReadBytes('\n')
		if err != nil {
			if err == io.EOF {
				log.Printf("[MCP-Stdio] EOF received from stdout")
			} else {
				log.Printf("[MCP-Stdio] Read error: %v", err)
			}
			break
		}

		log.Printf("[MCP-Stdio] Raw line received (len=%d): %s", len(line), string(line))
		// 去除换行符
		line = []byte(strings.TrimSpace(string(line)))
		log.Printf("[MCP-Stdio] Trimmed line (len=%d): %s", len(line), string(line))
		if len(line) == 0 {
			log.Printf("[MCP-Stdio] Empty line after trim, skipping")
			continue
		}

		// 尝试解析为 JSON-RPC 响应
		var resp JSONRPCResponse
		if err := json.Unmarshal(line, &resp); err != nil {
			// 可能是通知或其他格式
			log.Printf("[MCP-Stdio] Failed to parse as JSON-RPC response: %v", err)
			continue
		}
		log.Printf("[MCP-Stdio] Parsed JSON-RPC response: id=%v", resp.ID)

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
				log.Printf("[MCP-Stdio] Sending response to pending channel (id=%d)", id)
				ch <- &resp
			} else {
				log.Printf("[MCP-Stdio] No pending request found for id=%d", id)
			}
		}
	}

	// 连接断开
	log.Printf("[MCP-Stdio] readResponses() goroutine exiting, marking as disconnected")
	t.mu.Lock()
	t.connected = false
	t.mu.Unlock()
}

// readStderr 读取 stderr (用于调试)
func (t *StdioTransport) readStderr() {
	log.Printf("[MCP-Stdio] readStderr() goroutine started")
	scanner := bufio.NewScanner(t.stderr)
	for scanner.Scan() {
		line := scanner.Text()
		log.Printf("[MCP-Stdio] STDERR: %s", line)
	}
	if err := scanner.Err(); err != nil {
		log.Printf("[MCP-Stdio] Stderr scanner error: %v", err)
	}
	log.Printf("[MCP-Stdio] readStderr() goroutine exiting")
}

// readStderrWithBuffer 读取 stderr 并同时写入 buffer（用于启动错误检测）
func (t *StdioTransport) readStderrWithBuffer(buf *strings.Builder) {
	log.Printf("[MCP-Stdio] readStderrWithBuffer() goroutine started")
	scanner := bufio.NewScanner(t.stderr)
	for scanner.Scan() {
		line := scanner.Text()
		log.Printf("[MCP-Stdio] STDERR: %s", line)
		buf.WriteString(line)
		buf.WriteString("\n")
	}
	if err := scanner.Err(); err != nil {
		log.Printf("[MCP-Stdio] Stderr scanner error: %v", err)
	}
	log.Printf("[MCP-Stdio] readStderrWithBuffer() goroutine exiting")
}

// SetNotifyHandler 设置通知处理器
func (t *StdioTransport) SetNotifyHandler(handler func(method string, params json.RawMessage)) {
	t.notifyHandler = handler
}
