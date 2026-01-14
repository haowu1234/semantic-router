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

// SSETransport implements MCP transport over Server-Sent Events
type SSETransport struct {
	url        string
	headers    map[string]string
	client     *http.Client
	sessionURL string
	mu         sync.Mutex
}

// NewSSETransport creates a new SSE transport
func NewSSETransport(url string, headers map[string]string) (*SSETransport, error) {
	return &SSETransport{
		url:     url,
		headers: headers,
		client: &http.Client{
			Timeout: 30 * time.Second,
		},
	}, nil
}

// Send sends a JSON-RPC request over SSE
func (t *SSETransport) Send(ctx context.Context, req *JSONRPCRequest) (*JSONRPCResponse, error) {
	t.mu.Lock()
	defer t.mu.Unlock()

	// For initialize, we need to establish SSE connection first
	if req.Method == "initialize" {
		return t.initialize(ctx, req)
	}

	// For other requests, use HTTP POST to the session URL
	if t.sessionURL == "" {
		return nil, fmt.Errorf("not initialized")
	}

	return t.sendRequest(ctx, req)
}

// initialize establishes the SSE connection
func (t *SSETransport) initialize(ctx context.Context, req *JSONRPCRequest) (*JSONRPCResponse, error) {
	// Connect to SSE endpoint
	httpReq, err := http.NewRequestWithContext(ctx, "GET", t.url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Accept", "text/event-stream")
	for k, v := range t.headers {
		httpReq.Header.Set(k, v)
	}

	resp, err := t.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("failed to connect: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		return nil, fmt.Errorf("SSE connection failed: %s", resp.Status)
	}

	// Read SSE events to get endpoint
	reader := bufio.NewReader(resp.Body)
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			resp.Body.Close()
			return nil, fmt.Errorf("failed to read SSE: %w", err)
		}

		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "event: endpoint") {
			// Next line should be data with the endpoint URL
			dataLine, err := reader.ReadString('\n')
			if err != nil {
				resp.Body.Close()
				return nil, fmt.Errorf("failed to read endpoint data: %w", err)
			}
			dataLine = strings.TrimSpace(dataLine)
			if strings.HasPrefix(dataLine, "data: ") {
				t.sessionURL = strings.TrimPrefix(dataLine, "data: ")
				break
			}
		}
	}

	// Close the SSE connection for now (we'll use HTTP for requests)
	resp.Body.Close()

	// Send initialize request
	return t.sendRequest(ctx, req)
}

// sendRequest sends a request to the session URL
func (t *SSETransport) sendRequest(ctx context.Context, req *JSONRPCRequest) (*JSONRPCResponse, error) {
	data, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", t.sessionURL, bytes.NewReader(data))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	httpReq.Header.Set("Content-Type", "application/json")
	for k, v := range t.headers {
		httpReq.Header.Set(k, v)
	}

	resp, err := t.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusAccepted {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("request failed: %s - %s", resp.Status, string(body))
	}

	// For notifications, no response expected
	if req.ID == 0 {
		return &JSONRPCResponse{JSONRPC: "2.0"}, nil
	}

	var rpcResp JSONRPCResponse
	if err := json.NewDecoder(resp.Body).Decode(&rpcResp); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	return &rpcResp, nil
}

// Close closes the SSE transport
func (t *SSETransport) Close() error {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.sessionURL = ""
	return nil
}
