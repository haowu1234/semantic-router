package handlers

import (
	"encoding/json"
	"net/http"
	"strings"

	"github.com/google/uuid"
	"github.com/vllm-project/semantic-router/dashboard/backend/mcp"
	"github.com/vllm-project/semantic-router/dashboard/backend/middleware"
)

// MCPHandler MCP 相关的 HTTP 处理器
type MCPHandler struct {
	manager      *mcp.Manager
	readonlyMode bool
}

// NewMCPHandler 创建 MCP Handler
func NewMCPHandler(manager *mcp.Manager, readonlyMode bool) *MCPHandler {
	return &MCPHandler{
		manager:      manager,
		readonlyMode: readonlyMode,
	}
}

// ========== Server Config Handlers ==========

// ListServersHandler GET /api/mcp/servers - 获取所有服务器配置
func (h *MCPHandler) ListServersHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}

		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		states := h.manager.GetAllServerStates()

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"servers": states,
		})
	}
}

// CreateServerHandler POST /api/mcp/servers - 创建服务器配置
func (h *MCPHandler) CreateServerHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}

		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		if h.readonlyMode {
			http.Error(w, "Operation not allowed in readonly mode", http.StatusForbidden)
			return
		}

		var config mcp.ServerConfig
		if err := json.NewDecoder(r.Body).Decode(&config); err != nil {
			http.Error(w, "Invalid request body: "+err.Error(), http.StatusBadRequest)
			return
		}

		// 生成 ID
		if config.ID == "" {
			config.ID = uuid.New().String()
		}

		// 验证必填字段
		if config.Name == "" {
			http.Error(w, "Name is required", http.StatusBadRequest)
			return
		}

		if config.Transport == "" {
			http.Error(w, "Transport is required", http.StatusBadRequest)
			return
		}

		// 验证传输类型
		if config.Transport != mcp.TransportStdio && config.Transport != mcp.TransportStreamableHTTP {
			http.Error(w, "Invalid transport type. Must be 'stdio' or 'streamable-http'", http.StatusBadRequest)
			return
		}

		// 验证连接配置
		switch config.Transport {
		case mcp.TransportStdio:
			if config.Connection.Command == "" {
				http.Error(w, "Command is required for stdio transport", http.StatusBadRequest)
				return
			}
		case mcp.TransportStreamableHTTP:
			if config.Connection.URL == "" {
				http.Error(w, "URL is required for streamable-http transport", http.StatusBadRequest)
				return
			}
		}

		if err := h.manager.AddServer(&config); err != nil {
			http.Error(w, "Failed to add server: "+err.Error(), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		_ = json.NewEncoder(w).Encode(config)
	}
}

// UpdateServerHandler PUT /api/mcp/servers/:id - 更新服务器配置
func (h *MCPHandler) UpdateServerHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}

		if r.Method != http.MethodPut {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		if h.readonlyMode {
			http.Error(w, "Operation not allowed in readonly mode", http.StatusForbidden)
			return
		}

		// 从 URL 提取 ID
		id := strings.TrimPrefix(r.URL.Path, "/api/mcp/servers/")
		if id == "" {
			http.Error(w, "Server ID is required", http.StatusBadRequest)
			return
		}

		var config mcp.ServerConfig
		if err := json.NewDecoder(r.Body).Decode(&config); err != nil {
			http.Error(w, "Invalid request body: "+err.Error(), http.StatusBadRequest)
			return
		}

		config.ID = id

		if err := h.manager.UpdateServer(&config); err != nil {
			http.Error(w, "Failed to update server: "+err.Error(), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(config)
	}
}

// DeleteServerHandler DELETE /api/mcp/servers/:id - 删除服务器配置
func (h *MCPHandler) DeleteServerHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}

		if r.Method != http.MethodDelete {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		if h.readonlyMode {
			http.Error(w, "Operation not allowed in readonly mode", http.StatusForbidden)
			return
		}

		// 从 URL 提取 ID
		id := strings.TrimPrefix(r.URL.Path, "/api/mcp/servers/")
		if id == "" {
			http.Error(w, "Server ID is required", http.StatusBadRequest)
			return
		}

		if err := h.manager.DeleteServer(id); err != nil {
			http.Error(w, "Failed to delete server: "+err.Error(), http.StatusInternalServerError)
			return
		}

		w.WriteHeader(http.StatusNoContent)
	}
}

// ========== Connection Handlers ==========

// ConnectServerHandler POST /api/mcp/servers/:id/connect - 连接服务器
func (h *MCPHandler) ConnectServerHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}

		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// 从 URL 提取 ID
		path := strings.TrimPrefix(r.URL.Path, "/api/mcp/servers/")
		id := strings.TrimSuffix(path, "/connect")
		if id == "" {
			http.Error(w, "Server ID is required", http.StatusBadRequest)
			return
		}

		if err := h.manager.Connect(r.Context(), id); err != nil {
			http.Error(w, "Failed to connect: "+err.Error(), http.StatusInternalServerError)
			return
		}

		state, _ := h.manager.GetServerStatus(id)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(state)
	}
}

// DisconnectServerHandler POST /api/mcp/servers/:id/disconnect - 断开连接
func (h *MCPHandler) DisconnectServerHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}

		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// 从 URL 提取 ID
		path := strings.TrimPrefix(r.URL.Path, "/api/mcp/servers/")
		id := strings.TrimSuffix(path, "/disconnect")
		if id == "" {
			http.Error(w, "Server ID is required", http.StatusBadRequest)
			return
		}

		if err := h.manager.Disconnect(id); err != nil {
			http.Error(w, "Failed to disconnect: "+err.Error(), http.StatusInternalServerError)
			return
		}

		state, _ := h.manager.GetServerStatus(id)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(state)
	}
}

// GetServerStatusHandler GET /api/mcp/servers/:id/status - 获取服务器状态
func (h *MCPHandler) GetServerStatusHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}

		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// 从 URL 提取 ID
		path := strings.TrimPrefix(r.URL.Path, "/api/mcp/servers/")
		id := strings.TrimSuffix(path, "/status")
		if id == "" {
			http.Error(w, "Server ID is required", http.StatusBadRequest)
			return
		}

		state, err := h.manager.GetServerStatus(id)
		if err != nil {
			http.Error(w, err.Error(), http.StatusNotFound)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(state)
	}
}

// TestConnectionHandler POST /api/mcp/servers/:id/test - 测试连接
func (h *MCPHandler) TestConnectionHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}

		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var config mcp.ServerConfig
		if err := json.NewDecoder(r.Body).Decode(&config); err != nil {
			http.Error(w, "Invalid request body: "+err.Error(), http.StatusBadRequest)
			return
		}

		if err := h.manager.TestConnection(r.Context(), &config); err != nil {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusOK)
			json.NewEncoder(w).Encode(map[string]interface{}{
				"success": false,
				"error":   err.Error(),
			})
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"success": true,
		})
	}
}

// ========== Tool Handlers ==========

// ListToolsHandler GET /api/mcp/tools - 获取所有工具
func (h *MCPHandler) ListToolsHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}

		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		tools := h.manager.GetAllTools()

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"tools": tools,
		})
	}
}

// ExecuteToolHandler POST /api/mcp/tools/execute - 执行工具
func (h *MCPHandler) ExecuteToolHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}

		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req mcp.ToolExecuteRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid request body: "+err.Error(), http.StatusBadRequest)
			return
		}

		if req.ServerID == "" {
			http.Error(w, "server_id is required", http.StatusBadRequest)
			return
		}

		if req.ToolName == "" {
			http.Error(w, "tool_name is required", http.StatusBadRequest)
			return
		}

		result, err := h.manager.ExecuteTool(r.Context(), req.ServerID, req.ToolName, req.Arguments)
		if err != nil {
			http.Error(w, "Failed to execute tool: "+err.Error(), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(result)
	}
}

// ExecuteToolStreamHandler POST /api/mcp/tools/execute/stream - 流式执行工具
func (h *MCPHandler) ExecuteToolStreamHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}

		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req mcp.ToolExecuteRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid request body: "+err.Error(), http.StatusBadRequest)
			return
		}

		if req.ServerID == "" {
			http.Error(w, "server_id is required", http.StatusBadRequest)
			return
		}

		if req.ToolName == "" {
			http.Error(w, "tool_name is required", http.StatusBadRequest)
			return
		}

		// 设置 SSE headers
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")

		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "Streaming not supported", http.StatusInternalServerError)
			return
		}

		err := h.manager.ExecuteToolStreaming(r.Context(), req.ServerID, req.ToolName, req.Arguments, func(chunk mcp.StreamChunk) error {
			data, _ := json.Marshal(chunk)
			_, err := w.Write([]byte("event: message\n"))
			if err != nil {
				return err
			}
			_, err = w.Write([]byte("data: " + string(data) + "\n\n"))
			if err != nil {
				return err
			}
			flusher.Flush()
			return nil
		})

		if err != nil {
			// 发送错误事件
			errData, _ := json.Marshal(map[string]string{"error": err.Error()})
			_, _ = w.Write([]byte("event: error\n"))
			_, _ = w.Write([]byte("data: " + string(errData) + "\n\n"))
			flusher.Flush()
		}
	}
}
