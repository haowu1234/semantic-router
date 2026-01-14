package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/vllm-project/semantic-router/dashboard/backend/models"
)

// ServerConnection holds an MCP server and its client
type ServerConnection struct {
	Server *models.MCPServer
	Client *Client
	Tools  []MCPToolDefinition
}

// Manager manages MCP server connections
type Manager struct {
	mu         sync.RWMutex
	servers    map[string]*ServerConnection
	configPath string
}

// NewManager creates a new MCP manager
func NewManager(configPath string) *Manager {
	m := &Manager{
		servers:    make(map[string]*ServerConnection),
		configPath: configPath,
	}
	// Load saved servers
	m.loadServers()
	return m
}

// AddServer adds and connects to an MCP server
func (m *Manager) AddServer(ctx context.Context, server *models.MCPServer) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	// Generate ID if not provided
	if server.ID == "" {
		server.ID = uuid.New().String()
	}

	server.CreatedAt = time.Now()
	server.UpdatedAt = time.Now()
	server.Status = models.MCPServerConnecting

	// Create transport based on type
	var transport Transport
	var err error

	switch server.TransportType {
	case models.MCPTransportStdio:
		transport, err = NewStdioTransport(server.Command, server.Args, server.Env)
	case models.MCPTransportSSE:
		transport, err = NewSSETransport(server.URL, server.Headers)
	case models.MCPTransportHTTP:
		transport, err = NewSSETransport(server.URL, server.Headers) // Use SSE transport for HTTP too
	default:
		return fmt.Errorf("unsupported transport type: %s", server.TransportType)
	}

	if err != nil {
		server.Status = models.MCPServerError
		server.Error = err.Error()
		m.servers[server.ID] = &ServerConnection{Server: server}
		m.saveServers()
		return err
	}

	// Create client and initialize
	client := NewClient(transport)
	if err := client.Initialize(ctx); err != nil {
		transport.Close()
		server.Status = models.MCPServerError
		server.Error = err.Error()
		m.servers[server.ID] = &ServerConnection{Server: server}
		m.saveServers()
		return err
	}

	// Discover tools
	tools, err := client.ListTools(ctx)
	if err != nil {
		client.Close()
		server.Status = models.MCPServerError
		server.Error = err.Error()
		m.servers[server.ID] = &ServerConnection{Server: server}
		m.saveServers()
		return err
	}

	server.Status = models.MCPServerConnected
	server.ToolCount = len(tools)
	server.Error = ""
	server.UpdatedAt = time.Now()

	m.servers[server.ID] = &ServerConnection{
		Server: server,
		Client: client,
		Tools:  tools,
	}

	m.saveServers()
	return nil
}

// RemoveServer disconnects and removes an MCP server
func (m *Manager) RemoveServer(id string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	conn, ok := m.servers[id]
	if !ok {
		return fmt.Errorf("server not found: %s", id)
	}

	if conn.Client != nil {
		conn.Client.Close()
	}

	delete(m.servers, id)
	m.saveServers()
	return nil
}

// GetServers returns all MCP servers
func (m *Manager) GetServers() []*models.MCPServer {
	m.mu.RLock()
	defer m.mu.RUnlock()

	servers := make([]*models.MCPServer, 0, len(m.servers))
	for _, conn := range m.servers {
		servers = append(servers, conn.Server)
	}
	return servers
}

// GetServer returns a specific MCP server
func (m *Manager) GetServer(id string) (*models.MCPServer, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	conn, ok := m.servers[id]
	if !ok {
		return nil, false
	}
	return conn.Server, true
}

// GetServerTools returns tools from a specific server
func (m *Manager) GetServerTools(serverID string) ([]MCPToolDefinition, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	conn, ok := m.servers[serverID]
	if !ok {
		return nil, fmt.Errorf("server not found: %s", serverID)
	}

	return conn.Tools, nil
}

// RefreshTools re-discovers tools from a server
func (m *Manager) RefreshTools(ctx context.Context, serverID string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	conn, ok := m.servers[serverID]
	if !ok {
		return fmt.Errorf("server not found: %s", serverID)
	}

	if conn.Client == nil {
		return fmt.Errorf("server not connected: %s", serverID)
	}

	tools, err := conn.Client.ListTools(ctx)
	if err != nil {
		return err
	}

	conn.Tools = tools
	conn.Server.ToolCount = len(tools)
	conn.Server.UpdatedAt = time.Now()

	m.saveServers()
	return nil
}

// CallTool calls a tool on an MCP server
func (m *Manager) CallTool(ctx context.Context, serverID, toolName string, args map[string]interface{}) (interface{}, error) {
	m.mu.RLock()
	conn, ok := m.servers[serverID]
	m.mu.RUnlock()

	if !ok {
		return nil, fmt.Errorf("server not found: %s", serverID)
	}

	if conn.Client == nil {
		return nil, fmt.Errorf("server not connected: %s", serverID)
	}

	result, err := conn.Client.CallTool(ctx, toolName, args)
	if err != nil {
		return nil, err
	}

	if result.IsError {
		return nil, fmt.Errorf("tool execution failed")
	}

	// Extract text content
	for _, content := range result.Content {
		if content.Type == "text" {
			return content.Text, nil
		}
	}

	return result.Content, nil
}

// Reconnect attempts to reconnect a disconnected server
func (m *Manager) Reconnect(ctx context.Context, serverID string) error {
	m.mu.Lock()
	conn, ok := m.servers[serverID]
	if !ok {
		m.mu.Unlock()
		return fmt.Errorf("server not found: %s", serverID)
	}

	// Close existing connection if any
	if conn.Client != nil {
		conn.Client.Close()
		conn.Client = nil
	}

	server := conn.Server
	m.mu.Unlock()

	// Re-add the server (this will reconnect)
	return m.AddServer(ctx, server)
}

// saveServers persists server configurations to disk
func (m *Manager) saveServers() {
	if m.configPath == "" {
		return
	}

	// Create config directory if needed
	mcpConfigPath := filepath.Join(m.configPath, "mcp_servers.json")

	servers := make([]*models.MCPServer, 0, len(m.servers))
	for _, conn := range m.servers {
		servers = append(servers, conn.Server)
	}

	data, err := json.MarshalIndent(servers, "", "  ")
	if err != nil {
		return
	}

	os.WriteFile(mcpConfigPath, data, 0644)
}

// loadServers loads server configurations from disk
func (m *Manager) loadServers() {
	if m.configPath == "" {
		return
	}

	mcpConfigPath := filepath.Join(m.configPath, "mcp_servers.json")
	data, err := os.ReadFile(mcpConfigPath)
	if err != nil {
		return
	}

	var servers []*models.MCPServer
	if err := json.Unmarshal(data, &servers); err != nil {
		return
	}

	// Add servers without connecting (they'll be in disconnected state)
	for _, server := range servers {
		server.Status = models.MCPServerDisconnected
		server.ToolCount = 0
		m.servers[server.ID] = &ServerConnection{Server: server}
	}
}
