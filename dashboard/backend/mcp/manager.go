package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"gopkg.in/yaml.v3"
)

// Manager MCP 客户端管理器
type Manager struct {
	configDir string

	mu      sync.RWMutex
	clients map[string]*Client
	configs map[string]*ServerConfig
}

// NewManager 创建管理器
func NewManager(configDir string) *Manager {
	return &Manager{
		configDir: configDir,
		clients:   make(map[string]*Client),
		configs:   make(map[string]*ServerConfig),
	}
}

// LoadConfig 从配置文件加载服务器配置
func (m *Manager) LoadConfig() error {
	configPath := filepath.Join(m.configDir, "mcp-servers.yaml")

	data, err := os.ReadFile(configPath)
	if err != nil {
		if os.IsNotExist(err) {
			// 配置文件不存在，创建默认配置
			return m.saveConfig()
		}
		return fmt.Errorf("failed to read config file: %w", err)
	}

	var configFile ServersConfigFile
	if err := yaml.Unmarshal(data, &configFile); err != nil {
		return fmt.Errorf("failed to parse config file: %w", err)
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	// 清理现有配置
	m.configs = make(map[string]*ServerConfig)

	// 加载新配置
	for i := range configFile.Servers {
		config := &configFile.Servers[i]
		m.configs[config.ID] = config
	}

	return nil
}

// saveConfig 保存配置到文件
func (m *Manager) saveConfig() error {
	m.mu.RLock()
	servers := make([]ServerConfig, 0, len(m.configs))
	for _, config := range m.configs {
		servers = append(servers, *config)
	}
	m.mu.RUnlock()

	configFile := ServersConfigFile{
		Version:         "1.0",
		ProtocolVersion: "2025-06-18",
		Servers:         servers,
	}

	data, err := yaml.Marshal(&configFile)
	if err != nil {
		return fmt.Errorf("failed to marshal config: %w", err)
	}

	configPath := filepath.Join(m.configDir, "mcp-servers.yaml")
	if err := os.WriteFile(configPath, data, 0644); err != nil {
		return fmt.Errorf("failed to write config file: %w", err)
	}

	return nil
}

// GetServers 获取所有服务器配置
func (m *Manager) GetServers() []*ServerConfig {
	m.mu.RLock()
	defer m.mu.RUnlock()

	servers := make([]*ServerConfig, 0, len(m.configs))
	for _, config := range m.configs {
		servers = append(servers, config)
	}
	return servers
}

// GetServer 获取单个服务器配置
func (m *Manager) GetServer(id string) (*ServerConfig, bool) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	config, ok := m.configs[id]
	return config, ok
}

// AddServer 添加服务器配置
func (m *Manager) AddServer(config *ServerConfig) error {
	m.mu.Lock()
	if _, exists := m.configs[config.ID]; exists {
		m.mu.Unlock()
		return fmt.Errorf("server with ID %s already exists", config.ID)
	}
	m.configs[config.ID] = config
	m.mu.Unlock()

	return m.saveConfig()
}

// UpdateServer 更新服务器配置
func (m *Manager) UpdateServer(config *ServerConfig) error {
	m.mu.Lock()
	if _, exists := m.configs[config.ID]; !exists {
		m.mu.Unlock()
		return fmt.Errorf("server with ID %s not found", config.ID)
	}

	// 如果已连接，先断开
	if client, ok := m.clients[config.ID]; ok {
		_ = client.Disconnect()
		delete(m.clients, config.ID)
	}

	m.configs[config.ID] = config
	m.mu.Unlock()

	return m.saveConfig()
}

// DeleteServer 删除服务器配置
func (m *Manager) DeleteServer(id string) error {
	m.mu.Lock()
	if _, exists := m.configs[id]; !exists {
		m.mu.Unlock()
		return fmt.Errorf("server with ID %s not found", id)
	}

	// 如果已连接，先断开
	if client, ok := m.clients[id]; ok {
		_ = client.Disconnect()
		delete(m.clients, id)
	}

	delete(m.configs, id)
	m.mu.Unlock()

	return m.saveConfig()
}

// Connect 连接到指定服务器
func (m *Manager) Connect(ctx context.Context, id string) error {
	m.mu.Lock()
	config, ok := m.configs[id]
	if !ok {
		m.mu.Unlock()
		return fmt.Errorf("server with ID %s not found", id)
	}

	// 如果已有客户端，先断开
	if client, ok := m.clients[id]; ok {
		_ = client.Disconnect()
	}

	// 创建新客户端
	client, err := NewClient(config)
	if err != nil {
		m.mu.Unlock()
		return err
	}

	m.clients[id] = client
	m.mu.Unlock()

	// 连接
	return client.Connect(ctx)
}

// Disconnect 断开与指定服务器的连接
func (m *Manager) Disconnect(id string) error {
	m.mu.Lock()
	defer m.mu.Unlock()

	client, ok := m.clients[id]
	if !ok {
		return nil
	}

	err := client.Disconnect()
	delete(m.clients, id)

	return err
}

// GetServerStatus 获取服务器状态
func (m *Manager) GetServerStatus(id string) (*ServerState, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()

	config, ok := m.configs[id]
	if !ok {
		return nil, fmt.Errorf("server with ID %s not found", id)
	}

	client, ok := m.clients[id]
	if !ok {
		return &ServerState{
			Config: config,
			Status: StatusDisconnected,
		}, nil
	}

	return client.GetState(), nil
}

// GetAllServerStates 获取所有服务器状态
func (m *Manager) GetAllServerStates() []*ServerState {
	m.mu.RLock()
	defer m.mu.RUnlock()

	states := make([]*ServerState, 0, len(m.configs))
	for id, config := range m.configs {
		if client, ok := m.clients[id]; ok {
			states = append(states, client.GetState())
		} else {
			states = append(states, &ServerState{
				Config: config,
				Status: StatusDisconnected,
			})
		}
	}

	return states
}

// GetAllTools 获取所有已连接服务器的工具
func (m *Manager) GetAllTools() []Tool {
	m.mu.RLock()
	defer m.mu.RUnlock()

	var tools []Tool
	for id, client := range m.clients {
		if client.GetStatus() != StatusConnected {
			continue
		}

		config := m.configs[id]
		for _, tool := range client.GetTools() {
			tools = append(tools, Tool{
				ToolDefinition: tool,
				ServerID:       id,
				ServerName:     config.Name,
			})
		}
	}

	return tools
}

// ExecuteTool 执行工具
func (m *Manager) ExecuteTool(ctx context.Context, serverID, toolName string, arguments json.RawMessage) (*ToolResult, error) {
	m.mu.RLock()
	client, ok := m.clients[serverID]
	m.mu.RUnlock()

	if !ok {
		return nil, fmt.Errorf("server %s not connected", serverID)
	}

	if client.GetStatus() != StatusConnected {
		return nil, fmt.Errorf("server %s not connected", serverID)
	}

	start := time.Now()
	result, err := client.CallTool(ctx, toolName, arguments)
	elapsed := time.Since(start)

	if err != nil {
		return &ToolResult{
			Success:         false,
			Error:           err.Error(),
			ExecutionTimeMs: elapsed.Milliseconds(),
		}, nil
	}

	// 转换内容
	var content interface{}
	if len(result.Content) > 0 {
		if len(result.Content) == 1 && result.Content[0].Type == "text" {
			content = result.Content[0].Text
		} else {
			content = result.Content
		}
	}

	return &ToolResult{
		Success:         !result.IsError,
		Result:          content,
		ExecutionTimeMs: elapsed.Milliseconds(),
	}, nil
}

// ExecuteToolStreaming 流式执行工具
func (m *Manager) ExecuteToolStreaming(ctx context.Context, serverID, toolName string, arguments json.RawMessage, onChunk func(StreamChunk) error) error {
	m.mu.RLock()
	client, ok := m.clients[serverID]
	m.mu.RUnlock()

	if !ok {
		return fmt.Errorf("server %s not connected", serverID)
	}

	if client.GetStatus() != StatusConnected {
		return fmt.Errorf("server %s not connected", serverID)
	}

	return client.CallToolStreaming(ctx, toolName, arguments, onChunk)
}

// TestConnection 测试连接
func (m *Manager) TestConnection(ctx context.Context, config *ServerConfig) error {
	client, err := NewClient(config)
	if err != nil {
		return err
	}
	defer client.Disconnect()

	return client.Connect(ctx)
}

// ConnectEnabled 连接所有已启用的服务器
func (m *Manager) ConnectEnabled(ctx context.Context) {
	m.mu.RLock()
	configs := make([]*ServerConfig, 0)
	for _, config := range m.configs {
		if config.Enabled {
			configs = append(configs, config)
		}
	}
	m.mu.RUnlock()

	for _, config := range configs {
		go func(c *ServerConfig) {
			if err := m.Connect(ctx, c.ID); err != nil {
				fmt.Printf("Failed to connect to MCP server %s: %v\n", c.Name, err)
			} else {
				fmt.Printf("Connected to MCP server %s\n", c.Name)
			}
		}(config)
	}
}

// DisconnectAll 断开所有连接
func (m *Manager) DisconnectAll() {
	m.mu.Lock()
	defer m.mu.Unlock()

	for id, client := range m.clients {
		client.Disconnect()
		delete(m.clients, id)
	}
}
