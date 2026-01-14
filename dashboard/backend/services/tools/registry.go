package tools

import (
	"fmt"
	"sync"

	"github.com/vllm-project/semantic-router/dashboard/backend/models"
	"github.com/vllm-project/semantic-router/dashboard/backend/services/tools/builtin"
)

// Registry manages all available tools
type Registry struct {
	mu           sync.RWMutex
	builtinTools map[string]BuiltinTool
	mcpTools     map[string]*models.Tool // key: "serverID/toolName"
	toolEnabled  map[string]bool         // track enabled/disabled state
}

var (
	globalRegistry *Registry
	once           sync.Once
)

// GetRegistry returns the global tool registry
func GetRegistry() *Registry {
	once.Do(func() {
		globalRegistry = &Registry{
			builtinTools: make(map[string]BuiltinTool),
			mcpTools:     make(map[string]*models.Tool),
			toolEnabled:  make(map[string]bool),
		}
		// Register built-in tools
		globalRegistry.RegisterBuiltin(&builtin.Calculator{})
		globalRegistry.RegisterBuiltin(builtin.NewWebSearch())
		globalRegistry.RegisterBuiltin(builtin.NewWeather())
	})
	return globalRegistry
}

// RegisterBuiltin registers a built-in tool
func (r *Registry) RegisterBuiltin(tool BuiltinTool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.builtinTools[tool.Name()] = tool
	r.toolEnabled[tool.Name()] = true
}

// RegisterMCPTool registers an MCP tool
func (r *Registry) RegisterMCPTool(serverID string, tool *models.Tool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	key := serverID + "/" + tool.Name
	r.mcpTools[key] = tool
	r.toolEnabled[key] = true
}

// UnregisterMCPTools removes all tools from an MCP server
func (r *Registry) UnregisterMCPTools(serverID string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	for key := range r.mcpTools {
		if len(key) > len(serverID) && key[:len(serverID)+1] == serverID+"/" {
			delete(r.mcpTools, key)
			delete(r.toolEnabled, key)
		}
	}
}

// GetAllTools returns all registered tools
func (r *Registry) GetAllTools() []*models.Tool {
	r.mu.RLock()
	defer r.mu.RUnlock()

	tools := make([]*models.Tool, 0, len(r.builtinTools)+len(r.mcpTools))

	// Add built-in tools
	for _, bt := range r.builtinTools {
		enabled := r.toolEnabled[bt.Name()]
		tools = append(tools, &models.Tool{
			ID:          bt.Name(),
			Name:        bt.Name(),
			Description: bt.Description(),
			Source:      models.ToolSourceBuiltin,
			Parameters:  bt.Parameters(),
			Enabled:     enabled,
		})
	}

	// Add MCP tools
	for key, mt := range r.mcpTools {
		enabled := r.toolEnabled[key]
		toolCopy := *mt
		toolCopy.Enabled = enabled
		tools = append(tools, &toolCopy)
	}

	return tools
}

// GetEnabledTools returns only enabled tools
func (r *Registry) GetEnabledTools() []*models.Tool {
	r.mu.RLock()
	defer r.mu.RUnlock()

	tools := make([]*models.Tool, 0)

	// Add enabled built-in tools
	for _, bt := range r.builtinTools {
		if r.toolEnabled[bt.Name()] {
			tools = append(tools, &models.Tool{
				ID:          bt.Name(),
				Name:        bt.Name(),
				Description: bt.Description(),
				Source:      models.ToolSourceBuiltin,
				Parameters:  bt.Parameters(),
				Enabled:     true,
			})
		}
	}

	// Add enabled MCP tools
	for key, mt := range r.mcpTools {
		if r.toolEnabled[key] {
			toolCopy := *mt
			toolCopy.Enabled = true
			tools = append(tools, &toolCopy)
		}
	}

	return tools
}

// GetTool returns a specific tool by ID
func (r *Registry) GetTool(id string) (*models.Tool, BuiltinTool, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	// Check built-in tools first
	if bt, ok := r.builtinTools[id]; ok {
		return nil, bt, true
	}

	// Check MCP tools
	if mt, ok := r.mcpTools[id]; ok {
		return mt, nil, true
	}

	return nil, nil, false
}

// GetBuiltinTool returns a built-in tool by name
func (r *Registry) GetBuiltinTool(name string) (BuiltinTool, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	bt, ok := r.builtinTools[name]
	return bt, ok
}

// SetToolEnabled enables or disables a tool
func (r *Registry) SetToolEnabled(toolID string, enabled bool) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Check if tool exists
	if _, ok := r.builtinTools[toolID]; ok {
		r.toolEnabled[toolID] = enabled
		return nil
	}
	if _, ok := r.mcpTools[toolID]; ok {
		r.toolEnabled[toolID] = enabled
		return nil
	}

	return fmt.Errorf("tool not found: %s", toolID)
}

// IsToolEnabled checks if a tool is enabled
func (r *Registry) IsToolEnabled(toolID string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.toolEnabled[toolID]
}
