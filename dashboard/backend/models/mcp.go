package models

import "time"

// MCPTransportType defines the transport type for MCP
type MCPTransportType string

const (
	MCPTransportStdio MCPTransportType = "stdio"
	MCPTransportSSE   MCPTransportType = "sse"
	MCPTransportHTTP  MCPTransportType = "http"
)

// MCPServerStatus defines the status of an MCP server
type MCPServerStatus string

const (
	MCPServerConnected    MCPServerStatus = "connected"
	MCPServerDisconnected MCPServerStatus = "disconnected"
	MCPServerError        MCPServerStatus = "error"
	MCPServerConnecting   MCPServerStatus = "connecting"
)

// MCPServer represents an MCP server configuration
type MCPServer struct {
	ID            string            `json:"id"`
	Name          string            `json:"name"`
	TransportType MCPTransportType  `json:"transport_type"`

	// stdio transport config
	Command string            `json:"command,omitempty"`
	Args    []string          `json:"args,omitempty"`
	Env     map[string]string `json:"env,omitempty"`

	// SSE/HTTP transport config
	URL     string            `json:"url,omitempty"`
	Headers map[string]string `json:"headers,omitempty"`

	// Status
	Status    MCPServerStatus `json:"status"`
	Error     string          `json:"error,omitempty"`
	ToolCount int             `json:"tool_count"`

	// Timestamps
	CreatedAt time.Time `json:"created_at"`
	UpdatedAt time.Time `json:"updated_at"`
}

// MCPToolDefinition represents a tool discovered from MCP
type MCPToolDefinition struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	InputSchema map[string]interface{} `json:"inputSchema"`
}

// ToTool converts MCP tool definition to internal Tool model
func (m *MCPToolDefinition) ToTool(serverID, serverName string) *Tool {
	params := []ToolParameter{}

	if m.InputSchema != nil {
		if props, ok := m.InputSchema["properties"].(map[string]interface{}); ok {
			required := []string{}
			if req, ok := m.InputSchema["required"].([]interface{}); ok {
				for _, r := range req {
					if s, ok := r.(string); ok {
						required = append(required, s)
					}
				}
			}

			for name, prop := range props {
				if propMap, ok := prop.(map[string]interface{}); ok {
					param := ToolParameter{
						Name: name,
					}
					if t, ok := propMap["type"].(string); ok {
						param.Type = t
					}
					if d, ok := propMap["description"].(string); ok {
						param.Description = d
					}
					// Check if required
					for _, r := range required {
						if r == name {
							param.Required = true
							break
						}
					}
					if e, ok := propMap["enum"].([]interface{}); ok {
						param.Enum = e
					}
					if def, ok := propMap["default"]; ok {
						param.Default = def
					}
					params = append(params, param)
				}
			}
		}
	}

	return &Tool{
		ID:          serverID + "/" + m.Name,
		Name:        m.Name,
		Description: m.Description,
		Source:      ToolSourceMCP,
		MCPServer:   serverName,
		Parameters:  params,
		Enabled:     true,
	}
}
