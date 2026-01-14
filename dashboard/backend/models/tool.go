package models

// ToolSource indicates where the tool comes from
type ToolSource string

const (
	ToolSourceBuiltin ToolSource = "builtin"
	ToolSourceMCP     ToolSource = "mcp"
)

// ToolParameter defines a parameter for a tool
type ToolParameter struct {
	Name        string `json:"name"`
	Type        string `json:"type"`
	Description string `json:"description"`
	Required    bool   `json:"required"`
	Default     any    `json:"default,omitempty"`
	Enum        []any  `json:"enum,omitempty"`
}

// Tool represents a callable tool
type Tool struct {
	ID          string                 `json:"id"`
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Source      ToolSource             `json:"source"`
	MCPServer   string                 `json:"mcp_server,omitempty"`
	Parameters  []ToolParameter        `json:"parameters"`
	Enabled     bool                   `json:"enabled"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
}

// ToOpenAIFormat converts the tool to OpenAI function calling format
func (t *Tool) ToOpenAIFormat() map[string]interface{} {
	properties := make(map[string]interface{})
	required := []string{}

	for _, p := range t.Parameters {
		prop := map[string]interface{}{
			"type":        p.Type,
			"description": p.Description,
		}
		if len(p.Enum) > 0 {
			prop["enum"] = p.Enum
		}
		if p.Default != nil {
			prop["default"] = p.Default
		}
		properties[p.Name] = prop
		if p.Required {
			required = append(required, p.Name)
		}
	}

	return map[string]interface{}{
		"type": "function",
		"function": map[string]interface{}{
			"name":        t.Name,
			"description": t.Description,
			"parameters": map[string]interface{}{
				"type":       "object",
				"properties": properties,
				"required":   required,
			},
		},
	}
}

// ToolExecutionRequest represents a request to execute a tool
type ToolExecutionRequest struct {
	ToolID    string                 `json:"tool_id"`
	Arguments map[string]interface{} `json:"arguments"`
}

// ToolExecutionResult represents the result of a tool execution
type ToolExecutionResult struct {
	Success  bool        `json:"success"`
	Result   interface{} `json:"result,omitempty"`
	Error    string      `json:"error,omitempty"`
	Duration int64       `json:"duration_ms"`
}
