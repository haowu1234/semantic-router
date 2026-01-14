package tools

import (
	"context"

	"github.com/vllm-project/semantic-router/dashboard/backend/models"
)

// BuiltinTool defines the interface for built-in tools
type BuiltinTool interface {
	// Name returns the tool name
	Name() string

	// Description returns the tool description
	Description() string

	// Parameters returns the tool parameters
	Parameters() []models.ToolParameter

	// Execute runs the tool with the given arguments
	Execute(ctx context.Context, args map[string]interface{}) (interface{}, error)
}
