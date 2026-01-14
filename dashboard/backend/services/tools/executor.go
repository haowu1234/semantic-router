package tools

import (
	"context"
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/models"
)

// MCPCaller interface for calling MCP tools
type MCPCaller interface {
	CallTool(ctx context.Context, serverID, toolName string, args map[string]interface{}) (interface{}, error)
}

// Executor executes tools
type Executor struct {
	registry  *Registry
	mcpCaller MCPCaller
}

// NewExecutor creates a new tool executor
func NewExecutor(registry *Registry, mcpCaller MCPCaller) *Executor {
	return &Executor{
		registry:  registry,
		mcpCaller: mcpCaller,
	}
}

// Execute runs a tool with the given request
func (e *Executor) Execute(ctx context.Context, req *models.ToolExecutionRequest) *models.ToolExecutionResult {
	start := time.Now()

	tool, builtinTool, found := e.registry.GetTool(req.ToolID)
	if !found {
		return &models.ToolExecutionResult{
			Success:  false,
			Error:    fmt.Sprintf("tool not found: %s", req.ToolID),
			Duration: time.Since(start).Milliseconds(),
		}
	}

	// Check if tool is enabled
	if !e.registry.IsToolEnabled(req.ToolID) {
		return &models.ToolExecutionResult{
			Success:  false,
			Error:    fmt.Sprintf("tool is disabled: %s", req.ToolID),
			Duration: time.Since(start).Milliseconds(),
		}
	}

	var result interface{}
	var err error

	if builtinTool != nil {
		// Execute built-in tool
		result, err = builtinTool.Execute(ctx, req.Arguments)
	} else if tool != nil && tool.Source == models.ToolSourceMCP {
		// Execute MCP tool
		if e.mcpCaller == nil {
			return &models.ToolExecutionResult{
				Success:  false,
				Error:    "MCP caller not configured",
				Duration: time.Since(start).Milliseconds(),
			}
		}
		// Extract server ID from tool ID (format: "serverID/toolName")
		serverID := ""
		toolName := tool.Name
		for i := 0; i < len(req.ToolID); i++ {
			if req.ToolID[i] == '/' {
				serverID = req.ToolID[:i]
				break
			}
		}
		result, err = e.mcpCaller.CallTool(ctx, serverID, toolName, req.Arguments)
	} else {
		return &models.ToolExecutionResult{
			Success:  false,
			Error:    "unknown tool source",
			Duration: time.Since(start).Milliseconds(),
		}
	}

	if err != nil {
		return &models.ToolExecutionResult{
			Success:  false,
			Error:    err.Error(),
			Duration: time.Since(start).Milliseconds(),
		}
	}

	return &models.ToolExecutionResult{
		Success:  true,
		Result:   result,
		Duration: time.Since(start).Milliseconds(),
	}
}

// ExecuteByName executes a tool by name (for built-in tools)
func (e *Executor) ExecuteByName(ctx context.Context, name string, args map[string]interface{}) *models.ToolExecutionResult {
	return e.Execute(ctx, &models.ToolExecutionRequest{
		ToolID:    name,
		Arguments: args,
	})
}
