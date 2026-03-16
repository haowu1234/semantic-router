package nlauthor

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/vllm-project/semantic-router/dashboard/backend/mcp"
)

const (
	PlannerToolSourceMCP = "mcp"
)

func NewMCPPlannerToolSource(manager *mcp.Manager, allowlistCSV string) PlannerToolSource {
	return mcpPlannerToolSource{
		manager: manager,
		allowed: normalizeCommaSeparatedValues([]string{allowlistCSV}),
	}
}

type mcpPlannerToolSource struct {
	manager *mcp.Manager
	allowed []string
}

func (s mcpPlannerToolSource) SourceName() string {
	return PlannerToolSourceMCP
}

func (s mcpPlannerToolSource) Tools(_ Session, _ TurnRequest) []PlannerTool {
	if s.manager == nil || len(s.allowed) == 0 {
		return nil
	}

	allowed := make(map[string]struct{}, len(s.allowed))
	for _, name := range s.allowed {
		allowed[name] = struct{}{}
	}

	tools := make([]PlannerTool, 0)
	for _, tool := range s.manager.GetAllTools() {
		if _, ok := allowed[tool.Name]; !ok {
			continue
		}
		captured := tool
		tools = append(tools, staticPlannerTool{
			definition: PlannerToolDefinition{
				Name:        captured.Name,
				Description: captured.Description,
				InputSchema: captured.InputSchema,
				Readonly:    true,
				Source:      PlannerToolSourceMCP,
			},
			invokeFn: func(ctx context.Context, _ Session, _ TurnRequest, arguments json.RawMessage) (PlannerToolResult, error) {
				result, err := s.manager.ExecuteTool(ctx, captured.ServerID, captured.Name, arguments)
				if err != nil {
					return PlannerToolResult{}, fmt.Errorf("execute MCP tool %q: %w", captured.Name, err)
				}
				return marshalToolJSON(result)
			},
		})
	}
	return tools
}
