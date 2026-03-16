package nlauthor

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

const (
	PlannerToolSourceBuiltinBuilder = "builtin_builder"
)

type ToolPolicy struct {
	AllowedSources     []string
	AllowedToolNames   []string
	MaxCalls           int
	MaxToolResultChars int
}

func DefaultToolPolicy() ToolPolicy {
	return ToolPolicy{
		AllowedSources:     []string{PlannerToolSourceBuiltinBuilder},
		MaxCalls:           4,
		MaxToolResultChars: 4000,
	}
}

type PlannerToolDefinition struct {
	Name        string
	Description string
	InputSchema json.RawMessage
	Readonly    bool
	Source      string
}

type PlannerToolResult struct {
	Content string
}

type PlannerTool interface {
	Definition() PlannerToolDefinition
	Invoke(ctx context.Context, session Session, request TurnRequest, arguments json.RawMessage) (PlannerToolResult, error)
}

type PlannerToolSource interface {
	SourceName() string
	Tools(session Session, request TurnRequest) []PlannerTool
}

type PlannerToolRegistry struct {
	sources []PlannerToolSource
}

func NewPlannerToolRegistry(sources ...PlannerToolSource) PlannerToolRegistry {
	return PlannerToolRegistry{sources: sources}
}

func (r PlannerToolRegistry) List(session Session, request TurnRequest, policy ToolPolicy) []PlannerToolDefinition {
	definitions := make([]PlannerToolDefinition, 0)
	for _, tool := range r.allowedTools(session, request, policy) {
		definitions = append(definitions, tool.Definition())
	}
	return definitions
}

func (r PlannerToolRegistry) Invoke(
	ctx context.Context,
	session Session,
	request TurnRequest,
	name string,
	arguments json.RawMessage,
	policy ToolPolicy,
) (PlannerToolResult, error) {
	for _, tool := range r.allowedTools(session, request, policy) {
		definition := tool.Definition()
		if definition.Name != name {
			continue
		}

		result, err := tool.Invoke(ctx, session, request, arguments)
		if err != nil {
			return PlannerToolResult{}, err
		}
		if policy.MaxToolResultChars > 0 && len(result.Content) > policy.MaxToolResultChars {
			result.Content = result.Content[:policy.MaxToolResultChars] + "\n... [truncated]"
		}
		return result, nil
	}
	return PlannerToolResult{}, fmt.Errorf("planner tool %q is not available", name)
}

func (r PlannerToolRegistry) Definition(
	session Session,
	request TurnRequest,
	name string,
	policy ToolPolicy,
) (PlannerToolDefinition, bool) {
	for _, tool := range r.allowedTools(session, request, policy) {
		definition := tool.Definition()
		if definition.Name == name {
			return definition, true
		}
	}
	return PlannerToolDefinition{}, false
}

func (r PlannerToolRegistry) allowedTools(session Session, request TurnRequest, policy ToolPolicy) []PlannerTool {
	allowedSources := make(map[string]struct{}, len(policy.AllowedSources))
	for _, source := range policy.AllowedSources {
		allowedSources[source] = struct{}{}
	}
	allowedToolNames := make(map[string]struct{}, len(policy.AllowedToolNames))
	for _, name := range policy.AllowedToolNames {
		allowedToolNames[name] = struct{}{}
	}

	tools := make([]PlannerTool, 0)
	for _, source := range r.sources {
		if len(allowedSources) > 0 {
			if _, ok := allowedSources[source.SourceName()]; !ok {
				continue
			}
		}
		for _, tool := range source.Tools(session, request) {
			definition := tool.Definition()
			if len(allowedToolNames) > 0 {
				if _, ok := allowedToolNames[definition.Name]; !ok {
					continue
				}
			}
			tools = append(tools, tool)
		}
	}
	return tools
}

type staticPlannerTool struct {
	definition PlannerToolDefinition
	invokeFn   func(ctx context.Context, session Session, request TurnRequest, arguments json.RawMessage) (PlannerToolResult, error)
}

func (t staticPlannerTool) Definition() PlannerToolDefinition {
	return t.definition
}

func (t staticPlannerTool) Invoke(ctx context.Context, session Session, request TurnRequest, arguments json.RawMessage) (PlannerToolResult, error) {
	return t.invokeFn(ctx, session, request, arguments)
}

func marshalToolJSON(value any) (PlannerToolResult, error) {
	raw, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return PlannerToolResult{}, err
	}
	return PlannerToolResult{Content: string(raw)}, nil
}

func parseOptionalName(arguments json.RawMessage) string {
	if len(arguments) == 0 {
		return ""
	}
	var payload struct {
		Name string `json:"name"`
	}
	if err := json.Unmarshal(arguments, &payload); err != nil {
		return ""
	}
	return strings.TrimSpace(payload.Name)
}
