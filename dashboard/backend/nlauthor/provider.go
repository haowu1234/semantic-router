package nlauthor

import (
	"context"
	"encoding/json"
)

// ProviderMessage is one planner-provider chat turn.
type ProviderMessage struct {
	Role       string             `json:"role"`
	Content    string             `json:"content,omitempty"`
	ToolCallID string             `json:"tool_call_id,omitempty"`
	ToolCalls  []ProviderToolCall `json:"tool_calls,omitempty"`
}

// StructuredOutputSchema describes the strict JSON schema the provider must return.
type StructuredOutputSchema struct {
	Name   string          `json:"name"`
	Schema json.RawMessage `json:"schema"`
	Strict bool            `json:"strict"`
}

// StructuredGenerationRequest is the provider-facing planner generation contract.
type StructuredGenerationRequest struct {
	Model           string
	Messages        []ProviderMessage
	ResponseSchema  StructuredOutputSchema
	MaxOutputTokens int
}

// StructuredGenerationResponse contains the provider's JSON string payload.
type StructuredGenerationResponse struct {
	Content string
}

type ProviderToolDefinition struct {
	Name        string
	Description string
	InputSchema json.RawMessage
}

type ProviderToolCall struct {
	ID        string
	Name      string
	Arguments string
}

func (c ProviderToolCall) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		ID       string `json:"id"`
		Type     string `json:"type"`
		Function struct {
			Name      string `json:"name"`
			Arguments string `json:"arguments"`
		} `json:"function"`
	}{
		ID:   c.ID,
		Type: "function",
		Function: struct {
			Name      string `json:"name"`
			Arguments string `json:"arguments"`
		}{
			Name:      c.Name,
			Arguments: c.Arguments,
		},
	})
}

type ToolCallingRequest struct {
	Model           string
	Messages        []ProviderMessage
	Tools           []ProviderToolDefinition
	ResponseSchema  StructuredOutputSchema
	MaxOutputTokens int
}

type ToolCallingResponse struct {
	Content   string
	ToolCalls []ProviderToolCall
}

// PlannerProvider is the model-provider seam used by model-backed planner implementations.
type PlannerProvider interface {
	Name() string
	Available() bool
	GenerateStructured(ctx context.Context, request StructuredGenerationRequest) (StructuredGenerationResponse, error)
	GenerateToolCalls(ctx context.Context, request ToolCallingRequest) (ToolCallingResponse, error)
}
