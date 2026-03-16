package nlauthor

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
)

const providerResponseLogCharBudget = 2000

type openAICompatibleProvider struct {
	baseURL string
	apiKey  string
	client  *http.Client
}

type openAICompatibleRequest struct {
	Model          string                             `json:"model"`
	Messages       []ProviderMessage                  `json:"messages"`
	Temperature    float64                            `json:"temperature"`
	MaxTokens      int                                `json:"max_tokens,omitempty"`
	ResponseFormat openAICompatibleResponseFormatSpec `json:"response_format"`
	Tools          []openAICompatibleToolSpec         `json:"tools,omitempty"`
	ToolChoice     string                             `json:"tool_choice,omitempty"`
}

type openAICompatibleResponseFormatSpec struct {
	Type       string                               `json:"type"`
	JSONSchema openAICompatibleResponseFormatSchema `json:"json_schema"`
}

type openAICompatibleResponseFormatSchema struct {
	Name   string          `json:"name"`
	Schema json.RawMessage `json:"schema"`
	Strict bool            `json:"strict"`
}

type openAICompatibleResponse struct {
	Choices []struct {
		Message struct {
			Content   string                     `json:"content"`
			ToolCalls []openAICompatibleToolCall `json:"tool_calls,omitempty"`
		} `json:"message"`
	} `json:"choices"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

type openAICompatibleToolSpec struct {
	Type     string                           `json:"type"`
	Function openAICompatibleToolFunctionSpec `json:"function"`
}

type openAICompatibleToolFunctionSpec struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters"`
}

type openAICompatibleToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

func NewOpenAICompatibleProvider(config RuntimeConfig) PlannerProvider {
	normalized := NormalizeRuntimeConfig(config)
	return openAICompatibleProvider{
		baseURL: normalized.BaseURL,
		apiKey:  normalized.APIKey,
		client: &http.Client{
			Timeout: normalized.Timeout,
		},
	}
}

func (p openAICompatibleProvider) Name() string {
	return string(PlannerProviderOpenAICompatible)
}

func (p openAICompatibleProvider) Available() bool {
	return strings.TrimSpace(p.baseURL) != ""
}

func (p openAICompatibleProvider) GenerateStructured(ctx context.Context, request StructuredGenerationRequest) (StructuredGenerationResponse, error) {
	response, err := p.doChatCompletion(ctx, openAICompatibleRequest{
		Model:       strings.TrimSpace(request.Model),
		Messages:    request.Messages,
		Temperature: 0,
		ResponseFormat: openAICompatibleResponseFormatSpec{
			Type: "json_schema",
			JSONSchema: openAICompatibleResponseFormatSchema{
				Name:   request.ResponseSchema.Name,
				Schema: request.ResponseSchema.Schema,
				Strict: request.ResponseSchema.Strict,
			},
		},
		MaxTokens: request.MaxOutputTokens,
	})
	if err != nil {
		return StructuredGenerationResponse{}, err
	}
	if len(response.ToolCalls) > 0 {
		return StructuredGenerationResponse{}, fmt.Errorf("provider returned tool calls for structured generation")
	}
	return StructuredGenerationResponse{Content: response.Content}, nil
}

func (p openAICompatibleProvider) GenerateToolCalls(ctx context.Context, request ToolCallingRequest) (ToolCallingResponse, error) {
	toolSpecs := make([]openAICompatibleToolSpec, 0, len(request.Tools))
	for _, tool := range request.Tools {
		toolSpecs = append(toolSpecs, openAICompatibleToolSpec{
			Type: "function",
			Function: openAICompatibleToolFunctionSpec{
				Name:        tool.Name,
				Description: tool.Description,
				Parameters:  tool.InputSchema,
			},
		})
	}

	return p.doChatCompletion(ctx, openAICompatibleRequest{
		Model:       strings.TrimSpace(request.Model),
		Messages:    request.Messages,
		Temperature: 0,
		ResponseFormat: openAICompatibleResponseFormatSpec{
			Type: "json_schema",
			JSONSchema: openAICompatibleResponseFormatSchema{
				Name:   request.ResponseSchema.Name,
				Schema: request.ResponseSchema.Schema,
				Strict: request.ResponseSchema.Strict,
			},
		},
		Tools:      toolSpecs,
		ToolChoice: "auto",
		MaxTokens:  request.MaxOutputTokens,
	})
}

func (p openAICompatibleProvider) doChatCompletion(ctx context.Context, payload openAICompatibleRequest) (ToolCallingResponse, error) {
	if !p.Available() {
		return ToolCallingResponse{}, fmt.Errorf("openai-compatible provider base URL is not configured")
	}
	if payload.Model == "" {
		return ToolCallingResponse{}, fmt.Errorf("structured planner model is required")
	}

	raw, err := json.Marshal(payload)
	if err != nil {
		return ToolCallingResponse{}, fmt.Errorf("marshal provider request: %w", err)
	}

	httpRequest, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		strings.TrimRight(p.baseURL, "/")+"/chat/completions",
		bytes.NewReader(raw),
	)
	if err != nil {
		return ToolCallingResponse{}, fmt.Errorf("build provider request: %w", err)
	}
	httpRequest.Header.Set("Content-Type", "application/json")
	httpRequest.Header.Set("Accept", "application/json")
	if p.apiKey != "" {
		httpRequest.Header.Set("Authorization", "Bearer "+p.apiKey)
	}

	response, err := p.client.Do(httpRequest)
	if err != nil {
		return ToolCallingResponse{}, fmt.Errorf("execute provider request: %w", err)
	}
	defer response.Body.Close()

	body, err := io.ReadAll(response.Body)
	if err != nil {
		return ToolCallingResponse{}, fmt.Errorf("read provider response: %w", err)
	}
	if response.StatusCode < http.StatusOK || response.StatusCode >= http.StatusMultipleChoices {
		logProviderResponseIssue(payload.Model, response.StatusCode, "non_2xx_status", body)
		return ToolCallingResponse{}, fmt.Errorf("provider returned %d: %s", response.StatusCode, strings.TrimSpace(string(body)))
	}

	var parsed openAICompatibleResponse
	if err := json.Unmarshal(body, &parsed); err != nil {
		logProviderResponseIssue(payload.Model, response.StatusCode, "decode_error", body)
		return ToolCallingResponse{}, fmt.Errorf("decode provider response: %w", err)
	}
	if parsed.Error != nil && strings.TrimSpace(parsed.Error.Message) != "" {
		logProviderResponseIssue(payload.Model, response.StatusCode, "provider_error_field", body)
		return ToolCallingResponse{}, fmt.Errorf("%s", parsed.Error.Message)
	}
	if len(parsed.Choices) == 0 {
		logProviderResponseIssue(payload.Model, response.StatusCode, "no_choices", body)
		return ToolCallingResponse{}, fmt.Errorf("provider returned no choices")
	}

	if len(parsed.Choices[0].Message.ToolCalls) > 0 {
		toolCalls := make([]ProviderToolCall, 0, len(parsed.Choices[0].Message.ToolCalls))
		for _, toolCall := range parsed.Choices[0].Message.ToolCalls {
			toolCalls = append(toolCalls, ProviderToolCall{
				ID:        toolCall.ID,
				Name:      toolCall.Function.Name,
				Arguments: toolCall.Function.Arguments,
			})
		}
		return ToolCallingResponse{ToolCalls: toolCalls}, nil
	}

	content := strings.TrimSpace(parsed.Choices[0].Message.Content)
	if content == "" {
		logProviderResponseIssue(payload.Model, response.StatusCode, "empty_content", body)
		return ToolCallingResponse{}, fmt.Errorf("provider returned empty content")
	}
	return ToolCallingResponse{Content: content}, nil
}

func logProviderResponseIssue(model string, statusCode int, reason string, body []byte) {
	excerpt := strings.TrimSpace(string(body))
	if len(excerpt) > providerResponseLogCharBudget {
		excerpt = excerpt[:providerResponseLogCharBudget] + "... [truncated]"
	}
	log.Printf(
		"[NL-Planner-Provider] provider=%s model=%s status_code=%d reason=%s body=%q",
		PlannerProviderOpenAICompatible,
		fallbackPlannerObservationValue(model, "unconfigured"),
		statusCode,
		fallbackPlannerObservationValue(reason, "unknown"),
		excerpt,
	)
}
