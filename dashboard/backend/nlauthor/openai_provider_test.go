package nlauthor

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestOpenAICompatibleProviderGenerateStructured(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			t.Fatalf("path = %q, want /v1/chat/completions", r.URL.Path)
		}
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Fatalf("authorization = %q, want Bearer test-key", got)
		}

		var payload map[string]any
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatalf("decode request error = %v", err)
		}
		if payload["model"] != "gpt-test" {
			t.Fatalf("model = %#v, want gpt-test", payload["model"])
		}
		if payload["temperature"] != float64(0) {
			t.Fatalf("temperature = %#v, want 0", payload["temperature"])
		}

		responseFormat, ok := payload["response_format"].(map[string]any)
		if !ok || responseFormat["type"] != "json_schema" {
			t.Fatalf("response_format = %#v, want json_schema", payload["response_format"])
		}

		_, _ = w.Write([]byte(`{"choices":[{"message":{"content":"{\"status\":\"unsupported\",\"explanation\":\"unsupported for this prompt\",\"error\":\"unsupported prompt\"}"}}]}`))
	}))
	defer server.Close()

	provider := NewOpenAICompatibleProvider(RuntimeConfig{
		BaseURL: server.URL + "/v1",
		APIKey:  "test-key",
		Timeout: 0,
	})

	response, err := provider.GenerateStructured(context.Background(), StructuredGenerationRequest{
		Model: "gpt-test",
		Messages: []ProviderMessage{
			{Role: "system", Content: "You are a planner."},
			{Role: "user", Content: "{}"},
		},
		ResponseSchema: StructuredOutputSchema{
			Name:   "planner_result",
			Schema: plannerResultJSONSchema(),
			Strict: true,
		},
		MaxOutputTokens: 1024,
	})
	if err != nil {
		t.Fatalf("GenerateStructured error = %v", err)
	}

	if response.Content == "" {
		t.Fatal("content is empty")
	}
}

func TestOpenAICompatibleProviderGenerateToolCalls(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload map[string]any
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatalf("decode request error = %v", err)
		}

		messages, ok := payload["messages"].([]any)
		if !ok || len(messages) != 3 {
			t.Fatalf("messages = %#v, want 3 chat messages", payload["messages"])
		}
		assistantMessage, ok := messages[1].(map[string]any)
		if !ok {
			t.Fatalf("assistant message = %#v, want object", messages[1])
		}
		toolCalls, ok := assistantMessage["tool_calls"].([]any)
		if !ok || len(toolCalls) != 1 {
			t.Fatalf("assistant tool_calls = %#v, want 1 tool call", assistantMessage["tool_calls"])
		}
		toolCall, ok := toolCalls[0].(map[string]any)
		if !ok {
			t.Fatalf("tool call = %#v, want object", toolCalls[0])
		}
		if toolCall["id"] != "call_1" {
			t.Fatalf("tool call id = %#v, want call_1", toolCall["id"])
		}
		if toolCall["type"] != "function" {
			t.Fatalf("tool call type = %#v, want function", toolCall["type"])
		}
		function, ok := toolCall["function"].(map[string]any)
		if !ok {
			t.Fatalf("tool call function = %#v, want object", toolCall["function"])
		}
		if function["name"] != "list_symbols" {
			t.Fatalf("tool call function.name = %#v, want list_symbols", function["name"])
		}
		if function["arguments"] != "{}" {
			t.Fatalf("tool call function.arguments = %#v, want {}", function["arguments"])
		}

		_, _ = w.Write([]byte(`{"choices":[{"message":{"tool_calls":[{"id":"call_1","type":"function","function":{"name":"list_symbols","arguments":"{}"}}]}}]}`))
	}))
	defer server.Close()

	provider := NewOpenAICompatibleProvider(RuntimeConfig{
		BaseURL: server.URL + "/v1",
		Timeout: 0,
	})

	response, err := provider.GenerateToolCalls(context.Background(), ToolCallingRequest{
		Model: "gpt-test",
		Messages: []ProviderMessage{
			{Role: "system", Content: "You are a planner."},
			{Role: "assistant", ToolCalls: []ProviderToolCall{{ID: "call_1", Name: "list_symbols", Arguments: `{}`}}},
			{Role: "tool", ToolCallID: "call_1", Content: `{"routes":["support_route"]}`},
		},
		Tools: []ProviderToolDefinition{
			{Name: "list_symbols", Description: "List symbols", InputSchema: json.RawMessage(`{"type":"object","properties":{}}`)},
		},
		ResponseSchema: StructuredOutputSchema{
			Name:   "planner_result",
			Schema: plannerResultJSONSchema(),
			Strict: true,
		},
		MaxOutputTokens: 1024,
	})
	if err != nil {
		t.Fatalf("GenerateToolCalls error = %v", err)
	}
	if len(response.ToolCalls) != 1 || response.ToolCalls[0].Name != "list_symbols" {
		t.Fatalf("toolCalls = %+v, want list_symbols", response.ToolCalls)
	}
}
