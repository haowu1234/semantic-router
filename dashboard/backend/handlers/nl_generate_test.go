package handlers

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// ─────────────────────────────────────────────
// Test Helpers
// ─────────────────────────────────────────────

func newNLConfig(endpoint, apiKey, model string) *NLConfig {
	return &NLConfig{
		DefaultEndpoint: endpoint,
		DefaultAPIKey:   apiKey,
		DefaultModel:    model,
	}
}

func doNLRequest(handler http.HandlerFunc, method, body string) *httptest.ResponseRecorder {
	var reader io.Reader
	if body != "" {
		reader = strings.NewReader(body)
	}
	req := httptest.NewRequest(method, "/api/nl/generate", reader)
	if body != "" {
		req.Header.Set("Content-Type", "application/json")
	}
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)
	return w
}

// ─────────────────────────────────────────────
// NLGenerateHandler Tests
// ─────────────────────────────────────────────

func TestNLGenerate_MethodNotAllowed(t *testing.T) {
	cfg := newNLConfig("http://localhost:8080", "", "")
	handler := NLGenerateHandler(cfg)

	for _, method := range []string{"GET", "PUT", "DELETE", "PATCH"} {
		w := doNLRequest(handler, method, "")
		if w.Code != http.StatusMethodNotAllowed {
			t.Errorf("%s: expected 405, got %d", method, w.Code)
		}
	}
}

func TestNLGenerate_InvalidJSON(t *testing.T) {
	cfg := newNLConfig("http://localhost:8080", "", "")
	handler := NLGenerateHandler(cfg)

	w := doNLRequest(handler, "POST", "not json")
	if w.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", w.Code)
	}

	var resp NLProxyErrorResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode error: %v", err)
	}
	if resp.Code != "INVALID_JSON" {
		t.Errorf("expected code INVALID_JSON, got %s", resp.Code)
	}
}

func TestNLGenerate_MissingMessages(t *testing.T) {
	cfg := newNLConfig("http://localhost:8080", "", "")
	handler := NLGenerateHandler(cfg)

	w := doNLRequest(handler, "POST", `{"messages":[]}`)
	if w.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", w.Code)
	}

	var resp NLProxyErrorResponse
	_ = json.NewDecoder(w.Body).Decode(&resp)
	if resp.Code != "MISSING_MESSAGES" {
		t.Errorf("expected code MISSING_MESSAGES, got %s", resp.Code)
	}
}

func TestNLGenerate_NoEndpoint(t *testing.T) {
	cfg := newNLConfig("", "", "")
	handler := NLGenerateHandler(cfg)

	w := doNLRequest(handler, "POST", `{"messages":[{"role":"user","content":"test"}]}`)
	if w.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", w.Code)
	}

	var resp NLProxyErrorResponse
	_ = json.NewDecoder(w.Body).Decode(&resp)
	if resp.Code != "NO_ENDPOINT" {
		t.Errorf("expected code NO_ENDPOINT, got %s", resp.Code)
	}
}

func TestNLGenerate_InvalidEndpoint(t *testing.T) {
	cfg := newNLConfig("ftp://invalid", "", "")
	handler := NLGenerateHandler(cfg)

	w := doNLRequest(handler, "POST", `{"messages":[{"role":"user","content":"test"}]}`)
	if w.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", w.Code)
	}

	var resp NLProxyErrorResponse
	_ = json.NewDecoder(w.Body).Decode(&resp)
	if resp.Code != "INVALID_ENDPOINT" {
		t.Errorf("expected code INVALID_ENDPOINT, got %s", resp.Code)
	}
}

func TestNLGenerate_ProxiesToUpstream(t *testing.T) {
	// Create a mock LLM server
	mockLLM := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify request
		if r.Method != "POST" {
			t.Errorf("expected POST, got %s", r.Method)
		}
		if r.Header.Get("Authorization") != "Bearer test-key" {
			t.Errorf("expected Bearer test-key, got %s", r.Header.Get("Authorization"))
		}

		body, _ := io.ReadAll(r.Body)
		var req map[string]interface{}
		_ = json.Unmarshal(body, &req)

		if req["model"] != "test-model" {
			t.Errorf("expected model test-model, got %v", req["model"])
		}

		// Return mock OpenAI response
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"choices": []map[string]interface{}{
				{
					"message": map[string]interface{}{
						"role":    "assistant",
						"content": `{"intents":[]}`,
					},
				},
			},
		})
	}))
	defer mockLLM.Close()

	cfg := newNLConfig(mockLLM.URL, "test-key", "test-model")
	handler := NLGenerateHandler(cfg)

	w := doNLRequest(handler, "POST", `{
		"messages": [
			{"role": "system", "content": "You are a DSL generator."},
			{"role": "user", "content": "Create a routing config for math"}
		],
		"temperature": 0.1,
		"response_format": {"type": "json_object"}
	}`)

	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	// Verify response is the mock LLM response
	var resp map[string]interface{}
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	choices, ok := resp["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		t.Error("expected choices in response")
	}
}

func TestNLGenerate_RequestEndpointOverride(t *testing.T) {
	// Mock LLM server
	mockLLM := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"choices": []map[string]interface{}{
				{"message": map[string]interface{}{"role": "assistant", "content": "ok"}},
			},
		})
	}))
	defer mockLLM.Close()

	// Server has no default endpoint — frontend provides it
	cfg := newNLConfig("", "", "")
	handler := NLGenerateHandler(cfg)

	body := fmt.Sprintf(`{
		"messages": [{"role": "user", "content": "test"}],
		"endpoint": %q
	}`, mockLLM.URL)

	w := doNLRequest(handler, "POST", body)
	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
}

func TestNLGenerate_ServerKeyPrecedence(t *testing.T) {
	receivedAuth := ""
	mockLLM := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"choices": []map[string]interface{}{
				{"message": map[string]interface{}{"role": "assistant", "content": "ok"}},
			},
		})
	}))
	defer mockLLM.Close()

	// Server key should take precedence over frontend key
	cfg := newNLConfig(mockLLM.URL, "server-key", "")
	handler := NLGenerateHandler(cfg)

	w := doNLRequest(handler, "POST", `{
		"messages": [{"role": "user", "content": "test"}],
		"api_key": "frontend-key"
	}`)

	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}
	if receivedAuth != "Bearer server-key" {
		t.Errorf("expected server key to take precedence, got %s", receivedAuth)
	}
}

func TestNLGenerate_FrontendKeyFallback(t *testing.T) {
	receivedAuth := ""
	mockLLM := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedAuth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"choices": []map[string]interface{}{
				{"message": map[string]interface{}{"role": "assistant", "content": "ok"}},
			},
		})
	}))
	defer mockLLM.Close()

	// No server key — frontend key should be used
	cfg := newNLConfig(mockLLM.URL, "", "")
	handler := NLGenerateHandler(cfg)

	w := doNLRequest(handler, "POST", `{
		"messages": [{"role": "user", "content": "test"}],
		"api_key": "frontend-key"
	}`)

	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}
	if receivedAuth != "Bearer frontend-key" {
		t.Errorf("expected frontend key, got %s", receivedAuth)
	}
}

func TestNLGenerate_UpstreamError(t *testing.T) {
	mockLLM := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"error": map[string]interface{}{
				"message": "Rate limit exceeded",
				"type":    "rate_limit_error",
			},
		})
	}))
	defer mockLLM.Close()

	cfg := newNLConfig(mockLLM.URL, "key", "model")
	handler := NLGenerateHandler(cfg)

	w := doNLRequest(handler, "POST", `{"messages":[{"role":"user","content":"test"}]}`)

	// Should forward the upstream status code
	if w.Code != http.StatusTooManyRequests {
		t.Errorf("expected 429, got %d", w.Code)
	}
}

func TestNLGenerate_StreamResponse(t *testing.T) {
	// Mock streaming LLM server
	mockLLM := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]interface{}
		_ = json.Unmarshal(body, &req)

		if req["stream"] != true {
			t.Error("expected stream=true in upstream request")
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		flusher := w.(http.Flusher)

		// Send a few SSE chunks
		chunks := []string{
			`data: {"choices":[{"delta":{"content":"hel"}}]}`,
			`data: {"choices":[{"delta":{"content":"lo"}}]}`,
			`data: [DONE]`,
		}
		for _, chunk := range chunks {
			fmt.Fprintf(w, "%s\n\n", chunk)
			flusher.Flush()
		}
	}))
	defer mockLLM.Close()

	cfg := newNLConfig(mockLLM.URL, "key", "model")
	handler := NLGenerateHandler(cfg)

	w := doNLRequest(handler, "POST", `{
		"messages":[{"role":"user","content":"test"}],
		"stream": true
	}`)

	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}
	if ct := w.Header().Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("expected text/event-stream, got %s", ct)
	}
	if !strings.Contains(w.Body.String(), "[DONE]") {
		t.Error("expected [DONE] in streamed response")
	}
}

func TestNLGenerate_DefaultModel(t *testing.T) {
	receivedModel := ""
	mockLLM := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]interface{}
		_ = json.Unmarshal(body, &req)
		receivedModel = req["model"].(string)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"choices": []map[string]interface{}{
				{"message": map[string]interface{}{"role": "assistant", "content": "ok"}},
			},
		})
	}))
	defer mockLLM.Close()

	// No model in config or request → default "qwen3-32b"
	cfg := newNLConfig(mockLLM.URL, "", "")
	handler := NLGenerateHandler(cfg)

	w := doNLRequest(handler, "POST", `{"messages":[{"role":"user","content":"test"}]}`)
	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}
	if receivedModel != "qwen3-32b" {
		t.Errorf("expected default model qwen3-32b, got %s", receivedModel)
	}
}

func TestNLGenerate_RequestModelOverride(t *testing.T) {
	receivedModel := ""
	mockLLM := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]interface{}
		_ = json.Unmarshal(body, &req)
		receivedModel = req["model"].(string)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"choices": []map[string]interface{}{
				{"message": map[string]interface{}{"role": "assistant", "content": "ok"}},
			},
		})
	}))
	defer mockLLM.Close()

	cfg := newNLConfig(mockLLM.URL, "", "server-model")
	handler := NLGenerateHandler(cfg)

	// Request model should override server default
	w := doNLRequest(handler, "POST", `{
		"messages":[{"role":"user","content":"test"}],
		"model": "gpt-4o"
	}`)
	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}
	if receivedModel != "gpt-4o" {
		t.Errorf("expected request model gpt-4o, got %s", receivedModel)
	}
}

// ─────────────────────────────────────────────
// NLConfigHandler Tests
// ─────────────────────────────────────────────

func TestNLConfig_ReturnsConfig(t *testing.T) {
	cfg := newNLConfig("https://api.openai.com/v1/chat/completions", "sk-test", "gpt-4o")
	handler := NLConfigHandler(cfg)

	req := httptest.NewRequest("GET", "/api/nl/config", nil)
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}

	var resp NLConfigResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode: %v", err)
	}

	if !resp.HasServerKey {
		t.Error("expected HasServerKey=true")
	}
	if resp.DefaultModel != "gpt-4o" {
		t.Errorf("expected model gpt-4o, got %s", resp.DefaultModel)
	}
	// Endpoint should be masked
	if strings.Contains(resp.DefaultEndpoint, "chat/completions") {
		t.Error("endpoint should be masked")
	}
	if !strings.Contains(resp.DefaultEndpoint, "api.openai.com") {
		t.Error("masked endpoint should contain host")
	}
}

func TestNLConfig_NoServerKey(t *testing.T) {
	cfg := newNLConfig("", "", "")
	handler := NLConfigHandler(cfg)

	req := httptest.NewRequest("GET", "/api/nl/config", nil)
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	var resp NLConfigResponse
	_ = json.NewDecoder(w.Body).Decode(&resp)

	if resp.HasServerKey {
		t.Error("expected HasServerKey=false")
	}
}

func TestNLConfig_MethodNotAllowed(t *testing.T) {
	cfg := newNLConfig("", "", "")
	handler := NLConfigHandler(cfg)

	req := httptest.NewRequest("POST", "/api/nl/config", nil)
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("expected 405, got %d", w.Code)
	}
}

// ─────────────────────────────────────────────
// NLExplainHandler Tests
// ─────────────────────────────────────────────

func TestNLExplain_MissingDSL(t *testing.T) {
	cfg := newNLConfig("http://localhost:8080", "key", "model")
	handler := NLExplainHandler(cfg)

	req := httptest.NewRequest("POST", "/api/nl/explain", strings.NewReader(`{"dsl":""}`))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", w.Code)
	}

	var resp NLProxyErrorResponse
	_ = json.NewDecoder(w.Body).Decode(&resp)
	if resp.Code != "MISSING_DSL" {
		t.Errorf("expected code MISSING_DSL, got %s", resp.Code)
	}
}

func TestNLExplain_ProxiesToUpstream(t *testing.T) {
	mockLLM := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req map[string]interface{}
		_ = json.Unmarshal(body, &req)

		// Verify it built the explain prompt
		messages := req["messages"].([]interface{})
		if len(messages) != 2 {
			t.Errorf("expected 2 messages, got %d", len(messages))
		}
		userMsg := messages[1].(map[string]interface{})
		if !strings.Contains(userMsg["content"].(string), "SIGNAL domain math") {
			t.Error("expected DSL in user message")
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"choices": []map[string]interface{}{
				{
					"message": map[string]interface{}{
						"role":    "assistant",
						"content": `{"summary":"Routes math queries","entities":[]}`,
					},
				},
			},
		})
	}))
	defer mockLLM.Close()

	cfg := newNLConfig(mockLLM.URL, "key", "model")
	handler := NLExplainHandler(cfg)

	req := httptest.NewRequest("POST", "/api/nl/explain",
		strings.NewReader(`{"dsl":"SIGNAL domain math {\n  patterns: [\"math\"]\n}"}`))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d: %s", w.Code, w.Body.String())
	}
}

func TestNLExplain_NoEndpoint(t *testing.T) {
	cfg := newNLConfig("", "", "")
	handler := NLExplainHandler(cfg)

	req := httptest.NewRequest("POST", "/api/nl/explain",
		strings.NewReader(`{"dsl":"SIGNAL domain math {}"}`))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", w.Code)
	}
}

// ─────────────────────────────────────────────
// Helper Tests
// ─────────────────────────────────────────────

func TestMaskEndpoint(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"", ""},
		{"https://api.openai.com/v1/chat/completions", "https://api.openai.com/..."},
		{"http://localhost:8080/v1/chat/completions", "http://localhost:8080/..."},
	}

	for _, tt := range tests {
		result := maskEndpoint(tt.input)
		if result != tt.expected {
			t.Errorf("maskEndpoint(%q) = %q, want %q", tt.input, result, tt.expected)
		}
	}
}

func TestBuildUpstreamRequest(t *testing.T) {
	temp := 0.1
	maxTokens := 4096
	req := NLProxyRequest{
		Messages: []NLMessage{
			{Role: "user", Content: "test"},
		},
		Temperature:    &temp,
		MaxTokens:      &maxTokens,
		ResponseFormat: map[string]interface{}{"type": "json_object"},
		Stream:         true,
	}

	result := buildUpstreamRequest(req, "gpt-4o")

	if result["model"] != "gpt-4o" {
		t.Errorf("expected model gpt-4o, got %v", result["model"])
	}
	if result["stream"] != true {
		t.Error("expected stream=true")
	}
	if result["temperature"] != 0.1 {
		t.Errorf("expected temperature 0.1, got %v", result["temperature"])
	}
	if result["max_tokens"] != 4096 {
		t.Errorf("expected max_tokens 4096, got %v", result["max_tokens"])
	}
	if result["response_format"] == nil {
		t.Error("expected response_format to be set")
	}
}

func TestBuildUpstreamRequest_MinimalFields(t *testing.T) {
	req := NLProxyRequest{
		Messages: []NLMessage{{Role: "user", Content: "hi"}},
	}

	result := buildUpstreamRequest(req, "model")

	if _, ok := result["temperature"]; ok {
		t.Error("temperature should not be set when nil")
	}
	if _, ok := result["max_tokens"]; ok {
		t.Error("max_tokens should not be set when nil")
	}
	if _, ok := result["response_format"]; ok {
		t.Error("response_format should not be set when nil")
	}
}
