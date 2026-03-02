package handlers

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/middleware"
)

// ─────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────

const (
	nlProxyTimeout    = 120 * time.Second // LLM generation can be slow
	nlMaxRequestSize  = 512 * 1024        // 512 KB max request body
	nlMaxResponseSize = 2 * 1024 * 1024   // 2 MB max response body
)

// NLConfig holds NL proxy configuration resolved from environment variables.
type NLConfig struct {
	// DefaultEndpoint is the LLM API endpoint (e.g., "https://api.openai.com/v1/chat/completions").
	// If empty, the frontend must provide its own endpoint.
	DefaultEndpoint string
	// DefaultAPIKey is the LLM API key. If set, it's used server-side so the
	// frontend doesn't need to send keys over the wire.
	DefaultAPIKey string
	// DefaultModel is the default model name (e.g., "qwen3-32b").
	DefaultModel string
}

// LoadNLConfig reads NL configuration from environment variables.
func LoadNLConfig() *NLConfig {
	return &NLConfig{
		DefaultEndpoint: os.Getenv("NL_LLM_ENDPOINT"),
		DefaultAPIKey:   os.Getenv("NL_LLM_API_KEY"),
		DefaultModel:    os.Getenv("NL_LLM_MODEL"),
	}
}

// ─────────────────────────────────────────────
// Request / Response Types
// ─────────────────────────────────────────────

// NLProxyRequest is the request body from the frontend.
// It is an OpenAI-compatible chat completion request, optionally with
// overrides for endpoint/model.
type NLProxyRequest struct {
	// OpenAI-compatible fields
	Model          string                 `json:"model,omitempty"`
	Messages       []NLMessage            `json:"messages"`
	Temperature    *float64               `json:"temperature,omitempty"`
	MaxTokens      *int                   `json:"max_tokens,omitempty"`
	ResponseFormat map[string]interface{} `json:"response_format,omitempty"`
	Stream         bool                   `json:"stream,omitempty"`

	// Extension fields for NL proxy
	Endpoint string `json:"endpoint,omitempty"` // Override LLM endpoint
	APIKey   string `json:"api_key,omitempty"`  // Override API key (frontend-provided)
}

// NLMessage is a chat message in the OpenAI format.
type NLMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// NLProxyErrorResponse is returned on error.
type NLProxyErrorResponse struct {
	Error   string `json:"error"`
	Code    string `json:"code,omitempty"`
	Details string `json:"details,omitempty"`
}

// NLConfigResponse is returned by GET /api/nl/config.
type NLConfigResponse struct {
	HasServerKey    bool   `json:"has_server_key"`
	DefaultModel    string `json:"default_model,omitempty"`
	DefaultEndpoint string `json:"default_endpoint,omitempty"`
}

// ─────────────────────────────────────────────
// Handler
// ─────────────────────────────────────────────

// NLGenerateHandler returns an HTTP handler that proxies NL→DSL LLM requests.
//
// POST /api/nl/generate — Proxy chat completion to LLM API (supports streaming)
// GET  /api/nl/config   — Return NL configuration (whether server key is set, etc.)
//
// The handler acts as a secure LLM proxy:
//  1. Hides the actual LLM API key from the frontend
//  2. Validates and sanitises requests
//  3. Supports both streaming (SSE) and non-streaming responses
//  4. Adds CORS headers
func NLGenerateHandler(nlCfg *NLConfig) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}

		if r.Method != http.MethodPost {
			writeNLError(w, http.StatusMethodNotAllowed, "METHOD_NOT_ALLOWED", "Use POST")
			return
		}

		// Read and limit request body
		body, err := io.ReadAll(io.LimitReader(r.Body, nlMaxRequestSize))
		if err != nil {
			writeNLError(w, http.StatusBadRequest, "READ_ERROR", fmt.Sprintf("Failed to read request: %v", err))
			return
		}
		defer r.Body.Close()

		var req NLProxyRequest
		if err := json.Unmarshal(body, &req); err != nil {
			writeNLError(w, http.StatusBadRequest, "INVALID_JSON", fmt.Sprintf("Invalid JSON: %v", err))
			return
		}

		// Validate request
		if len(req.Messages) == 0 {
			writeNLError(w, http.StatusBadRequest, "MISSING_MESSAGES", "messages field is required and must not be empty")
			return
		}

		// Resolve endpoint: request override → server config → error
		endpoint := nlCfg.DefaultEndpoint
		if req.Endpoint != "" {
			endpoint = req.Endpoint
		}
		if endpoint == "" {
			writeNLError(w, http.StatusBadRequest, "NO_ENDPOINT",
				"No LLM endpoint configured. Set NL_LLM_ENDPOINT on the server or provide 'endpoint' in the request.")
			return
		}

		// Validate endpoint URL
		parsedURL, err := url.Parse(endpoint)
		if err != nil || (parsedURL.Scheme != "http" && parsedURL.Scheme != "https") {
			writeNLError(w, http.StatusBadRequest, "INVALID_ENDPOINT", "Endpoint must be a valid http/https URL")
			return
		}

		// Resolve API key: server config → request override
		// Server key takes precedence (more secure — frontend doesn't need to know)
		apiKey := nlCfg.DefaultAPIKey
		if apiKey == "" && req.APIKey != "" {
			apiKey = req.APIKey
		}

		// Resolve model: request → server config → default
		model := req.Model
		if model == "" {
			model = nlCfg.DefaultModel
		}
		if model == "" {
			model = "qwen3-32b"
		}

		// Build the upstream OpenAI-compatible request
		upstreamReq := buildUpstreamRequest(req, model)
		upstreamBody, err := json.Marshal(upstreamReq)
		if err != nil {
			writeNLError(w, http.StatusInternalServerError, "MARSHAL_ERROR", "Failed to build upstream request")
			return
		}

		log.Printf("[NL] Proxying to %s (model=%s, stream=%v, messages=%d)",
			endpoint, model, req.Stream, len(req.Messages))

		// Create upstream HTTP request
		httpReq, err := http.NewRequestWithContext(r.Context(), http.MethodPost, endpoint, bytes.NewReader(upstreamBody))
		if err != nil {
			writeNLError(w, http.StatusInternalServerError, "REQUEST_ERROR", fmt.Sprintf("Failed to create request: %v", err))
			return
		}

		httpReq.Header.Set("Content-Type", "application/json")
		if apiKey != "" {
			httpReq.Header.Set("Authorization", "Bearer "+apiKey)
		}
		// Forward Accept header from client
		if accept := r.Header.Get("Accept"); accept != "" {
			httpReq.Header.Set("Accept", accept)
		}

		// Execute upstream request
		client := &http.Client{Timeout: nlProxyTimeout}
		resp, err := client.Do(httpReq)
		if err != nil {
			log.Printf("[NL] Upstream error: %v", err)
			writeNLError(w, http.StatusBadGateway, "UPSTREAM_ERROR", fmt.Sprintf("LLM API request failed: %v", err))
			return
		}
		defer resp.Body.Close()

		// Stream or buffer the response
		if req.Stream {
			proxyStreamResponse(w, resp)
		} else {
			proxyBufferedResponse(w, resp)
		}
	}
}

// NLConfigHandler returns an HTTP handler for GET /api/nl/config.
// Tells the frontend whether a server-side LLM key is configured.
func NLConfigHandler(nlCfg *NLConfig) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}

		if r.Method != http.MethodGet {
			writeNLError(w, http.StatusMethodNotAllowed, "METHOD_NOT_ALLOWED", "Use GET")
			return
		}

		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(NLConfigResponse{
			HasServerKey:    nlCfg.DefaultAPIKey != "",
			DefaultModel:    nlCfg.DefaultModel,
			DefaultEndpoint: maskEndpoint(nlCfg.DefaultEndpoint),
		})
	}
}

// NLExplainHandler returns an HTTP handler for POST /api/nl/explain.
// Proxies an "explain this DSL" request to the LLM.
func NLExplainHandler(nlCfg *NLConfig) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}

		if r.Method != http.MethodPost {
			writeNLError(w, http.StatusMethodNotAllowed, "METHOD_NOT_ALLOWED", "Use POST")
			return
		}

		// Parse request
		var req struct {
			DSL   string `json:"dsl"`
			Model string `json:"model,omitempty"`
		}
		if err := json.NewDecoder(io.LimitReader(r.Body, nlMaxRequestSize)).Decode(&req); err != nil {
			writeNLError(w, http.StatusBadRequest, "INVALID_JSON", fmt.Sprintf("Invalid JSON: %v", err))
			return
		}

		if strings.TrimSpace(req.DSL) == "" {
			writeNLError(w, http.StatusBadRequest, "MISSING_DSL", "dsl field is required")
			return
		}

		// Resolve endpoint and key
		endpoint := nlCfg.DefaultEndpoint
		if endpoint == "" {
			writeNLError(w, http.StatusBadRequest, "NO_ENDPOINT", "No LLM endpoint configured")
			return
		}
		apiKey := nlCfg.DefaultAPIKey
		model := req.Model
		if model == "" {
			model = nlCfg.DefaultModel
		}
		if model == "" {
			model = "qwen3-32b"
		}

		// Build explain prompt
		messages := []NLMessage{
			{
				Role: "system",
				Content: "You are an expert in the Signal DSL routing language. " +
					"Explain the following DSL configuration in clear, concise natural language. " +
					"Describe what each signal, route, plugin, backend, and algorithm does. " +
					"Format your response as a JSON object with two fields: " +
					"\"summary\" (one-paragraph overview) and " +
					"\"entities\" (array of {\"construct\", \"name\", \"description\"}).",
			},
			{
				Role:    "user",
				Content: "Explain this DSL configuration:\n\n```\n" + req.DSL + "\n```",
			},
		}

		upstreamBody, _ := json.Marshal(map[string]interface{}{
			"model":           model,
			"messages":        messages,
			"temperature":     0.2,
			"response_format": map[string]string{"type": "json_object"},
		})

		log.Printf("[NL/explain] Proxying to %s (model=%s, dsl_len=%d)", endpoint, model, len(req.DSL))

		httpReq, err := http.NewRequestWithContext(r.Context(), http.MethodPost, endpoint, bytes.NewReader(upstreamBody))
		if err != nil {
			writeNLError(w, http.StatusInternalServerError, "REQUEST_ERROR", err.Error())
			return
		}
		httpReq.Header.Set("Content-Type", "application/json")
		if apiKey != "" {
			httpReq.Header.Set("Authorization", "Bearer "+apiKey)
		}

		client := &http.Client{Timeout: nlProxyTimeout}
		resp, err := client.Do(httpReq)
		if err != nil {
			writeNLError(w, http.StatusBadGateway, "UPSTREAM_ERROR", err.Error())
			return
		}
		defer resp.Body.Close()

		proxyBufferedResponse(w, resp)
	}
}

// ─────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────

// buildUpstreamRequest builds the OpenAI-compatible request to send upstream.
func buildUpstreamRequest(req NLProxyRequest, model string) map[string]interface{} {
	upstream := map[string]interface{}{
		"model":    model,
		"messages": req.Messages,
		"stream":   req.Stream,
	}

	if req.Temperature != nil {
		upstream["temperature"] = *req.Temperature
	}
	if req.MaxTokens != nil {
		upstream["max_tokens"] = *req.MaxTokens
	}
	if req.ResponseFormat != nil {
		upstream["response_format"] = req.ResponseFormat
	}

	return upstream
}

// proxyStreamResponse streams the upstream SSE response to the client.
func proxyStreamResponse(w http.ResponseWriter, resp *http.Response) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeNLError(w, http.StatusInternalServerError, "STREAMING_NOT_SUPPORTED", "Server does not support streaming")
		return
	}

	// Copy upstream headers
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no") // Disable nginx buffering

	// Forward upstream status code
	if resp.StatusCode != http.StatusOK {
		// For errors, read body and return as JSON
		body, _ := io.ReadAll(io.LimitReader(resp.Body, nlMaxResponseSize))
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(resp.StatusCode)
		_, _ = w.Write(body)
		return
	}

	w.WriteHeader(http.StatusOK)
	flusher.Flush()

	// Stream chunks from upstream to client
	buf := make([]byte, 4096)
	for {
		n, err := resp.Body.Read(buf)
		if n > 0 {
			_, writeErr := w.Write(buf[:n])
			if writeErr != nil {
				log.Printf("[NL/stream] Client write error: %v", writeErr)
				return
			}
			flusher.Flush()
		}
		if err != nil {
			if err != io.EOF {
				log.Printf("[NL/stream] Upstream read error: %v", err)
			}
			return
		}
	}
}

// proxyBufferedResponse reads the full upstream response and sends it.
func proxyBufferedResponse(w http.ResponseWriter, resp *http.Response) {
	body, err := io.ReadAll(io.LimitReader(resp.Body, nlMaxResponseSize))
	if err != nil {
		writeNLError(w, http.StatusBadGateway, "READ_ERROR", fmt.Sprintf("Failed to read upstream response: %v", err))
		return
	}

	// Copy relevant headers
	ct := resp.Header.Get("Content-Type")
	if ct == "" {
		ct = "application/json"
	}
	w.Header().Set("Content-Type", ct)

	w.WriteHeader(resp.StatusCode)
	_, _ = w.Write(body)
}

// writeNLError writes a JSON error response.
func writeNLError(w http.ResponseWriter, statusCode int, code, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	_ = json.NewEncoder(w).Encode(NLProxyErrorResponse{
		Error:   message,
		Code:    code,
		Details: "",
	})
}

// maskEndpoint returns a masked version of the endpoint for the config response.
// Shows the host but not full path details.
func maskEndpoint(endpoint string) string {
	if endpoint == "" {
		return ""
	}
	parsed, err := url.Parse(endpoint)
	if err != nil {
		return "(configured)"
	}
	return fmt.Sprintf("%s://%s/...", parsed.Scheme, parsed.Host)
}
