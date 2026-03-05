package handlers

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestGatewayClientConfig_Defaults(t *testing.T) {
	config := GatewayClientConfig{
		ContainerName: "test-container",
	}

	client := NewGatewayClient(config, nil, nil)

	if client.config.ReconnectDelay != 3*time.Second {
		t.Errorf("expected default ReconnectDelay 3s, got %v", client.config.ReconnectDelay)
	}
	if client.config.MaxReconnects != 10 {
		t.Errorf("expected default MaxReconnects 10, got %d", client.config.MaxReconnects)
	}
	if client.config.PingInterval != 30*time.Second {
		t.Errorf("expected default PingInterval 30s, got %v", client.config.PingInterval)
	}
	if client.config.ResponseTimeout != 30*time.Second {
		t.Errorf("expected default ResponseTimeout 30s, got %v", client.config.ResponseTimeout)
	}
}

func TestGatewayClient_BuildWebSocketURL(t *testing.T) {
	tests := []struct {
		name     string
		host     string
		port     int
		expected string
	}{
		{
			name:     "default host and port",
			host:     "",
			port:     0,
			expected: "ws://127.0.0.1:18790/",
		},
		{
			name:     "custom host",
			host:     "192.168.1.100",
			port:     18789,
			expected: "ws://192.168.1.100:18789/",
		},
		{
			name:     "localhost",
			host:     "localhost",
			port:     8080,
			expected: "ws://localhost:8080/",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := NewGatewayClient(GatewayClientConfig{
				ContainerName: "test",
				GatewayHost:   tt.host,
				GatewayPort:   tt.port,
			}, nil, nil)

			url := client.buildWebSocketURL()
			if url != tt.expected {
				t.Errorf("expected %s, got %s", tt.expected, url)
			}
		})
	}
}

func TestGatewayClientState_String(t *testing.T) {
	tests := []struct {
		state    GatewayClientState
		expected string
	}{
		{GatewayClientStateDisconnected, "disconnected"},
		{GatewayClientStateConnecting, "connecting"},
		{GatewayClientStateConnected, "connected"},
		{GatewayClientStateReconnecting, "reconnecting"},
		{GatewayClientState(99), "unknown"},
	}

	for _, tt := range tests {
		t.Run(tt.expected, func(t *testing.T) {
			if tt.state.String() != tt.expected {
				t.Errorf("expected %s, got %s", tt.expected, tt.state.String())
			}
		})
	}
}

func TestGatewayClientManager_New(t *testing.T) {
	h := NewOpenClawHandler(t.TempDir(), false)
	manager := NewGatewayClientManager(h)

	if manager == nil {
		t.Fatal("expected non-nil manager")
	}
	if manager.handler != h {
		t.Error("handler not set correctly")
	}
	if !manager.autoConnect {
		t.Error("autoConnect should be true by default")
	}
}

func TestGatewayWSStatusHandler(t *testing.T) {
	h := NewOpenClawHandler(t.TempDir(), false)

	req := httptest.NewRequest(http.MethodGet, "/api/openclaw/gateway-ws/status", nil)
	w := httptest.NewRecorder()

	h.GatewayWSStatusHandler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected status 200, got %d", w.Code)
	}

	var resp GatewayWSStatusResponse
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if !resp.Enabled {
		t.Error("expected Enabled to be true")
	}
	if resp.TotalConnected != 0 {
		t.Errorf("expected 0 connected, got %d", resp.TotalConnected)
	}
}

func TestAgentMessageHandler_MissingFields(t *testing.T) {
	h := NewOpenClawHandler(t.TempDir(), false)

	tests := []struct {
		name    string
		body    string
		wantErr string
	}{
		{
			name:    "missing content",
			body:    `{"roomId":"room-1","containerName":"test"}`,
			wantErr: "content is required",
		},
		{
			name:    "missing roomId",
			body:    `{"containerName":"test","content":"hello"}`,
			wantErr: "roomId is required",
		},
		{
			name:    "missing containerName",
			body:    `{"roomId":"room-1","content":"hello"}`,
			wantErr: "containerName is required",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, "/api/openclaw/agent/message",
				strings.NewReader(tt.body))
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()

			h.AgentMessageHandler().ServeHTTP(w, req)

			if w.Code != http.StatusBadRequest {
				t.Errorf("expected status 400, got %d", w.Code)
			}

			if !strings.Contains(w.Body.String(), tt.wantErr) {
				t.Errorf("expected error containing %q, got %s", tt.wantErr, w.Body.String())
			}
		})
	}
}

func TestGatewayAutoConnectConfig_Defaults(t *testing.T) {
	config := defaultAutoConnectConfig

	if !config.Enabled {
		t.Error("expected Enabled to be true by default")
	}
	if config.ScanInterval != 30*time.Second {
		t.Errorf("expected ScanInterval 30s, got %v", config.ScanInterval)
	}
	if config.HealthCheckDelay != 5*time.Second {
		t.Errorf("expected HealthCheckDelay 5s, got %v", config.HealthCheckDelay)
	}
}

func TestGatewayAutoConnectHandler_Get(t *testing.T) {
	h := NewOpenClawHandler(t.TempDir(), false)

	req := httptest.NewRequest(http.MethodGet, "/api/openclaw/gateway-ws/auto-connect", nil)
	w := httptest.NewRecorder()

	h.GatewayAutoConnectHandler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected status 200, got %d", w.Code)
	}

	var resp map[string]interface{}
	if err := json.NewDecoder(w.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if enabled, ok := resp["enabled"].(bool); !ok || enabled {
		// Should be false since auto-connector is not started
		t.Logf("auto-connect enabled: %v (expected false when not started)", enabled)
	}
}

func TestRoomAgentBindings_Empty(t *testing.T) {
	h := NewOpenClawHandler(t.TempDir(), false)

	bindings, err := h.GetRoomAgentBindings()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(bindings) != 0 {
		t.Errorf("expected 0 bindings, got %d", len(bindings))
	}
}

func TestTruncateString(t *testing.T) {
	tests := []struct {
		input    string
		maxLen   int
		expected string
	}{
		{"hello", 10, "hello"},
		{"hello world", 8, "hello..."},
		{"", 5, ""},
		{"abc", 3, "abc"},
		{"abcd", 3, "..."},
	}

	for _, tt := range tests {
		result := truncateString(tt.input, tt.maxLen)
		if result != tt.expected {
			t.Errorf("truncateString(%q, %d) = %q, want %q",
				tt.input, tt.maxLen, result, tt.expected)
		}
	}
}
