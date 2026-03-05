package handlers

import (
	"crypto/ed25519"
	"encoding/base64"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
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

func TestDeviceKeyPairPersistence(t *testing.T) {
	// Create a temporary directory for the test
	tempDir := t.TempDir()

	// Reset global state for testing
	deviceKeyPairOnce = sync.Once{}
	globalDeviceKeyPair = nil

	// Set the data directory
	SetDeviceKeyPairDataDir(tempDir)

	// Get or create key pair (should generate new)
	kp1 := getOrCreateDeviceKeyPair()
	if kp1 == nil {
		t.Fatal("expected non-nil key pair")
	}
	if len(kp1.DeviceID) != 64 { // SHA256 hex = 64 chars
		t.Errorf("expected 64-char device ID, got %d", len(kp1.DeviceID))
	}
	if len(kp1.PublicKey) != 32 {
		t.Errorf("expected 32-byte public key, got %d", len(kp1.PublicKey))
	}
	if len(kp1.PrivateKey) != 64 {
		t.Errorf("expected 64-byte private key, got %d", len(kp1.PrivateKey))
	}

	// Verify key pair file was created
	keyPairPath := deviceKeyPairPath()
	if _, err := os.Stat(keyPairPath); os.IsNotExist(err) {
		t.Fatal("device key pair file was not created")
	}

	// Reset and load again (should load from disk)
	deviceKeyPairOnce = sync.Once{}
	globalDeviceKeyPair = nil

	kp2 := getOrCreateDeviceKeyPair()
	if kp2 == nil {
		t.Fatal("expected non-nil key pair on reload")
	}

	// Verify device ID is stable across restarts
	if kp2.DeviceID != kp1.DeviceID {
		t.Errorf("device ID changed after restart: %s -> %s", kp1.DeviceID[:16], kp2.DeviceID[:16])
	}

	// Verify keys are identical
	if string(kp2.PublicKey) != string(kp1.PublicKey) {
		t.Error("public key changed after restart")
	}
	if string(kp2.PrivateKey) != string(kp1.PrivateKey) {
		t.Error("private key changed after restart")
	}
}

func TestDeviceKeyPairSignNonce(t *testing.T) {
	tempDir := t.TempDir()

	// Reset global state
	deviceKeyPairOnce = sync.Once{}
	globalDeviceKeyPair = nil
	SetDeviceKeyPairDataDir(tempDir)

	kp := getOrCreateDeviceKeyPair()
	if kp == nil {
		t.Fatal("expected non-nil key pair")
	}

	// Sign a test nonce
	nonce := "test-challenge-nonce-12345"
	signature := kp.signNonce(nonce)

	if signature == "" {
		t.Error("expected non-empty signature")
	}

	// Verify signature is valid base64
	sigBytes, err := base64.StdEncoding.DecodeString(signature)
	if err != nil {
		t.Errorf("signature is not valid base64: %v", err)
	}

	// Ed25519 signatures are 64 bytes
	if len(sigBytes) != 64 {
		t.Errorf("expected 64-byte signature, got %d", len(sigBytes))
	}

	// Verify the signature is actually valid
	if !ed25519.Verify(kp.PublicKey, []byte(nonce), sigBytes) {
		t.Error("signature verification failed")
	}
}

func TestDeviceKeyPairPublicKeyBase64(t *testing.T) {
	tempDir := t.TempDir()

	// Reset global state
	deviceKeyPairOnce = sync.Once{}
	globalDeviceKeyPair = nil
	SetDeviceKeyPairDataDir(tempDir)

	kp := getOrCreateDeviceKeyPair()
	if kp == nil {
		t.Fatal("expected non-nil key pair")
	}

	pubKeyB64 := kp.publicKeyBase64()
	if pubKeyB64 == "" {
		t.Error("expected non-empty public key base64")
	}

	// Verify it's valid base64 and decodes to 32 bytes
	decoded, err := base64.StdEncoding.DecodeString(pubKeyB64)
	if err != nil {
		t.Errorf("public key base64 is invalid: %v", err)
	}
	if len(decoded) != 32 {
		t.Errorf("expected 32-byte public key, got %d", len(decoded))
	}
}
