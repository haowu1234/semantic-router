package handlers

import (
	"context"
	"crypto/ed25519"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"
)

// --- Gateway WebSocket Protocol Types ---

// GatewayWSMessageType represents the type of Gateway WebSocket message
type GatewayWSMessageType string

const (
	// Outbound (client -> gateway)
	GatewayWSTypeReq GatewayWSMessageType = "req"

	// Inbound (gateway -> client)
	GatewayWSTypeRes   GatewayWSMessageType = "res"
	GatewayWSTypeEvent GatewayWSMessageType = "event"
	GatewayWSTypeError GatewayWSMessageType = "error"
)

// GatewayWSMessage represents a message in the Gateway WebSocket protocol
type GatewayWSMessage struct {
	Type    GatewayWSMessageType   `json:"type"`
	ID      string                 `json:"id,omitempty"`
	Method  string                 `json:"method,omitempty"`
	Params  map[string]interface{} `json:"params,omitempty"`
	OK      *bool                  `json:"ok,omitempty"`      // Response success flag per protocol
	Payload interface{}            `json:"payload,omitempty"` // Response/Event payload per protocol
	Event   string                 `json:"event,omitempty"`
	Error   *GatewayWSError        `json:"error,omitempty"`
	Seq     *int64                 `json:"seq,omitempty"`          // Event sequence number
	StateV  *int64                 `json:"stateVersion,omitempty"` // Event state version
}

// GatewayWSError represents an error in Gateway WebSocket protocol
type GatewayWSError struct {
	Code    interface{} `json:"code,omitempty"` // Can be int or string
	Message string      `json:"message"`
}

// GatewayAgentEvent represents an agent event from the Gateway
type GatewayAgentEvent struct {
	Type       string                 `json:"type"`
	AgentID    string                 `json:"agentId,omitempty"`
	SessionID  string                 `json:"sessionId,omitempty"`
	Content    string                 `json:"content,omitempty"`
	ToolName   string                 `json:"toolName,omitempty"`
	ToolInput  map[string]interface{} `json:"toolInput,omitempty"`
	ToolOutput string                 `json:"toolOutput,omitempty"`
	Done       bool                   `json:"done,omitempty"`
	Error      string                 `json:"error,omitempty"`
	Timestamp  string                 `json:"timestamp,omitempty"`
}

// GatewayClientState represents the connection state
type GatewayClientState int32

const (
	GatewayClientStateDisconnected GatewayClientState = iota
	GatewayClientStateConnecting
	GatewayClientStateConnected
	GatewayClientStateReconnecting
)

func (s GatewayClientState) String() string {
	switch s {
	case GatewayClientStateDisconnected:
		return "disconnected"
	case GatewayClientStateConnecting:
		return "connecting"
	case GatewayClientStateConnected:
		return "connected"
	case GatewayClientStateReconnecting:
		return "reconnecting"
	default:
		return "unknown"
	}
}

// --- Device Key Pair for Gateway Authentication ---

// DeviceKeyPair holds the Ed25519 key pair for device authentication.
// Per OpenClaw Gateway protocol, the device.id is derived from the public key fingerprint
// to provide a stable identity across restarts. The key pair is persisted to disk.
type DeviceKeyPair struct {
	PrivateKey ed25519.PrivateKey
	PublicKey  ed25519.PublicKey
	DeviceID   string // SHA256 fingerprint of public key (stable identity)
}

// deviceKeyPairJSON is the JSON serialization format for persisting the key pair
type deviceKeyPairJSON struct {
	PrivateKey string `json:"privateKey"` // base64-encoded Ed25519 private key (64 bytes)
	PublicKey  string `json:"publicKey"`  // base64-encoded Ed25519 public key (32 bytes)
	DeviceID   string `json:"deviceId"`   // SHA256 fingerprint of public key
	CreatedAt  string `json:"createdAt"`  // ISO8601 timestamp
}

var (
	globalDeviceKeyPair   *DeviceKeyPair
	deviceKeyPairOnce     sync.Once
	deviceKeyPairDataDir  string
	deviceKeyPairDataMu   sync.Mutex
)

// SetDeviceKeyPairDataDir sets the data directory for persisting the device key pair.
// This must be called before any connections are established.
func SetDeviceKeyPairDataDir(dataDir string) {
	deviceKeyPairDataMu.Lock()
	defer deviceKeyPairDataMu.Unlock()
	deviceKeyPairDataDir = dataDir
}

// deviceKeyPairPath returns the file path for the persisted device key pair
func deviceKeyPairPath() string {
	deviceKeyPairDataMu.Lock()
	dir := deviceKeyPairDataDir
	deviceKeyPairDataMu.Unlock()

	if dir == "" {
		// Fallback to current directory if not configured
		dir = "."
	}
	return filepath.Join(dir, "device-keypair.json")
}

// loadDeviceKeyPair attempts to load an existing key pair from disk
func loadDeviceKeyPair() (*DeviceKeyPair, error) {
	path := deviceKeyPairPath()
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var stored deviceKeyPairJSON
	if err := json.Unmarshal(data, &stored); err != nil {
		return nil, fmt.Errorf("invalid device key pair JSON: %w", err)
	}

	privateKeyBytes, err := base64.StdEncoding.DecodeString(stored.PrivateKey)
	if err != nil {
		return nil, fmt.Errorf("invalid private key encoding: %w", err)
	}
	if len(privateKeyBytes) != ed25519.PrivateKeySize {
		return nil, fmt.Errorf("invalid private key size: got %d, want %d", len(privateKeyBytes), ed25519.PrivateKeySize)
	}

	publicKeyBytes, err := base64.StdEncoding.DecodeString(stored.PublicKey)
	if err != nil {
		return nil, fmt.Errorf("invalid public key encoding: %w", err)
	}
	if len(publicKeyBytes) != ed25519.PublicKeySize {
		return nil, fmt.Errorf("invalid public key size: got %d, want %d", len(publicKeyBytes), ed25519.PublicKeySize)
	}

	// Verify device ID matches public key fingerprint
	hash := sha256.Sum256(publicKeyBytes)
	expectedDeviceID := hex.EncodeToString(hash[:])
	if stored.DeviceID != expectedDeviceID {
		return nil, fmt.Errorf("device ID mismatch: stored %s, computed %s", stored.DeviceID[:16], expectedDeviceID[:16])
	}

	return &DeviceKeyPair{
		PrivateKey: ed25519.PrivateKey(privateKeyBytes),
		PublicKey:  ed25519.PublicKey(publicKeyBytes),
		DeviceID:   stored.DeviceID,
	}, nil
}

// saveDeviceKeyPair persists the key pair to disk
func saveDeviceKeyPair(kp *DeviceKeyPair) error {
	path := deviceKeyPairPath()

	// Ensure parent directory exists
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("failed to create directory for device key pair: %w", err)
	}

	stored := deviceKeyPairJSON{
		PrivateKey: base64.StdEncoding.EncodeToString(kp.PrivateKey),
		PublicKey:  base64.StdEncoding.EncodeToString(kp.PublicKey),
		DeviceID:   kp.DeviceID,
		CreatedAt:  time.Now().UTC().Format(time.RFC3339),
	}

	data, err := json.MarshalIndent(stored, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal device key pair: %w", err)
	}

	// Write with restricted permissions (private key!)
	if err := os.WriteFile(path, data, 0o600); err != nil {
		return fmt.Errorf("failed to write device key pair: %w", err)
	}

	log.Printf("openclaw-gw: persisted device key pair to %s", path)
	return nil
}

// getOrCreateDeviceKeyPair returns the global device key pair.
// It first attempts to load from disk for stable identity across restarts.
// If not found, generates a new key pair and persists it.
func getOrCreateDeviceKeyPair() *DeviceKeyPair {
	deviceKeyPairOnce.Do(func() {
		// First, try to load existing key pair from disk
		if loaded, err := loadDeviceKeyPair(); err == nil {
			globalDeviceKeyPair = loaded
			log.Printf("openclaw-gw: loaded persisted device key pair, deviceID=%s...", loaded.DeviceID[:16])
			return
		} else if !os.IsNotExist(err) {
			// Log non-NotExist errors but continue to generate new key pair
			log.Printf("openclaw-gw: failed to load device key pair (will generate new): %v", err)
		}

		// Generate new key pair
		publicKey, privateKey, err := ed25519.GenerateKey(rand.Reader)
		if err != nil {
			log.Printf("openclaw-gw: failed to generate device key pair: %v", err)
			return
		}

		// Generate device ID as SHA256 fingerprint of public key
		hash := sha256.Sum256(publicKey)
		deviceID := hex.EncodeToString(hash[:])

		globalDeviceKeyPair = &DeviceKeyPair{
			PrivateKey: privateKey,
			PublicKey:  publicKey,
			DeviceID:   deviceID,
		}

		log.Printf("openclaw-gw: generated new device key pair, deviceID=%s...", deviceID[:16])

		// Persist to disk for stable identity across restarts
		if err := saveDeviceKeyPair(globalDeviceKeyPair); err != nil {
			log.Printf("openclaw-gw: warning: failed to persist device key pair: %v", err)
		}
	})
	return globalDeviceKeyPair
}

// buildDeviceAuthPayloadV3 builds the canonical payload string for device authentication
// Format: v3|deviceId|clientId|clientMode|role|scopes|signedAtMs|token|nonce|platform|deviceFamily
// This must match the server-side payload construction in OpenClaw Gateway
func buildDeviceAuthPayloadV3(params DeviceAuthPayloadParams) string {
	scopes := strings.Join(params.Scopes, ",")
	token := params.Token
	if token == "" {
		token = ""
	}
	platform := normalizeDeviceMetadata(params.Platform)
	deviceFamily := normalizeDeviceMetadata(params.DeviceFamily)
	return strings.Join([]string{
		"v3",
		params.DeviceID,
		params.ClientID,
		params.ClientMode,
		params.Role,
		scopes,
		fmt.Sprintf("%d", params.SignedAtMs),
		token,
		params.Nonce,
		platform,
		deviceFamily,
	}, "|")
}

// DeviceAuthPayloadParams holds parameters for building device auth payload
type DeviceAuthPayloadParams struct {
	DeviceID     string
	ClientID     string
	ClientMode   string
	Role         string
	Scopes       []string
	SignedAtMs   int64
	Token        string
	Nonce        string
	Platform     string
	DeviceFamily string
}

// normalizeDeviceMetadata normalizes device metadata strings for signature
// This must match Gateway's normalizeDeviceMetadataForAuth:
// - trim whitespace
// - convert ASCII A-Z to lowercase (NOT unicode lowercase)
func normalizeDeviceMetadata(s string) string {
	trimmed := strings.TrimSpace(s)
	if trimmed == "" {
		return ""
	}
	// Only convert ASCII uppercase A-Z to lowercase a-z
	// This matches Gateway's toLowerAscii function
	result := make([]byte, len(trimmed))
	for i := 0; i < len(trimmed); i++ {
		c := trimmed[i]
		if c >= 'A' && c <= 'Z' {
			result[i] = c + 32 // Convert to lowercase
		} else {
			result[i] = c
		}
	}
	return string(result)
}

// publicKeyBase64Url returns the raw 32-byte public key as base64url string
// Gateway expects raw Ed25519 public key bytes (32 bytes), not SPKI format
func (kp *DeviceKeyPair) publicKeyBase64Url() string {
	// Ed25519 public keys in Go are already 32 bytes raw format
	return base64URLEncode(kp.PublicKey)
}

// base64URLEncode encodes bytes to base64url format (no padding)
func base64URLEncode(data []byte) string {
	encoded := base64.RawURLEncoding.EncodeToString(data)
	return encoded
}

// signPayload signs a payload string with the device's private key
// Returns signature in base64url format
func (kp *DeviceKeyPair) signPayload(payload string) string {
	signature := ed25519.Sign(kp.PrivateKey, []byte(payload))
	return base64URLEncode(signature)
}

// GatewayClientConfig holds configuration for a Gateway WebSocket client
type GatewayClientConfig struct {
	ContainerName   string
	GatewayHost     string
	GatewayPort     int
	AuthToken       string
	TeamID          string
	RoomID          string
	ReconnectDelay  time.Duration
	MaxReconnects   int
	PingInterval    time.Duration // Will be updated from server's tickIntervalMs
	ResponseTimeout time.Duration
}

// GatewayClientEventHandler handles events from the Gateway
type GatewayClientEventHandler interface {
	OnAgentEvent(containerName string, event GatewayAgentEvent)
	OnStateChange(containerName string, oldState, newState GatewayClientState)
	OnError(containerName string, err error)
}

// GatewayClient manages a WebSocket connection to an OpenClaw Gateway
type GatewayClient struct {
	config       GatewayClientConfig
	handler      GatewayClientEventHandler
	dashHandler  *OpenClawHandler
	conn         *websocket.Conn
	state        int32
	reqID        int64
	pendingReqs  sync.Map // reqID -> chan *GatewayWSMessage
	closeChan    chan struct{}
	sendChan     chan []byte
	mu           sync.Mutex
	reconnects   int
	lastActivity time.Time
	ctx          context.Context
	cancel       context.CancelFunc
}

// NewGatewayClient creates a new Gateway WebSocket client
func NewGatewayClient(config GatewayClientConfig, handler GatewayClientEventHandler, dashHandler *OpenClawHandler) *GatewayClient {
	if config.ReconnectDelay == 0 {
		config.ReconnectDelay = 3 * time.Second
	}
	if config.MaxReconnects == 0 {
		config.MaxReconnects = 10
	}
	if config.PingInterval == 0 {
		config.PingInterval = 30 * time.Second
	}
	if config.ResponseTimeout == 0 {
		config.ResponseTimeout = 30 * time.Second
	}

	ctx, cancel := context.WithCancel(context.Background())

	return &GatewayClient{
		config:      config,
		handler:     handler,
		dashHandler: dashHandler,
		closeChan:   make(chan struct{}),
		sendChan:    make(chan []byte, 64),
		ctx:         ctx,
		cancel:      cancel,
	}
}

// Connect establishes the WebSocket connection
func (c *GatewayClient) Connect() error {
	c.mu.Lock()
	if c.getState() == GatewayClientStateConnected {
		c.mu.Unlock()
		return nil
	}
	c.setState(GatewayClientStateConnecting)
	c.mu.Unlock()

	wsURL := c.buildWebSocketURL()
	log.Printf("openclaw-gw: connecting to %s for container %s", wsURL, c.config.ContainerName)

	dialer := websocket.Dialer{
		HandshakeTimeout: 10 * time.Second,
	}

	headers := http.Header{}
	if c.config.AuthToken != "" {
		headers.Set("Authorization", "Bearer "+c.config.AuthToken)
		headers.Set("X-OpenClaw-Token", c.config.AuthToken)
	}

	conn, _, err := dialer.DialContext(c.ctx, wsURL, headers)
	if err != nil {
		c.setState(GatewayClientStateDisconnected)
		return fmt.Errorf("failed to connect to gateway: %w", err)
	}

	c.conn = conn
	c.lastActivity = time.Now()

	// Perform handshake
	if err := c.performHandshake(); err != nil {
		c.conn.Close()
		c.setState(GatewayClientStateDisconnected)
		return fmt.Errorf("handshake failed: %w", err)
	}

	c.setState(GatewayClientStateConnected)
	c.reconnects = 0

	log.Printf("openclaw-gw: connected to gateway for container %s", c.config.ContainerName)

	// Start read/write/ping goroutines
	go c.readPump()
	go c.writePump()
	go c.pingPump()

	return nil
}

// Disconnect closes the WebSocket connection
func (c *GatewayClient) Disconnect() {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.getState() == GatewayClientStateDisconnected {
		return
	}

	c.cancel()
	close(c.closeChan)

	if c.conn != nil {
		_ = c.conn.WriteMessage(websocket.CloseMessage, websocket.FormatCloseMessage(websocket.CloseNormalClosure, ""))
		_ = c.conn.Close()
	}

	c.setState(GatewayClientStateDisconnected)
	log.Printf("openclaw-gw: disconnected from gateway for container %s", c.config.ContainerName)
}

// State returns the current connection state
func (c *GatewayClient) State() GatewayClientState {
	return c.getState()
}

// ContainerName returns the container name this client is connected to
func (c *GatewayClient) ContainerName() string {
	return c.config.ContainerName
}

// SendRequest sends a request to the Gateway and waits for response
func (c *GatewayClient) SendRequest(method string, params map[string]interface{}) (*GatewayWSMessage, error) {
	if c.getState() != GatewayClientStateConnected {
		return nil, fmt.Errorf("not connected to gateway")
	}

	reqID := fmt.Sprintf("%d", atomic.AddInt64(&c.reqID, 1))
	msg := GatewayWSMessage{
		Type:   GatewayWSTypeReq,
		ID:     reqID,
		Method: method,
		Params: params,
	}

	respChan := make(chan *GatewayWSMessage, 1)
	c.pendingReqs.Store(reqID, respChan)
	defer c.pendingReqs.Delete(reqID)

	data, err := json.Marshal(msg)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	select {
	case c.sendChan <- data:
	case <-c.ctx.Done():
		return nil, fmt.Errorf("client shutting down")
	case <-time.After(5 * time.Second):
		return nil, fmt.Errorf("send timeout")
	}

	select {
	case resp := <-respChan:
		if resp.Error != nil {
			return nil, fmt.Errorf("gateway error: %s", resp.Error.Message)
		}
		return resp, nil
	case <-c.ctx.Done():
		return nil, fmt.Errorf("client shutting down")
	case <-time.After(c.config.ResponseTimeout):
		return nil, fmt.Errorf("response timeout")
	}
}

// --- Internal methods ---

func (c *GatewayClient) buildWebSocketURL() string {
	host := c.config.GatewayHost
	if host == "" {
		host = "127.0.0.1"
	}
	port := c.config.GatewayPort
	if port == 0 {
		port = 18790
	}

	u := url.URL{
		Scheme: "ws",
		Host:   fmt.Sprintf("%s:%d", host, port),
		Path:   "/",
	}
	return u.String()
}

func (c *GatewayClient) performHandshake() error {
	// Get or create device key pair for authentication
	keyPair := getOrCreateDeviceKeyPair()
	if keyPair == nil {
		return fmt.Errorf("failed to get device key pair")
	}

	// First, wait for connect.challenge event from Gateway
	_ = c.conn.SetReadDeadline(time.Now().Add(10 * time.Second))
	_, challengeData, err := c.conn.ReadMessage()
	_ = c.conn.SetReadDeadline(time.Time{})

	var challengeNonce string
	if err == nil {
		// Try to parse challenge
		var challengeMsg struct {
			Type    string `json:"type"`
			Event   string `json:"event"`
			Payload struct {
				Nonce string `json:"nonce"`
				Ts    int64  `json:"ts"`
			} `json:"payload"`
		}
		if json.Unmarshal(challengeData, &challengeMsg) == nil {
			if challengeMsg.Event == "connect.challenge" {
				challengeNonce = challengeMsg.Payload.Nonce
				log.Printf("openclaw-gw: received connect.challenge for container %s, nonce=%s...",
					c.config.ContainerName, challengeNonce[:min(16, len(challengeNonce))])
			}
		}
	}

	// Define client parameters
	// Valid client.id values (from GATEWAY_CLIENT_IDS):
	//   "webchat-ui", "openclaw-control-ui", "webchat", "cli", "gateway-client",
	//   "openclaw-macos", "openclaw-ios", "openclaw-android", "node-host",
	//   "test", "fingerprint", "openclaw-probe"
	// Valid client.mode values (from GATEWAY_CLIENT_MODES):
	//   "webchat", "cli", "ui", "backend", "node", "probe", "test"
	clientID := "gateway-client"
	clientMode := "backend"
	role := "operator"
	scopes := []string{"operator.read", "operator.write"}
	platform := "linux"
	signedAtMs := time.Now().UnixMilli()

	// Build device authentication object per protocol spec
	// See: OpenClaw Gateway device-auth.ts
	var deviceAuth map[string]interface{}
	if challengeNonce != "" {
		// Build the canonical v3 payload for signing
		payload := buildDeviceAuthPayloadV3(DeviceAuthPayloadParams{
			DeviceID:     keyPair.DeviceID,
			ClientID:     clientID,
			ClientMode:   clientMode,
			Role:         role,
			Scopes:       scopes,
			SignedAtMs:   signedAtMs,
			Token:        c.config.AuthToken,
			Nonce:        challengeNonce,
			Platform:     platform,
			DeviceFamily: "",
		})

		// Debug log the payload (truncate nonce for security)
		truncatedPayload := payload
		if len(truncatedPayload) > 100 {
			truncatedPayload = truncatedPayload[:100] + "..."
		}
		log.Printf("openclaw-gw: signing payload: %s", truncatedPayload)

		signature := keyPair.signPayload(payload)

		deviceAuth = map[string]interface{}{
			"id":        keyPair.DeviceID,
			"publicKey": keyPair.publicKeyBase64Url(),
			"signature": signature,
			"signedAt":  signedAtMs,
			"nonce":     challengeNonce,
		}

		log.Printf("openclaw-gw: signed device auth payload for container %s, signedAt=%d", c.config.ContainerName, signedAtMs)
	} else {
		// No challenge received - some configurations may allow this
		deviceAuth = map[string]interface{}{
			"id":        keyPair.DeviceID,
			"publicKey": keyPair.publicKeyBase64Url(),
		}
	}

	// Send connect request per OpenClaw Gateway protocol
	params := map[string]interface{}{
		"minProtocol": 3,
		"maxProtocol": 3,
		"client": map[string]interface{}{
			"id":       clientID,  // For backend service connections
			"version":  "1.0.0",
			"platform": platform,
			"mode":     clientMode, // Backend service mode
		},
		"role":      role,
		"scopes":    scopes,
		"device":    deviceAuth,
		"locale":    "en-US",
		"userAgent": "semantic-router-dashboard/1.0.0 (Go-http-client)",
	}

	if c.config.AuthToken != "" {
		params["auth"] = map[string]interface{}{
			"token": c.config.AuthToken,
		}
	}

	reqID := fmt.Sprintf("%d", atomic.AddInt64(&c.reqID, 1))
	msg := GatewayWSMessage{
		Type:   GatewayWSTypeReq,
		ID:     reqID,
		Method: "connect",
		Params: params,
	}

	data, err := json.Marshal(msg)
	if err != nil {
		return fmt.Errorf("failed to marshal handshake: %w", err)
	}

	log.Printf("openclaw-gw: sending connect request for container %s", c.config.ContainerName)

	if err := c.conn.WriteMessage(websocket.TextMessage, data); err != nil {
		return fmt.Errorf("failed to send handshake: %w", err)
	}

	// Wait for response
	_ = c.conn.SetReadDeadline(time.Now().Add(10 * time.Second))
	_, respData, err := c.conn.ReadMessage()
	if err != nil {
		return fmt.Errorf("failed to read handshake response: %w", err)
	}
	_ = c.conn.SetReadDeadline(time.Time{})

	var resp struct {
		Type    string          `json:"type"`
		ID      string          `json:"id"`
		OK      bool            `json:"ok"`
		Payload json.RawMessage `json:"payload"`
		Error   *GatewayWSError `json:"error"`
	}
	if err := json.Unmarshal(respData, &resp); err != nil {
		return fmt.Errorf("invalid handshake response: %w", err)
	}

	if resp.Error != nil {
		return fmt.Errorf("handshake rejected: %s", resp.Error.Message)
	}

	if !resp.OK {
		return fmt.Errorf("handshake failed: ok=false")
	}

	// Parse hello-ok payload to extract policy.tickIntervalMs
	if resp.Payload != nil {
		var helloOK struct {
			Type     string `json:"type"`
			Protocol int    `json:"protocol"`
			Policy   struct {
				TickIntervalMs int `json:"tickIntervalMs"`
			} `json:"policy"`
		}
		if json.Unmarshal(resp.Payload, &helloOK) == nil {
			if helloOK.Policy.TickIntervalMs > 0 {
				// Update ping interval based on server policy
				c.config.PingInterval = time.Duration(helloOK.Policy.TickIntervalMs) * time.Millisecond
				log.Printf("openclaw-gw: using server tickInterval %dms for container %s",
					helloOK.Policy.TickIntervalMs, c.config.ContainerName)
			}
		}
	}

	log.Printf("openclaw-gw: handshake successful for container %s", c.config.ContainerName)
	return nil
}

func (c *GatewayClient) readPump() {
	defer func() {
		c.handleDisconnect()
	}()

	for {
		select {
		case <-c.ctx.Done():
			return
		default:
		}

		messageType, data, err := c.conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				log.Printf("openclaw-gw: read error for %s: %v", c.config.ContainerName, err)
			}
			return
		}

		if messageType != websocket.TextMessage {
			continue
		}

		c.lastActivity = time.Now()
		c.handleMessage(data)
	}
}

func (c *GatewayClient) writePump() {
	for {
		select {
		case <-c.ctx.Done():
			return
		case data := <-c.sendChan:
			if err := c.conn.WriteMessage(websocket.TextMessage, data); err != nil {
				log.Printf("openclaw-gw: write error for %s: %v", c.config.ContainerName, err)
				return
			}
			c.lastActivity = time.Now()
		}
	}
}

func (c *GatewayClient) pingPump() {
	ticker := time.NewTicker(c.config.PingInterval)
	defer ticker.Stop()

	for {
		select {
		case <-c.ctx.Done():
			return
		case <-ticker.C:
			if err := c.conn.WriteMessage(websocket.PingMessage, nil); err != nil {
				log.Printf("openclaw-gw: ping error for %s: %v", c.config.ContainerName, err)
				return
			}
		}
	}
}

func (c *GatewayClient) handleMessage(data []byte) {
	var msg GatewayWSMessage
	if err := json.Unmarshal(data, &msg); err != nil {
		log.Printf("openclaw-gw: failed to parse message for %s: %v", c.config.ContainerName, err)
		return
	}

	switch msg.Type {
	case GatewayWSTypeRes:
		c.handleResponse(msg)
	case GatewayWSTypeEvent:
		c.handleEvent(msg)
	case GatewayWSTypeError:
		if c.handler != nil {
			errMsg := "unknown error"
			if msg.Error != nil {
				errMsg = msg.Error.Message
			}
			c.handler.OnError(c.config.ContainerName, fmt.Errorf("gateway error: %s", errMsg))
		}
	}
}

func (c *GatewayClient) handleResponse(msg GatewayWSMessage) {
	if msg.ID == "" {
		return
	}

	if respChanI, ok := c.pendingReqs.Load(msg.ID); ok {
		if respChan, ok := respChanI.(chan *GatewayWSMessage); ok {
			select {
			case respChan <- &msg:
			default:
			}
		}
	}
}

func (c *GatewayClient) handleEvent(msg GatewayWSMessage) {
	if c.handler == nil {
		return
	}

	switch msg.Event {
	case "agent":
		c.handleAgentEvent(msg)
	case "message", "agent:message":
		c.handleAgentMessageEvent(msg)
	case "tool", "agent:tool":
		c.handleAgentToolEvent(msg)
	default:
		log.Printf("openclaw-gw: unknown event type %q for %s", msg.Event, c.config.ContainerName)
	}
}

func (c *GatewayClient) handleAgentEvent(msg GatewayWSMessage) {
	if msg.Payload == nil {
		return
	}

	data, err := json.Marshal(msg.Payload)
	if err != nil {
		return
	}

	var event GatewayAgentEvent
	if err := json.Unmarshal(data, &event); err != nil {
		log.Printf("openclaw-gw: failed to parse agent event: %v", err)
		return
	}

	event.Timestamp = time.Now().UTC().Format(time.RFC3339)
	c.handler.OnAgentEvent(c.config.ContainerName, event)
}

func (c *GatewayClient) handleAgentMessageEvent(msg GatewayWSMessage) {
	if msg.Payload == nil {
		return
	}

	data, err := json.Marshal(msg.Payload)
	if err != nil {
		return
	}

	var payload struct {
		Content   string `json:"content"`
		SessionID string `json:"sessionId"`
		AgentID   string `json:"agentId"`
		Done      bool   `json:"done"`
	}
	if err := json.Unmarshal(data, &payload); err != nil {
		return
	}

	event := GatewayAgentEvent{
		Type:      "message",
		AgentID:   payload.AgentID,
		SessionID: payload.SessionID,
		Content:   payload.Content,
		Done:      payload.Done,
		Timestamp: time.Now().UTC().Format(time.RFC3339),
	}

	c.handler.OnAgentEvent(c.config.ContainerName, event)
}

func (c *GatewayClient) handleAgentToolEvent(msg GatewayWSMessage) {
	if msg.Payload == nil {
		return
	}

	data, err := json.Marshal(msg.Payload)
	if err != nil {
		return
	}

	var payload struct {
		ToolName   string                 `json:"toolName"`
		ToolInput  map[string]interface{} `json:"toolInput"`
		ToolOutput string                 `json:"toolOutput"`
		SessionID  string                 `json:"sessionId"`
		AgentID    string                 `json:"agentId"`
	}
	if err := json.Unmarshal(data, &payload); err != nil {
		return
	}

	event := GatewayAgentEvent{
		Type:       "tool",
		AgentID:    payload.AgentID,
		SessionID:  payload.SessionID,
		ToolName:   payload.ToolName,
		ToolInput:  payload.ToolInput,
		ToolOutput: payload.ToolOutput,
		Timestamp:  time.Now().UTC().Format(time.RFC3339),
	}

	c.handler.OnAgentEvent(c.config.ContainerName, event)
}

func (c *GatewayClient) handleDisconnect() {
	oldState := c.getState()
	if oldState == GatewayClientStateDisconnected {
		return
	}

	c.setState(GatewayClientStateReconnecting)

	if c.handler != nil {
		c.handler.OnStateChange(c.config.ContainerName, oldState, GatewayClientStateReconnecting)
	}

	// Attempt reconnection
	go c.reconnect()
}

func (c *GatewayClient) reconnect() {
	for c.reconnects < c.config.MaxReconnects {
		select {
		case <-c.ctx.Done():
			return
		case <-time.After(c.config.ReconnectDelay):
		}

		c.reconnects++
		log.Printf("openclaw-gw: reconnect attempt %d/%d for %s",
			c.reconnects, c.config.MaxReconnects, c.config.ContainerName)

		// Create new context for reconnection
		c.ctx, c.cancel = context.WithCancel(context.Background())
		c.closeChan = make(chan struct{})
		c.sendChan = make(chan []byte, 64)

		if err := c.Connect(); err != nil {
			log.Printf("openclaw-gw: reconnect failed for %s: %v", c.config.ContainerName, err)
			continue
		}

		return
	}

	c.setState(GatewayClientStateDisconnected)
	if c.handler != nil {
		c.handler.OnError(c.config.ContainerName, fmt.Errorf("max reconnection attempts reached"))
	}
}

func (c *GatewayClient) getState() GatewayClientState {
	return GatewayClientState(atomic.LoadInt32(&c.state))
}

func (c *GatewayClient) setState(state GatewayClientState) {
	oldState := c.getState()
	atomic.StoreInt32(&c.state, int32(state))
	if c.handler != nil && oldState != state {
		c.handler.OnStateChange(c.config.ContainerName, oldState, state)
	}
}

// --- Gateway Client Manager ---

// GatewayClientManager manages multiple Gateway WebSocket clients
type GatewayClientManager struct {
	handler     *OpenClawHandler
	clients     sync.Map // containerName -> *GatewayClient
	eventSubs   sync.Map // subscriberID -> chan GatewayAgentEvent
	mu          sync.RWMutex
	autoConnect bool
}

// NewGatewayClientManager creates a new manager
func NewGatewayClientManager(handler *OpenClawHandler) *GatewayClientManager {
	return &GatewayClientManager{
		handler:     handler,
		autoConnect: true,
	}
}

// SetAutoConnect enables/disables automatic connection to new containers
func (m *GatewayClientManager) SetAutoConnect(enabled bool) {
	m.mu.Lock()
	m.autoConnect = enabled
	m.mu.Unlock()
}

// ConnectContainer connects to a specific container's Gateway
func (m *GatewayClientManager) ConnectContainer(containerName string) error {
	// Check if already connected
	if _, exists := m.clients.Load(containerName); exists {
		return nil
	}

	// Get container info
	entry := m.handler.findEntry(containerName)
	if entry == nil {
		return fmt.Errorf("container %q not found in registry", containerName)
	}

	// Resolve gateway host
	gatewayHost := m.resolveGatewayHost(containerName, entry.Port)

	config := GatewayClientConfig{
		ContainerName: containerName,
		GatewayHost:   gatewayHost,
		GatewayPort:   entry.Port,
		AuthToken:     entry.Token,
		TeamID:        entry.TeamID,
	}

	client := NewGatewayClient(config, m, m.handler)

	if err := client.Connect(); err != nil {
		return err
	}

	m.clients.Store(containerName, client)
	return nil
}

// DisconnectContainer disconnects from a container's Gateway
func (m *GatewayClientManager) DisconnectContainer(containerName string) {
	if clientI, exists := m.clients.LoadAndDelete(containerName); exists {
		if client, ok := clientI.(*GatewayClient); ok {
			client.Disconnect()
		}
	}
}

// DisconnectAll disconnects from all containers
func (m *GatewayClientManager) DisconnectAll() {
	m.clients.Range(func(key, value interface{}) bool {
		if client, ok := value.(*GatewayClient); ok {
			client.Disconnect()
		}
		m.clients.Delete(key)
		return true
	})
}

// GetClient returns the client for a container
func (m *GatewayClientManager) GetClient(containerName string) (*GatewayClient, bool) {
	if clientI, exists := m.clients.Load(containerName); exists {
		if client, ok := clientI.(*GatewayClient); ok {
			return client, true
		}
	}
	return nil, false
}

// SubscribeEvents subscribes to agent events from all containers
func (m *GatewayClientManager) SubscribeEvents(subscriberID string) chan GatewayAgentEvent {
	ch := make(chan GatewayAgentEvent, 64)
	m.eventSubs.Store(subscriberID, ch)
	return ch
}

// UnsubscribeEvents unsubscribes from agent events
func (m *GatewayClientManager) UnsubscribeEvents(subscriberID string) {
	if chI, exists := m.eventSubs.LoadAndDelete(subscriberID); exists {
		if ch, ok := chI.(chan GatewayAgentEvent); ok {
			close(ch)
		}
	}
}

// ConnectedContainers returns the list of connected container names
func (m *GatewayClientManager) ConnectedContainers() []string {
	var names []string
	m.clients.Range(func(key, value interface{}) bool {
		if name, ok := key.(string); ok {
			if client, ok := value.(*GatewayClient); ok {
				if client.State() == GatewayClientStateConnected {
					names = append(names, name)
				}
			}
		}
		return true
	})
	return names
}

// --- GatewayClientEventHandler implementation ---

func (m *GatewayClientManager) OnAgentEvent(containerName string, event GatewayAgentEvent) {
	log.Printf("openclaw-gw: agent event from %s: type=%s content=%s",
		containerName, event.Type, truncateString(event.Content, 100))

	// Forward to all subscribers
	m.eventSubs.Range(func(_, value interface{}) bool {
		if ch, ok := value.(chan GatewayAgentEvent); ok {
			select {
			case ch <- event:
			default:
				// Channel full, skip
			}
		}
		return true
	})

	// Forward to room if configured
	m.forwardEventToRoom(containerName, event)
}

func (m *GatewayClientManager) OnStateChange(containerName string, oldState, newState GatewayClientState) {
	log.Printf("openclaw-gw: container %s state changed: %s -> %s",
		containerName, oldState, newState)
}

func (m *GatewayClientManager) OnError(containerName string, err error) {
	log.Printf("openclaw-gw: container %s error: %v", containerName, err)
}

// --- Helper methods ---

func (m *GatewayClientManager) resolveGatewayHost(containerName string, port int) string {
	// Try to get from existing target base logic
	if targetBase, ok := m.handler.TargetBaseForContainer(containerName); ok {
		if u, err := url.Parse(targetBase); err == nil {
			return u.Hostname()
		}
	}

	// Fallback to container name (for Docker DNS) or localhost
	entry := m.handler.findEntry(containerName)
	if entry != nil && entry.DataDir != "" {
		// Check if using bridge network
		networkMode := strings.TrimSpace(entry.DataDir)
		if networkMode != "" && networkMode != "host" && !strings.HasPrefix(networkMode, "container:") {
			return containerName
		}
	}

	return "127.0.0.1"
}

func (m *GatewayClientManager) forwardEventToRoom(containerName string, event GatewayAgentEvent) {
	// Only forward message events with content
	if event.Type != "message" || strings.TrimSpace(event.Content) == "" {
		return
	}

	// Find container entry
	entry := m.handler.findEntry(containerName)
	if entry == nil || entry.TeamID == "" {
		return
	}

	// Find room for this team
	m.handler.mu.RLock()
	rooms, err := m.handler.loadRooms()
	m.handler.mu.RUnlock()
	if err != nil {
		log.Printf("openclaw-gw: failed to load rooms for forwarding: %v", err)
		return
	}

	var targetRoom *ClawRoomEntry
	for i := range rooms {
		if rooms[i].TeamID == entry.TeamID {
			targetRoom = &rooms[i]
			break
		}
	}

	if targetRoom == nil {
		return
	}

	// Create room message from agent event
	senderType := "worker"
	if normalizeRoleKind(entry.RoleKind) == "leader" {
		senderType = "leader"
	}

	senderName := workerDisplayName(*entry)
	message := newRoomMessage(
		*targetRoom,
		senderType,
		containerName,
		senderName,
		event.Content,
		map[string]string{
			"source":    "gateway-ws",
			"sessionId": event.SessionID,
			"agentId":   event.AgentID,
		},
	)

	if err := m.handler.appendRoomMessage(targetRoom.ID, message); err != nil {
		log.Printf("openclaw-gw: failed to forward event to room: %v", err)
	}
}

func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen-3] + "..."
}
