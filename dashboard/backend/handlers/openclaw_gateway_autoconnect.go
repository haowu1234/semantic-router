package handlers

import (
	"encoding/json"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"
)

// --- Auto-Connect Management ---

// GatewayAutoConnectConfig configures automatic Gateway WebSocket connection behavior
type GatewayAutoConnectConfig struct {
	Enabled          bool          `json:"enabled"`
	ScanInterval     time.Duration `json:"scanInterval"`
	HealthCheckDelay time.Duration `json:"healthCheckDelay"`
}

var defaultAutoConnectConfig = GatewayAutoConnectConfig{
	Enabled:          true,
	ScanInterval:     30 * time.Second,
	HealthCheckDelay: 5 * time.Second,
}

// GatewayAutoConnector manages automatic connection to new/restarted containers
type GatewayAutoConnector struct {
	handler       *OpenClawHandler
	config        GatewayAutoConnectConfig
	stopChan      chan struct{}
	running       bool
	mu            sync.Mutex
	lastScan      time.Time
	pendingChecks sync.Map // containerName -> time.Time (scheduled check time)
}

// NewGatewayAutoConnector creates a new auto-connector
func NewGatewayAutoConnector(handler *OpenClawHandler, config GatewayAutoConnectConfig) *GatewayAutoConnector {
	return &GatewayAutoConnector{
		handler:  handler,
		config:   config,
		stopChan: make(chan struct{}),
	}
}

// Start begins the auto-connection scanner
func (ac *GatewayAutoConnector) Start() {
	ac.mu.Lock()
	if ac.running {
		ac.mu.Unlock()
		return
	}
	ac.running = true
	ac.mu.Unlock()

	go ac.scanLoop()
	log.Printf("openclaw-gw: auto-connector started (interval: %s)", ac.config.ScanInterval)
}

// Stop stops the auto-connection scanner
func (ac *GatewayAutoConnector) Stop() {
	ac.mu.Lock()
	defer ac.mu.Unlock()

	if !ac.running {
		return
	}

	close(ac.stopChan)
	ac.running = false
	log.Printf("openclaw-gw: auto-connector stopped")
}

func (ac *GatewayAutoConnector) scanLoop() {
	ticker := time.NewTicker(ac.config.ScanInterval)
	defer ticker.Stop()

	// Initial scan
	ac.performScan()

	for {
		select {
		case <-ac.stopChan:
			return
		case <-ticker.C:
			ac.performScan()
		}
	}
}

func (ac *GatewayAutoConnector) performScan() {
	ac.lastScan = time.Now()

	entries, err := ac.handler.loadRegistry()
	if err != nil {
		log.Printf("openclaw-gw: auto-connect scan failed to load registry: %v", err)
		return
	}

	manager := ac.handler.GatewayWSManager()

	for _, entry := range entries {
		// Skip if already connected
		if _, exists := manager.GetClient(entry.Name); exists {
			continue
		}

		// Check if container is running
		if !ac.handler.containerRunning(entry.Name) {
			continue
		}

		// Schedule a delayed health check (gives container time to fully start)
		checkTime := time.Now().Add(ac.config.HealthCheckDelay)
		ac.pendingChecks.Store(entry.Name, checkTime)

		go ac.delayedConnect(entry.Name, checkTime)
	}
}

func (ac *GatewayAutoConnector) delayedConnect(containerName string, scheduledTime time.Time) {
	// Wait until scheduled time
	waitDuration := time.Until(scheduledTime)
	if waitDuration > 0 {
		time.Sleep(waitDuration)
	}

	// Verify this is still the pending check
	if stored, ok := ac.pendingChecks.Load(containerName); ok {
		if stored.(time.Time) != scheduledTime {
			// A newer check was scheduled, skip this one
			return
		}
	}
	ac.pendingChecks.Delete(containerName)

	// Get entry info
	entry := ac.handler.findEntry(containerName)
	if entry == nil {
		return
	}

	// Check gateway health before connecting
	if !ac.handler.gatewayReachable(containerName, entry.Port) {
		log.Printf("openclaw-gw: auto-connect skipped %s (gateway not reachable)", containerName)
		return
	}

	// Connect
	manager := ac.handler.GatewayWSManager()
	if err := manager.ConnectContainer(containerName); err != nil {
		log.Printf("openclaw-gw: auto-connect failed for %s: %v", containerName, err)
	} else {
		log.Printf("openclaw-gw: auto-connected to %s", containerName)
	}
}

// ScheduleCheck schedules an immediate check for a specific container
// (useful after container start/restart)
func (ac *GatewayAutoConnector) ScheduleCheck(containerName string) {
	checkTime := time.Now().Add(ac.config.HealthCheckDelay)
	ac.pendingChecks.Store(containerName, checkTime)
	go ac.delayedConnect(containerName, checkTime)
}

// --- Lifecycle Hooks ---

// OnContainerStarted should be called when a container is started
func (h *OpenClawHandler) OnContainerStarted(containerName string) {
	h.mu.Lock()
	autoConnector := h.gatewayAutoConnector
	h.mu.Unlock()

	if autoConnector != nil {
		autoConnector.ScheduleCheck(containerName)
	}
}

// OnContainerStopped should be called when a container is stopped
func (h *OpenClawHandler) OnContainerStopped(containerName string) {
	if h.gatewayWSManager != nil {
		h.gatewayWSManager.DisconnectContainer(containerName)
	}
}

// --- Enhanced Handler Methods ---

// StartGatewayAutoConnect starts automatic Gateway WebSocket connection management
func (h *OpenClawHandler) StartGatewayAutoConnect(config *GatewayAutoConnectConfig) {
	h.mu.Lock()
	defer h.mu.Unlock()

	if h.gatewayAutoConnector != nil {
		return
	}

	cfg := defaultAutoConnectConfig
	if config != nil {
		cfg = *config
	}

	h.gatewayAutoConnector = NewGatewayAutoConnector(h, cfg)
	h.gatewayAutoConnector.Start()
}

// StopGatewayAutoConnect stops automatic Gateway WebSocket connection management
func (h *OpenClawHandler) StopGatewayAutoConnect() {
	h.mu.Lock()
	defer h.mu.Unlock()

	if h.gatewayAutoConnector != nil {
		h.gatewayAutoConnector.Stop()
		h.gatewayAutoConnector = nil
	}
}

// --- HTTP Handler for Auto-Connect Control ---

// GatewayAutoConnectHandler handles auto-connect configuration
func (h *OpenClawHandler) GatewayAutoConnectHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			h.handleGetAutoConnectStatus(w, r)
		case http.MethodPost:
			h.handleStartAutoConnect(w, r)
		case http.MethodDelete:
			h.handleStopAutoConnect(w, r)
		default:
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		}
	}
}

func (h *OpenClawHandler) handleGetAutoConnectStatus(w http.ResponseWriter, _ *http.Request) {
	h.mu.Lock()
	autoConnector := h.gatewayAutoConnector
	h.mu.Unlock()

	running := autoConnector != nil && autoConnector.running
	var lastScan string
	if autoConnector != nil {
		lastScan = autoConnector.lastScan.Format(time.RFC3339)
	}

	resp := map[string]interface{}{
		"enabled":  running,
		"lastScan": lastScan,
	}

	if autoConnector != nil {
		resp["config"] = map[string]interface{}{
			"scanInterval":     autoConnector.config.ScanInterval.String(),
			"healthCheckDelay": autoConnector.config.HealthCheckDelay.String(),
		}
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(resp); err != nil {
		log.Printf("openclaw: auto-connect status encode error: %v", err)
	}
}

func (h *OpenClawHandler) handleStartAutoConnect(w http.ResponseWriter, r *http.Request) {
	var req struct {
		ScanInterval     string `json:"scanInterval"`
		HealthCheckDelay string `json:"healthCheckDelay"`
	}

	config := defaultAutoConnectConfig

	if r.ContentLength > 0 {
		if err := json.NewDecoder(r.Body).Decode(&req); err == nil {
			if interval, err := time.ParseDuration(req.ScanInterval); err == nil && interval > 0 {
				config.ScanInterval = interval
			}
			if delay, err := time.ParseDuration(req.HealthCheckDelay); err == nil && delay > 0 {
				config.HealthCheckDelay = delay
			}
		}
	}

	h.StartGatewayAutoConnect(&config)

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(map[string]interface{}{
		"success": true,
		"message": "Auto-connect started",
		"config": map[string]string{
			"scanInterval":     config.ScanInterval.String(),
			"healthCheckDelay": config.HealthCheckDelay.String(),
		},
	}); err != nil {
		log.Printf("openclaw: start auto-connect encode error: %v", err)
	}
}

func (h *OpenClawHandler) handleStopAutoConnect(w http.ResponseWriter, _ *http.Request) {
	h.StopGatewayAutoConnect()

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(map[string]interface{}{
		"success": true,
		"message": "Auto-connect stopped",
	}); err != nil {
		log.Printf("openclaw: stop auto-connect encode error: %v", err)
	}
}

// --- Room-Agent Binding ---

// RoomAgentBinding represents a binding between a room and connected agents
type RoomAgentBinding struct {
	RoomID       string   `json:"roomId"`
	TeamID       string   `json:"teamId"`
	BoundAgents  []string `json:"boundAgents"`  // Container names
	ActiveAgents []string `json:"activeAgents"` // Currently connected via Gateway WS
}

// GetRoomAgentBindings returns agent bindings for all rooms
func (h *OpenClawHandler) GetRoomAgentBindings() ([]RoomAgentBinding, error) {
	h.mu.RLock()
	rooms, err := h.loadRooms()
	entries, entryErr := h.loadRegistry()
	h.mu.RUnlock()

	if err != nil {
		return nil, err
	}
	if entryErr != nil {
		return nil, entryErr
	}

	manager := h.GatewayWSManager()
	connectedMap := make(map[string]bool)
	for _, name := range manager.ConnectedContainers() {
		connectedMap[name] = true
	}

	bindings := make([]RoomAgentBinding, 0, len(rooms))
	for _, room := range rooms {
		binding := RoomAgentBinding{
			RoomID:       room.ID,
			TeamID:       room.TeamID,
			BoundAgents:  make([]string, 0),
			ActiveAgents: make([]string, 0),
		}

		for _, entry := range entries {
			if entry.TeamID != room.TeamID {
				continue
			}
			binding.BoundAgents = append(binding.BoundAgents, entry.Name)
			if connectedMap[entry.Name] {
				binding.ActiveAgents = append(binding.ActiveAgents, entry.Name)
			}
		}

		bindings = append(bindings, binding)
	}

	return bindings, nil
}

// RoomAgentBindingsHandler returns agent bindings for rooms
func (h *OpenClawHandler) RoomAgentBindingsHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		roomID := strings.TrimSpace(r.URL.Query().Get("roomId"))

		bindings, err := h.GetRoomAgentBindings()
		if err != nil {
			writeJSONError(w, err.Error(), http.StatusInternalServerError)
			return
		}

		// Filter by roomId if specified
		if roomID != "" {
			filtered := make([]RoomAgentBinding, 0)
			for _, b := range bindings {
				if b.RoomID == roomID {
					filtered = append(filtered, b)
				}
			}
			bindings = filtered
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(bindings); err != nil {
			log.Printf("openclaw: room bindings encode error: %v", err)
		}
	}
}
