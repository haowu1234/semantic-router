package handlers

import (
	"encoding/json"
	"log"
	"net/http"
	"strings"
)

// --- Gateway WebSocket API Handlers ---

// GatewayWSStatusResponse represents the status of Gateway WebSocket connections
type GatewayWSStatusResponse struct {
	Enabled             bool                          `json:"enabled"`
	ConnectedContainers []GatewayWSContainerStatus    `json:"connectedContainers"`
	TotalConnected      int                           `json:"totalConnected"`
}

// GatewayWSContainerStatus represents a single container's Gateway WS status
type GatewayWSContainerStatus struct {
	ContainerName string `json:"containerName"`
	State         string `json:"state"`
	TeamID        string `json:"teamId,omitempty"`
	GatewayHost   string `json:"gatewayHost,omitempty"`
	GatewayPort   int    `json:"gatewayPort,omitempty"`
}

// GatewayWSConnectRequest represents a request to connect to a container's Gateway
type GatewayWSConnectRequest struct {
	ContainerName string `json:"containerName"`
}

// GatewayWSManager returns the Gateway WebSocket manager for this handler
func (h *OpenClawHandler) GatewayWSManager() *GatewayClientManager {
	h.mu.Lock()
	defer h.mu.Unlock()

	if h.gatewayWSManager == nil {
		h.gatewayWSManager = NewGatewayClientManager(h)
	}
	return h.gatewayWSManager
}

// InitGatewayWSManager initializes and optionally auto-connects to all running containers
func (h *OpenClawHandler) InitGatewayWSManager(autoConnect bool) error {
	manager := h.GatewayWSManager()

	if !autoConnect {
		return nil
	}

	// Get all running containers
	entries, err := h.loadRegistry()
	if err != nil {
		return err
	}

	for _, entry := range entries {
		if !h.containerRunning(entry.Name) {
			continue
		}
		if !h.gatewayReachable(entry.Name, entry.Port) {
			continue
		}

		if err := manager.ConnectContainer(entry.Name); err != nil {
			log.Printf("openclaw-gw: failed to auto-connect to %s: %v", entry.Name, err)
		}
	}

	return nil
}

// GatewayWSStatusHandler returns the status of Gateway WebSocket connections
func (h *OpenClawHandler) GatewayWSStatusHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		manager := h.GatewayWSManager()
		connectedNames := manager.ConnectedContainers()

		containerStatuses := make([]GatewayWSContainerStatus, 0, len(connectedNames))
		for _, name := range connectedNames {
			status := GatewayWSContainerStatus{
				ContainerName: name,
				State:         "connected",
			}

			if entry := h.findEntry(name); entry != nil {
				status.TeamID = entry.TeamID
				status.GatewayPort = entry.Port
			}

			if client, ok := manager.GetClient(name); ok {
				status.State = client.State().String()
				status.GatewayHost = client.config.GatewayHost
				status.GatewayPort = client.config.GatewayPort
			}

			containerStatuses = append(containerStatuses, status)
		}

		resp := GatewayWSStatusResponse{
			Enabled:             true,
			ConnectedContainers: containerStatuses,
			TotalConnected:      len(connectedNames),
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(resp); err != nil {
			log.Printf("openclaw: gateway-ws status encode error: %v", err)
		}
	}
}

// GatewayWSConnectHandler handles requests to connect/disconnect from container Gateways
func (h *OpenClawHandler) GatewayWSConnectHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodPost:
			h.handleGatewayWSConnect(w, r)
		case http.MethodDelete:
			h.handleGatewayWSDisconnect(w, r)
		default:
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		}
	}
}

func (h *OpenClawHandler) handleGatewayWSConnect(w http.ResponseWriter, r *http.Request) {
	var req GatewayWSConnectRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSONError(w, "Invalid request body", http.StatusBadRequest)
		return
	}

	containerName := sanitizeContainerName(req.ContainerName)
	if containerName == "" {
		writeJSONError(w, "containerName is required", http.StatusBadRequest)
		return
	}

	// Verify container exists
	entry := h.findEntry(containerName)
	if entry == nil {
		writeJSONError(w, "Container not found", http.StatusNotFound)
		return
	}

	// Check if container is running
	if !h.containerRunning(containerName) {
		writeJSONError(w, "Container is not running", http.StatusBadRequest)
		return
	}

	manager := h.GatewayWSManager()
	if err := manager.ConnectContainer(containerName); err != nil {
		writeJSONError(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(map[string]interface{}{
		"success":       true,
		"containerName": containerName,
		"message":       "Connected to Gateway WebSocket",
	}); err != nil {
		log.Printf("openclaw: gateway-ws connect encode error: %v", err)
	}
}

func (h *OpenClawHandler) handleGatewayWSDisconnect(w http.ResponseWriter, r *http.Request) {
	containerName := sanitizeContainerName(r.URL.Query().Get("containerName"))
	if containerName == "" {
		writeJSONError(w, "containerName query parameter is required", http.StatusBadRequest)
		return
	}

	manager := h.GatewayWSManager()
	manager.DisconnectContainer(containerName)

	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(map[string]interface{}{
		"success":       true,
		"containerName": containerName,
		"message":       "Disconnected from Gateway WebSocket",
	}); err != nil {
		log.Printf("openclaw: gateway-ws disconnect encode error: %v", err)
	}
}

// GatewayWSEventsHandler provides SSE stream of Gateway events
func (h *OpenClawHandler) GatewayWSEventsHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// Optional filter by container
		containerFilter := strings.TrimSpace(r.URL.Query().Get("containerName"))

		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.Header().Set("Access-Control-Allow-Origin", "*")

		flusher, ok := w.(http.Flusher)
		if !ok {
			http.Error(w, "Streaming not supported", http.StatusInternalServerError)
			return
		}

		manager := h.GatewayWSManager()
		subscriberID := generateRoomEntityID("gw-event-sub")
		eventChan := manager.SubscribeEvents(subscriberID)
		defer manager.UnsubscribeEvents(subscriberID)

		writeSSE(w, flusher, "connected", map[string]string{
			"subscriberId": subscriberID,
			"filter":       containerFilter,
		})

		ctx := r.Context()
		for {
			select {
			case <-ctx.Done():
				return
			case event, ok := <-eventChan:
				if !ok {
					return
				}
				// Apply container filter if specified
				if containerFilter != "" && !strings.EqualFold(event.AgentID, containerFilter) {
					// Check if the event came from the filtered container
					// (AgentID might be different from containerName in some cases)
					continue
				}
				writeSSE(w, flusher, "agent_event", event)
			}
		}
	}
}

// GatewayWSConnectAllHandler connects to all running containers
func (h *OpenClawHandler) GatewayWSConnectAllHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		h.mu.RLock()
		entries, err := h.loadRegistry()
		h.mu.RUnlock()
		if err != nil {
			writeJSONError(w, "Failed to load registry", http.StatusInternalServerError)
			return
		}

		manager := h.GatewayWSManager()
		connected := 0
		failed := 0
		results := make([]map[string]interface{}, 0)

		for _, entry := range entries {
			if !h.containerRunning(entry.Name) {
				continue
			}
			if !h.gatewayReachable(entry.Name, entry.Port) {
				results = append(results, map[string]interface{}{
					"containerName": entry.Name,
					"success":       false,
					"error":         "Gateway not reachable",
				})
				failed++
				continue
			}

			if err := manager.ConnectContainer(entry.Name); err != nil {
				results = append(results, map[string]interface{}{
					"containerName": entry.Name,
					"success":       false,
					"error":         err.Error(),
				})
				failed++
			} else {
				results = append(results, map[string]interface{}{
					"containerName": entry.Name,
					"success":       true,
				})
				connected++
			}
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(map[string]interface{}{
			"success":   failed == 0,
			"connected": connected,
			"failed":    failed,
			"results":   results,
		}); err != nil {
			log.Printf("openclaw: gateway-ws connect-all encode error: %v", err)
		}
	}
}
