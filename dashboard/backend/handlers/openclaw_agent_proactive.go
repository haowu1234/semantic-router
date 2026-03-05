package handlers

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"
)

// --- Agent Proactive Messaging ---

// AgentMessageRequest represents a request for an agent to send a message proactively
type AgentMessageRequest struct {
	RoomID        string `json:"roomId"`
	ContainerName string `json:"containerName,omitempty"`
	Content       string `json:"content"`
	MessageType   string `json:"messageType,omitempty"` // "text", "notification", "action_result"
}

// AgentMessageResponse represents the response from sending an agent message
type AgentMessageResponse struct {
	Success   bool             `json:"success"`
	Message   *ClawRoomMessage `json:"message,omitempty"`
	Error     string           `json:"error,omitempty"`
	Timestamp string           `json:"timestamp"`
}

// SendAgentMessage allows an agent to send a message to a room proactively
// This can be triggered by:
// 1. Agent's own decision via Gateway WebSocket
// 2. External API call
// 3. Scheduled task completion
func (h *OpenClawHandler) SendAgentMessage(roomID, containerName, content, messageType string) (*ClawRoomMessage, error) {
	if strings.TrimSpace(content) == "" {
		return nil, fmt.Errorf("content is required")
	}

	roomID = sanitizeRoomID(roomID)
	if roomID == "" {
		return nil, fmt.Errorf("roomId is required")
	}

	containerName = sanitizeContainerName(containerName)
	if containerName == "" {
		return nil, fmt.Errorf("containerName is required")
	}

	// Verify room exists
	h.mu.RLock()
	rooms, err := h.loadRooms()
	h.mu.RUnlock()
	if err != nil {
		return nil, fmt.Errorf("failed to load rooms: %w", err)
	}

	room := findRoomByID(rooms, roomID)
	if room == nil {
		return nil, fmt.Errorf("room %q not found", roomID)
	}

	// Get agent info
	entry := h.findEntry(containerName)
	if entry == nil {
		return nil, fmt.Errorf("container %q not found", containerName)
	}

	// Verify agent belongs to the room's team
	if entry.TeamID != room.TeamID {
		return nil, fmt.Errorf("agent %q does not belong to room's team", containerName)
	}

	// Determine sender type based on role
	senderType := "worker"
	if normalizeRoleKind(entry.RoleKind) == "leader" {
		senderType = "leader"
	}

	// Create metadata for the message
	metadata := map[string]string{
		"source":      "agent_proactive",
		"messageType": messageType,
	}

	// Create the message
	message := newRoomMessage(
		*room,
		senderType,
		containerName,
		workerDisplayName(*entry),
		content,
		metadata,
	)

	// Save and broadcast
	if err := h.appendRoomMessage(room.ID, message); err != nil {
		return nil, fmt.Errorf("failed to save message: %w", err)
	}

	log.Printf("openclaw: agent %s sent proactive message to room %s", containerName, roomID)
	return &message, nil
}

// AgentMessageHandler handles HTTP requests for agents to send messages
func (h *OpenClawHandler) AgentMessageHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		if h.readOnly {
			writeJSONError(w, "Read-only mode enabled", http.StatusForbidden)
			return
		}

		var req AgentMessageRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeJSONError(w, "Invalid request body", http.StatusBadRequest)
			return
		}

		message, err := h.SendAgentMessage(req.RoomID, req.ContainerName, req.Content, req.MessageType)
		if err != nil {
			writeJSONError(w, err.Error(), http.StatusBadRequest)
			return
		}

		resp := AgentMessageResponse{
			Success:   true,
			Message:   message,
			Timestamp: time.Now().UTC().Format(time.RFC3339),
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		if err := json.NewEncoder(w).Encode(resp); err != nil {
			log.Printf("openclaw: agent message encode error: %v", err)
		}
	}
}

// --- Agent Task Notification ---

// AgentTaskNotification represents a notification about a task event
type AgentTaskNotification struct {
	ContainerName string                 `json:"containerName"`
	RoomID        string                 `json:"roomId"`
	TaskID        string                 `json:"taskId,omitempty"`
	EventType     string                 `json:"eventType"` // "started", "progress", "completed", "failed"
	Summary       string                 `json:"summary"`
	Details       map[string]interface{} `json:"details,omitempty"`
}

// NotifyTaskEvent sends a task event notification to the room
func (h *OpenClawHandler) NotifyTaskEvent(notification AgentTaskNotification) error {
	eventEmoji := map[string]string{
		"started":   "🚀",
		"progress":  "⏳",
		"completed": "✅",
		"failed":    "❌",
	}

	emoji := eventEmoji[notification.EventType]
	if emoji == "" {
		emoji = "📋"
	}

	content := fmt.Sprintf("%s **%s**: %s", emoji, notification.EventType, notification.Summary)

	_, err := h.SendAgentMessage(
		notification.RoomID,
		notification.ContainerName,
		content,
		"notification",
	)
	return err
}

// --- Gateway Event to Room Forwarding (Enhanced) ---

// ProcessGatewayAgentEvent processes an agent event from the Gateway and forwards to room
func (h *OpenClawHandler) ProcessGatewayAgentEvent(containerName string, event GatewayAgentEvent) error {
	entry := h.findEntry(containerName)
	if entry == nil || entry.TeamID == "" {
		return fmt.Errorf("container %q not found or has no team", containerName)
	}

	// Find room for this team
	h.mu.RLock()
	rooms, err := h.loadRooms()
	h.mu.RUnlock()
	if err != nil {
		return fmt.Errorf("failed to load rooms: %w", err)
	}

	var targetRoom *ClawRoomEntry
	for i := range rooms {
		if rooms[i].TeamID == entry.TeamID {
			targetRoom = &rooms[i]
			break
		}
	}

	if targetRoom == nil {
		return fmt.Errorf("no room found for team %q", entry.TeamID)
	}

	// Process different event types
	switch event.Type {
	case "message":
		// Agent text output - forward to room
		if strings.TrimSpace(event.Content) == "" {
			return nil
		}

		senderType := "worker"
		if normalizeRoleKind(entry.RoleKind) == "leader" {
			senderType = "leader"
		}

		message := newRoomMessage(
			*targetRoom,
			senderType,
			containerName,
			workerDisplayName(*entry),
			event.Content,
			map[string]string{
				"source":     "gateway_ws",
				"sessionKey": event.SessionKey,
				"runId":      event.RunID,
			},
		)

		return h.appendRoomMessage(targetRoom.ID, message)

	case "tool":
		// Agent tool usage - optionally notify room
		if event.ToolName == "" {
			return nil
		}

		// Create a system notification about tool usage
		toolMsg := fmt.Sprintf("🔧 Using tool: **%s**", event.ToolName)
		if event.ToolOutput != "" && len(event.ToolOutput) < 200 {
			toolMsg += fmt.Sprintf("\n```\n%s\n```", event.ToolOutput)
		}

		message := newRoomMessage(
			*targetRoom,
			"system",
			"gateway-ws",
			"System",
			toolMsg,
			map[string]string{
				"source":     "gateway_ws_tool",
				"toolName":   event.ToolName,
				"sessionKey": event.SessionKey,
			},
		)

		return h.appendRoomMessage(targetRoom.ID, message)

	default:
		// Unknown event type, log but don't error
		log.Printf("openclaw: unknown gateway event type %q from %s", event.Type, containerName)
		return nil
	}
}

// --- Agent Session Management ---

// AgentSession tracks an active agent session
type AgentSession struct {
	SessionID     string    `json:"sessionId"`
	ContainerName string    `json:"containerName"`
	RoomID        string    `json:"roomId"`
	StartedAt     time.Time `json:"startedAt"`
	LastActivity  time.Time `json:"lastActivity"`
	Status        string    `json:"status"` // "active", "idle", "completed"
}

// GetActiveSessions returns all active agent sessions for a room
func (h *OpenClawHandler) GetActiveSessions(roomID string) []AgentSession {
	// This is a placeholder - in full implementation, sessions would be
	// tracked via the Gateway WebSocket connection state
	manager := h.GatewayWSManager()
	connectedContainers := manager.ConnectedContainers()

	sessions := make([]AgentSession, 0)
	for _, containerName := range connectedContainers {
		entry := h.findEntry(containerName)
		if entry == nil {
			continue
		}

		// Check if container's team matches the room's team
		h.mu.RLock()
		rooms, err := h.loadRooms()
		h.mu.RUnlock()
		if err != nil {
			continue
		}

		room := findRoomByID(rooms, roomID)
		if room == nil || room.TeamID != entry.TeamID {
			continue
		}

		client, ok := manager.GetClient(containerName)
		if !ok {
			continue
		}

		sessions = append(sessions, AgentSession{
			SessionID:     fmt.Sprintf("session-%s", containerName),
			ContainerName: containerName,
			RoomID:        roomID,
			StartedAt:     time.Now(), // Would be tracked properly in full implementation
			LastActivity:  client.lastActivity,
			Status:        "active",
		})
	}

	return sessions
}

// ActiveSessionsHandler returns active sessions for a room
func (h *OpenClawHandler) ActiveSessionsHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		roomID := sanitizeRoomID(r.URL.Query().Get("roomId"))
		if roomID == "" {
			writeJSONError(w, "roomId query parameter required", http.StatusBadRequest)
			return
		}

		sessions := h.GetActiveSessions(roomID)

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(sessions); err != nil {
			log.Printf("openclaw: active sessions encode error: %v", err)
		}
	}
}
