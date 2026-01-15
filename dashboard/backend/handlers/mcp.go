package handlers

import (
	"encoding/json"
	"log"
	"net/http"
	"strings"

	"github.com/vllm-project/semantic-router/dashboard/backend/models"
	"github.com/vllm-project/semantic-router/dashboard/backend/services/mcp"
	"github.com/vllm-project/semantic-router/dashboard/backend/services/tools"
)

// MCPServersListHandler returns all MCP servers
func MCPServersListHandler(manager *mcp.Manager) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		servers := manager.GetServers()

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"servers": servers,
			"total":   len(servers),
		})
	}
}

// MCPServerAddHandler adds a new MCP server
func MCPServerAddHandler(manager *mcp.Manager, registry *tools.Registry) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var server models.MCPServer
		if err := json.NewDecoder(r.Body).Decode(&server); err != nil {
			log.Printf("Error decoding MCP server: %v", err)
			http.Error(w, "Invalid request body", http.StatusBadRequest)
			return
		}

		// Validate required fields
		if server.Name == "" {
			http.Error(w, "name is required", http.StatusBadRequest)
			return
		}
		if server.TransportType == "" {
			http.Error(w, "transport_type is required", http.StatusBadRequest)
			return
		}

		// Validate transport-specific fields
		switch server.TransportType {
		case models.MCPTransportStdio:
			if server.Command == "" {
				http.Error(w, "command is required for stdio transport", http.StatusBadRequest)
				return
			}
		case models.MCPTransportSSE, models.MCPTransportHTTP:
			if server.URL == "" {
				http.Error(w, "url is required for SSE/HTTP transport", http.StatusBadRequest)
				return
			}
		default:
			http.Error(w, "invalid transport_type", http.StatusBadRequest)
			return
		}

		log.Printf("Adding MCP server: %s (%s)", server.Name, server.TransportType)

		if err := manager.AddServer(r.Context(), &server); err != nil {
			log.Printf("Error adding MCP server: %v", err)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			json.NewEncoder(w).Encode(map[string]interface{}{
				"error":  err.Error(),
				"server": server,
			})
			return
		}

		// Register MCP tools to the tool registry
		mcpTools, err := manager.GetServerTools(server.ID)
		if err == nil {
			for _, mcpTool := range mcpTools {
				tool := mcpTool.ToTool(server.ID, server.Name)
				registry.RegisterMCPTool(server.ID, tool)
				log.Printf("Registered MCP tool: %s/%s", server.ID, tool.Name)
			}
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusCreated)
		json.NewEncoder(w).Encode(server)
	}
}

// MCPServerDeleteHandler deletes an MCP server
func MCPServerDeleteHandler(manager *mcp.Manager, registry *tools.Registry) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodDelete {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// Extract server ID from path: /api/mcp/servers/{id}
		serverID := extractServerID(r.URL.Path)
		if serverID == "" {
			http.Error(w, "Server ID required", http.StatusBadRequest)
			return
		}

		log.Printf("Deleting MCP server: %s", serverID)

		// Unregister MCP tools from the tool registry
		registry.UnregisterMCPTools(serverID)
		log.Printf("Unregistered MCP tools for server: %s", serverID)

		if err := manager.RemoveServer(serverID); err != nil {
			http.Error(w, err.Error(), http.StatusNotFound)
			return
		}

		w.WriteHeader(http.StatusNoContent)
	}
}

// MCPServerGetHandler returns a specific MCP server
func MCPServerGetHandler(manager *mcp.Manager) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		serverID := extractServerID(r.URL.Path)
		if serverID == "" {
			http.Error(w, "Server ID required", http.StatusBadRequest)
			return
		}

		server, found := manager.GetServer(serverID)
		if !found {
			http.Error(w, "Server not found", http.StatusNotFound)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(server)
	}
}

// MCPServerToolsHandler returns tools from a specific MCP server
func MCPServerToolsHandler(manager *mcp.Manager) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// Extract server ID from path: /api/mcp/servers/{id}/tools
		serverID := extractServerID(r.URL.Path)
		if serverID == "" {
			http.Error(w, "Server ID required", http.StatusBadRequest)
			return
		}

		tools, err := manager.GetServerTools(serverID)
		if err != nil {
			http.Error(w, err.Error(), http.StatusNotFound)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"tools": tools,
			"total": len(tools),
		})
	}
}

// MCPServerRefreshHandler refreshes tools from an MCP server
func MCPServerRefreshHandler(manager *mcp.Manager, registry *tools.Registry) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		serverID := extractServerID(r.URL.Path)
		if serverID == "" {
			http.Error(w, "Server ID required", http.StatusBadRequest)
			return
		}

		server, found := manager.GetServer(serverID)
		if !found {
			http.Error(w, "Server not found", http.StatusNotFound)
			return
		}

		log.Printf("Refreshing MCP server tools: %s", serverID)

		if err := manager.RefreshTools(r.Context(), serverID); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		// Re-register MCP tools (first unregister old ones)
		registry.UnregisterMCPTools(serverID)

		mcpTools, _ := manager.GetServerTools(serverID)
		for _, mcpTool := range mcpTools {
			tool := mcpTool.ToTool(serverID, server.Name)
			registry.RegisterMCPTool(serverID, tool)
		}
		log.Printf("Re-registered %d MCP tools for server: %s", len(mcpTools), serverID)

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"tools": mcpTools,
			"total": len(mcpTools),
		})
	}
}

// MCPServerReconnectHandler reconnects to an MCP server
func MCPServerReconnectHandler(manager *mcp.Manager, registry *tools.Registry) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		serverID := extractServerID(r.URL.Path)
		if serverID == "" {
			http.Error(w, "Server ID required", http.StatusBadRequest)
			return
		}

		log.Printf("Reconnecting MCP server: %s", serverID)

		// Unregister old tools first
		registry.UnregisterMCPTools(serverID)

		if err := manager.Reconnect(r.Context(), serverID); err != nil {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			json.NewEncoder(w).Encode(map[string]interface{}{
				"error": err.Error(),
			})
			return
		}

		server, _ := manager.GetServer(serverID)

		// Register new tools
		mcpTools, err := manager.GetServerTools(serverID)
		if err == nil {
			for _, mcpTool := range mcpTools {
				tool := mcpTool.ToTool(serverID, server.Name)
				registry.RegisterMCPTool(serverID, tool)
			}
			log.Printf("Registered %d MCP tools after reconnect: %s", len(mcpTools), serverID)
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(server)
	}
}

// extractServerID extracts the server ID from the URL path
func extractServerID(path string) string {
	// Path format: /api/mcp/servers/{id} or /api/mcp/servers/{id}/tools
	parts := strings.Split(strings.Trim(path, "/"), "/")
	// api/mcp/servers/{id}...
	if len(parts) >= 4 {
		return parts[3]
	}
	return ""
}
