package handlers

import (
	"encoding/json"
	"log"
	"net/http"
	"strings"

	"github.com/vllm-project/semantic-router/dashboard/backend/models"
	"github.com/vllm-project/semantic-router/dashboard/backend/services/tools"
)

// ToolsListHandler returns all available tools
func ToolsListHandler(registry *tools.Registry) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		allTools := registry.GetAllTools()

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"tools": allTools,
			"total": len(allTools),
		})
	}
}

// ToolsEnabledHandler returns only enabled tools
func ToolsEnabledHandler(registry *tools.Registry) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		enabledTools := registry.GetEnabledTools()

		// Convert to OpenAI format
		openAITools := make([]map[string]interface{}, 0, len(enabledTools))
		for _, tool := range enabledTools {
			openAITools = append(openAITools, tool.ToOpenAIFormat())
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"tools":        enabledTools,
			"openai_tools": openAITools,
			"total":        len(enabledTools),
		})
	}
}

// ToolExecuteHandler executes a tool
func ToolExecuteHandler(executor *tools.Executor) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req models.ToolExecutionRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			log.Printf("Error decoding tool execution request: %v", err)
			http.Error(w, "Invalid request body", http.StatusBadRequest)
			return
		}

		log.Printf("Executing tool: %s with args: %v", req.ToolID, req.Arguments)

		result := executor.Execute(r.Context(), &req)

		w.Header().Set("Content-Type", "application/json")
		if !result.Success {
			w.WriteHeader(http.StatusBadRequest)
		}
		json.NewEncoder(w).Encode(result)
	}
}

// ToolToggleHandler enables or disables a tool
func ToolToggleHandler(registry *tools.Registry) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPut && r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// Extract tool ID from path: /api/tools/{id}/toggle
		path := r.URL.Path
		parts := strings.Split(strings.Trim(path, "/"), "/")
		if len(parts) < 3 {
			http.Error(w, "Invalid path", http.StatusBadRequest)
			return
		}
		toolID := parts[2] // api/tools/{id}/toggle

		var req struct {
			Enabled bool `json:"enabled"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid request body", http.StatusBadRequest)
			return
		}

		if err := registry.SetToolEnabled(toolID, req.Enabled); err != nil {
			http.Error(w, err.Error(), http.StatusNotFound)
			return
		}

		log.Printf("Tool %s enabled: %v", toolID, req.Enabled)

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]interface{}{
			"tool_id": toolID,
			"enabled": req.Enabled,
		})
	}
}

// ToolGetHandler returns a specific tool
func ToolGetHandler(registry *tools.Registry) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// Extract tool ID from path: /api/tools/{id}
		path := r.URL.Path
		parts := strings.Split(strings.Trim(path, "/"), "/")
		if len(parts) < 3 {
			http.Error(w, "Invalid path", http.StatusBadRequest)
			return
		}
		toolID := parts[2]

		tool, builtinTool, found := registry.GetTool(toolID)
		if !found {
			http.Error(w, "Tool not found", http.StatusNotFound)
			return
		}

		var result *models.Tool
		if builtinTool != nil {
			result = &models.Tool{
				ID:          builtinTool.Name(),
				Name:        builtinTool.Name(),
				Description: builtinTool.Description(),
				Source:      models.ToolSourceBuiltin,
				Parameters:  builtinTool.Parameters(),
				Enabled:     registry.IsToolEnabled(toolID),
			}
		} else {
			result = tool
			result.Enabled = registry.IsToolEnabled(toolID)
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(result)
	}
}
