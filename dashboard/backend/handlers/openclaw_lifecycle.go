package handlers

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

// --- Start / Stop / Delete ---

func (h *OpenClawHandler) deleteContainerByName(name string) error {
	_ = h.containerRun("rm", "-f", name)
	_ = h.containerRun("volume", "rm", "openclaw-state-"+name)

	h.mu.Lock()
	defer h.mu.Unlock()

	entries, err := h.loadRegistry()
	if err != nil {
		return err
	}
	teams, teamErr := h.loadTeams()
	filtered := entries[:0]
	deletedTeamID := ""
	for _, e := range entries {
		if e.Name != name {
			filtered = append(filtered, e)
			continue
		}
		deletedTeamID = e.TeamID
	}
	if err := h.saveRegistry(filtered); err != nil {
		return err
	}

	if teamErr != nil || deletedTeamID == "" {
		return nil
	}
	teamsChanged := false
	for i := range teams {
		if teams[i].ID == deletedTeamID && teams[i].LeaderID == name {
			teams[i].LeaderID = ""
			teams[i].UpdatedAt = time.Now().UTC().Format(time.RFC3339)
			teamsChanged = true
		}
	}
	if teamsChanged {
		if err := h.saveTeams(teams); err != nil {
			log.Printf("openclaw: failed to clear leader mapping for deleted worker %s: %v", name, err)
		}
	}
	return nil
}

func (h *OpenClawHandler) StartHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if !h.canManageOpenClaw() {
			h.writeReadOnlyError(w)
			return
		}
		var req struct {
			ContainerName string `json:"containerName"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeJSONError(w, "invalid request body", http.StatusBadRequest)
			return
		}
		if req.ContainerName == "" {
			writeJSONError(w, "containerName required", http.StatusBadRequest)
			return
		}

		// Before restarting, refresh openclaw.json so that environment-derived
		// values (e.g. OPENCLAW_MODEL_BASE_URL) are propagated into the config
		// that the container bind-mounts. Without this, a restarted container
		// will keep using stale model URLs from the original provision.
		h.refreshOpenClawConfigBeforeStart(req.ContainerName)

		out, err := h.containerCombinedOutput("start", req.ContainerName)
		if err != nil {
			writeJSONError(w, fmt.Sprintf("Failed to start: %s (%v)", strings.TrimSpace(string(out)), err), http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(map[string]interface{}{
			"success": true,
			"message": fmt.Sprintf("Container %s started", req.ContainerName),
		}); err != nil {
			log.Printf("openclaw: start encode error: %v", err)
		}
	}
}

// refreshOpenClawConfigBeforeStart re-resolves the model base URL from the
// current environment and patches the on-disk openclaw.json so that a
// restarted container picks up any URL changes (e.g. after a split-runtime
// topology change or Envoy container rename).
func (h *OpenClawHandler) refreshOpenClawConfigBeforeStart(containerName string) {
	entry := h.findEntry(containerName)
	if entry == nil || strings.TrimSpace(entry.DataDir) == "" {
		return
	}
	configPath := filepath.Join(strings.TrimSpace(entry.DataDir), "openclaw.json")
	data, err := os.ReadFile(configPath)
	if err != nil {
		log.Printf("openclaw: refresh config: cannot read %s: %v", configPath, err)
		return
	}

	var cfg map[string]interface{}
	if err := json.Unmarshal(data, &cfg); err != nil {
		log.Printf("openclaw: refresh config: cannot parse %s: %v", configPath, err)
		return
	}

	// Resolve the current model base URL (reads OPENCLAW_MODEL_BASE_URL env
	// var which runtime_stack.py sets based on the Envoy container name).
	currentBaseURL := h.resolveOpenClawModelBaseURL()
	if currentBaseURL == "" {
		return
	}

	// Patch models.providers.vllm.baseUrl in the config.
	models, _ := cfg["models"].(map[string]interface{})
	if models == nil {
		return
	}
	providers, _ := models["providers"].(map[string]interface{})
	if providers == nil {
		return
	}
	vllm, _ := providers["vllm"].(map[string]interface{})
	if vllm == nil {
		return
	}

	oldURL, _ := vllm["baseUrl"].(string)
	if oldURL == currentBaseURL {
		return // No change needed
	}

	vllm["baseUrl"] = currentBaseURL
	updated, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		log.Printf("openclaw: refresh config: cannot marshal: %v", err)
		return
	}
	if err := os.WriteFile(configPath, updated, 0o644); err != nil {
		log.Printf("openclaw: refresh config: cannot write %s: %v", configPath, err)
		return
	}
	log.Printf("openclaw: refreshed modelBaseUrl in %s: %s -> %s", configPath, oldURL, currentBaseURL)
}

func (h *OpenClawHandler) StopHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if !h.canManageOpenClaw() {
			h.writeReadOnlyError(w)
			return
		}
		var req struct {
			ContainerName string `json:"containerName"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeJSONError(w, "invalid request body", http.StatusBadRequest)
			return
		}
		if req.ContainerName == "" {
			writeJSONError(w, "containerName required", http.StatusBadRequest)
			return
		}
		out, err := h.containerCombinedOutput("stop", req.ContainerName)
		if err != nil {
			writeJSONError(w, fmt.Sprintf("Failed to stop: %s (%v)", strings.TrimSpace(string(out)), err), http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(map[string]interface{}{
			"success": true,
			"message": fmt.Sprintf("Container %s stopped", req.ContainerName),
		}); err != nil {
			log.Printf("openclaw: stop encode error: %v", err)
		}
	}
}

func (h *OpenClawHandler) DeleteHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodDelete {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if !h.canManageOpenClaw() {
			h.writeReadOnlyError(w)
			return
		}
		name := strings.TrimPrefix(r.URL.Path, "/api/openclaw/containers/")
		if name == "" {
			writeJSONError(w, "container name required in path", http.StatusBadRequest)
			return
		}

		if err := h.deleteContainerByName(name); err != nil {
			log.Printf("openclaw: failed to save registry on delete: %v", err)
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(map[string]interface{}{
			"success": true,
			"message": fmt.Sprintf("Container %s removed", name),
		}); err != nil {
			log.Printf("openclaw: delete encode error: %v", err)
		}
	}
}

// --- Dynamic Proxy Lookup ---

// PortForContainer returns the port for a registered container (used by dynamic proxy).
func (h *OpenClawHandler) PortForContainer(name string) (int, bool) {
	h.mu.RLock()
	defer h.mu.RUnlock()
	entries, err := h.loadRegistry()
	if err != nil {
		return 0, false
	}
	for _, e := range entries {
		if e.Name == name {
			return e.Port, true
		}
	}
	return 0, false
}

// TargetBaseForContainer resolves the HTTP base URL for a registered container.
// Uses the container name as hostname (DNS resolution via bridge network).
func (h *OpenClawHandler) TargetBaseForContainer(name string) (string, bool) {
	port, ok := h.PortForContainer(name)
	if !ok {
		return "", false
	}
	return h.gatewayBaseURL(name, port), true
}
