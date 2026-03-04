package handlers

import (
	"encoding/json"
	"net/http"

	"github.com/vllm-project/semantic-router/dashboard/backend/config"
)

// SettingsResponse represents the dashboard settings returned to frontend
type SettingsResponse struct {
	ReadonlyMode  bool   `json:"readonlyMode"`  // Effective readonly status (base readonly AND no valid invite)
	Platform      string `json:"platform"`
	EnvoyURL      string `json:"envoyUrl"`      // Envoy proxy URL for evaluation endpoint
	HasInvite     bool   `json:"hasInvite"`     // Whether user has valid invite
	InviteEnabled bool   `json:"inviteEnabled"` // Whether invite feature is available
}

// SettingsHandler returns dashboard settings for frontend consumption
func SettingsHandler(cfg *config.Config) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// Check if user has valid invite
		hasInvite := cfg.InviteSecret != "" && HasValidInvite(r, cfg.InviteSecret)

		// Effective readonly: base readonly AND no valid invite
		effectiveReadonly := cfg.ReadonlyMode && !hasInvite

		response := SettingsResponse{
			ReadonlyMode:  effectiveReadonly,
			Platform:      cfg.Platform,
			EnvoyURL:      cfg.EnvoyURL,
			HasInvite:     hasInvite,
			InviteEnabled: cfg.ReadonlyMode && cfg.InviteSecret != "",
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(response); err != nil {
			http.Error(w, "Failed to encode response", http.StatusInternalServerError)
		}
	}
}
