package handlers

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/nlauthor"
)

// NLAuthoringCapabilitiesHandler returns the current backend-owned NL contract status.
func NLAuthoringCapabilitiesHandler(cfg *config.Config, service *nlauthor.Service) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		writeJSON(w, service.Capabilities(effectiveReadonlyMode(cfg, r)))
	}
}

// NLAuthoringSchemaHandler returns the backend-owned NL schema manifest.
func NLAuthoringSchemaHandler(service *nlauthor.Service) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		writeJSON(w, service.Schema())
	}
}

func effectiveReadonlyMode(cfg *config.Config, r *http.Request) bool {
	readOnlyMode := cfg.ReadonlyMode
	if !readOnlyMode {
		if ac, ok := auth.AuthFromContext(r); ok && !ac.Perms[auth.PermConfigWrite] {
			readOnlyMode = true
		}
	}
	return readOnlyMode
}

func writeJSON(w http.ResponseWriter, payload interface{}) {
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(payload); err != nil {
		http.Error(w, "Failed to encode response", http.StatusInternalServerError)
	}
}

func writeJSONStatus(w http.ResponseWriter, statusCode int, payload interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(statusCode)
	if err := json.NewEncoder(w).Encode(payload); err != nil {
		http.Error(w, "Failed to encode response", http.StatusInternalServerError)
	}
}

func decodeJSONStrict(r *http.Request, target interface{}) error {
	decoder := json.NewDecoder(r.Body)
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(target); err != nil {
		return err
	}

	var trailing interface{}
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		if err == nil {
			return errors.New("request body must contain a single JSON value")
		}
		return err
	}
	return nil
}

func requestOwnerUserID(r *http.Request) string {
	if authContext, ok := auth.AuthFromContext(r); ok {
		return authContext.UserID
	}
	return ""
}
