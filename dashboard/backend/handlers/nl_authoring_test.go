package handlers

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/auth"
	"github.com/vllm-project/semantic-router/dashboard/backend/config"
	"github.com/vllm-project/semantic-router/dashboard/backend/nlauthor"
)

func TestNLAuthoringCapabilitiesHandlerReflectsReadonly(t *testing.T) {
	t.Parallel()

	t.Run("read users are limited to readonly capabilities", func(t *testing.T) {
		t.Parallel()

		req := httptest.NewRequest(http.MethodGet, "/api/builder/nl/capabilities", nil)
		req = req.WithContext(auth.WithAuthContext(req.Context(), auth.AuthContext{
			UserID: "user-read-1",
			Role:   auth.RoleRead,
			Perms: map[string]bool{
				auth.PermConfigRead: true,
			},
		}))

		recorder := httptest.NewRecorder()
		service := nlauthor.NewPreviewService(nlauthor.DefaultSchemaManifest())
		NLAuthoringCapabilitiesHandler(&config.Config{ReadonlyMode: false}, service).ServeHTTP(recorder, req)

		if recorder.Code != http.StatusOK {
			t.Fatalf("status = %d, want %d", recorder.Code, http.StatusOK)
		}

		var response nlauthor.Capabilities
		if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
			t.Fatalf("decode response error = %v", err)
		}
		if !response.Enabled {
			t.Fatalf("enabled = false, want true")
		}
		if !response.ReadonlyMode {
			t.Fatalf("readonlyMode = false, want true")
		}
		if response.SupportsApply {
			t.Fatalf("supportsApply = true, want false")
		}
		if !response.SupportsSessionAPI {
			t.Fatalf("supportsSessionApi = false, want true")
		}
		if !response.PlannerAvailable {
			t.Fatalf("plannerAvailable = false, want true")
		}
		if !response.SupportsClarification {
			t.Fatalf("supportsClarification = false, want true")
		}
		if len(response.SupportedSignalTypes) != 2 || response.SupportedSignalTypes[0] != "keyword" {
			t.Fatalf("supportedSignalTypes = %+v, want preview signal subset", response.SupportedSignalTypes)
		}
		if containsStringValue(response.SupportedSignalTypes, "pii") {
			t.Fatalf("supportedSignalTypes unexpectedly contain pii: %+v", response.SupportedSignalTypes)
		}
	})

	t.Run("write users keep apply capability while planner is preview-available", func(t *testing.T) {
		t.Parallel()

		req := httptest.NewRequest(http.MethodGet, "/api/builder/nl/capabilities", nil)
		req = req.WithContext(auth.WithAuthContext(req.Context(), auth.AuthContext{
			UserID: "user-write-1",
			Role:   auth.RoleWrite,
			Perms: map[string]bool{
				auth.PermConfigRead:  true,
				auth.PermConfigWrite: true,
			},
		}))

		recorder := httptest.NewRecorder()
		service := nlauthor.NewPreviewService(nlauthor.DefaultSchemaManifest())
		NLAuthoringCapabilitiesHandler(&config.Config{ReadonlyMode: false}, service).ServeHTTP(recorder, req)

		var response nlauthor.Capabilities
		if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
			t.Fatalf("decode response error = %v", err)
		}
		if response.ReadonlyMode {
			t.Fatalf("readonlyMode = true, want false")
		}
		if !response.SupportsApply {
			t.Fatalf("supportsApply = false, want true")
		}
		if !response.SupportsSessionAPI {
			t.Fatalf("supportsSessionApi = false, want true")
		}
		if !response.PlannerAvailable {
			t.Fatalf("plannerAvailable = false, want true")
		}
		if containsConstruct(response.SupportedConstructs, nlauthor.ConstructAlgorithm) {
			t.Fatalf("supportedConstructs unexpectedly contain algorithm: %+v", response.SupportedConstructs)
		}
		if len(response.SupportedPluginTypes) == 0 {
			t.Fatalf("supportedPluginTypes is empty, want preview plugin subset")
		}
	})
}

func TestNLAuthoringSchemaHandlerReturnsCanonicalManifest(t *testing.T) {
	t.Parallel()

	req := httptest.NewRequest(http.MethodGet, "/api/builder/nl/schema", nil)
	recorder := httptest.NewRecorder()

	service := nlauthor.NewService(nlauthor.DefaultSchemaManifest(), nlauthor.NewInMemorySessionStore(), nlauthor.UnavailablePlanner{}, time.Minute)
	NLAuthoringSchemaHandler(service).ServeHTTP(recorder, req)

	if recorder.Code != http.StatusOK {
		t.Fatalf("status = %d, want %d", recorder.Code, http.StatusOK)
	}

	var response nlauthor.SchemaManifest
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode response error = %v", err)
	}

	if response.Version != nlauthor.SchemaVersion {
		t.Fatalf("schema version = %q, want %q", response.Version, nlauthor.SchemaVersion)
	}

	if !containsType(response.Signals, "pii") {
		t.Fatalf("signals missing canonical pii signal type")
	}
	if containsType(response.Plugins, "pii") {
		t.Fatalf("plugins unexpectedly contain pii plugin type")
	}
	if !containsType(response.Plugins, "semantic_cache") {
		t.Fatalf("plugins missing semantic_cache")
	}
	if !containsType(response.Backends, "response_api") {
		t.Fatalf("backends missing response_api")
	}
	if containsType(response.Backends, "vector_store") {
		t.Fatalf("backends unexpectedly contain non-canonical vector_store type")
	}
}

func TestNLAuthoringSessionHandlers(t *testing.T) {
	t.Parallel()

	newService := func() *nlauthor.Service {
		return nlauthor.NewPreviewService(nlauthor.DefaultSchemaManifest())
	}

	withReadUser := func(req *http.Request, userID string) *http.Request {
		return req.WithContext(auth.WithAuthContext(req.Context(), auth.AuthContext{
			UserID: userID,
			Role:   auth.RoleRead,
			Perms: map[string]bool{
				auth.PermConfigRead: true,
			},
		}))
	}

	t.Run("read users can create sessions and receive ready planner turns", func(t *testing.T) {
		t.Parallel()

		service := newService()
		req := httptest.NewRequest(http.MethodPost, "/api/builder/nl/sessions", strings.NewReader(`{"context":{"baseDsl":"ROUTER demo {}"}}`))
		req = withReadUser(req, "reader-1")
		recorder := httptest.NewRecorder()

		NLAuthoringSessionsHandler(&config.Config{ReadonlyMode: false}, service).ServeHTTP(recorder, req)
		if recorder.Code != http.StatusCreated {
			t.Fatalf("create status = %d, want %d", recorder.Code, http.StatusCreated)
		}

		var createResponse nlauthor.SessionCreateResponse
		if err := json.Unmarshal(recorder.Body.Bytes(), &createResponse); err != nil {
			t.Fatalf("decode create response error = %v", err)
		}
		if createResponse.SessionID == "" {
			t.Fatal("sessionId is empty")
		}
		if !createResponse.Capabilities.ReadonlyMode {
			t.Fatalf("readonlyMode = false, want true")
		}
		if !createResponse.Capabilities.SupportsSessionAPI {
			t.Fatalf("supportsSessionApi = false, want true")
		}

		turnReq := httptest.NewRequest(http.MethodPost, "/api/builder/nl/sessions/"+createResponse.SessionID+"/turns", strings.NewReader(`{"prompt":"Create route support_route for model gpt-4o-mini","modeHint":"generate","context":{"symbols":{"models":["gpt-4o-mini"]}}}`))
		turnReq = withReadUser(turnReq, "reader-1")
		turnRecorder := httptest.NewRecorder()
		NLAuthoringSessionTurnsHandler(&config.Config{ReadonlyMode: false}, service).ServeHTTP(turnRecorder, turnReq)

		if turnRecorder.Code != http.StatusOK {
			t.Fatalf("turn status = %d, want %d", turnRecorder.Code, http.StatusOK)
		}

		var turnResponse nlauthor.TurnResponse
		if err := json.Unmarshal(turnRecorder.Body.Bytes(), &turnResponse); err != nil {
			t.Fatalf("decode turn response error = %v", err)
		}
		if turnResponse.SessionID != createResponse.SessionID {
			t.Fatalf("turn sessionId = %q, want %q", turnResponse.SessionID, createResponse.SessionID)
		}
		if turnResponse.Result.Status != nlauthor.PlannerStatusReady {
			t.Fatalf("turn status = %q, want %q", turnResponse.Result.Status, nlauthor.PlannerStatusReady)
		}
		if turnResponse.Result.IntentIR == nil || len(turnResponse.Result.IntentIR.Intents) != 1 {
			t.Fatal("turn intent IR is empty")
		}
	})

	t.Run("server rejects client supplied planner overrides", func(t *testing.T) {
		t.Parallel()

		service := newService()
		req := httptest.NewRequest(http.MethodPost, "/api/builder/nl/sessions", strings.NewReader(`{"endpoint":"https://example.com/v1"}`))
		req = withReadUser(req, "reader-2")
		recorder := httptest.NewRecorder()

		NLAuthoringSessionsHandler(&config.Config{ReadonlyMode: false}, service).ServeHTTP(recorder, req)
		if recorder.Code != http.StatusBadRequest {
			t.Fatalf("status = %d, want %d", recorder.Code, http.StatusBadRequest)
		}
		if !strings.Contains(recorder.Body.String(), "unknown field") {
			t.Fatalf("expected unknown field error, got %q", recorder.Body.String())
		}
	})

	t.Run("session owner is enforced across turn requests", func(t *testing.T) {
		t.Parallel()

		service := newService()
		createReq := httptest.NewRequest(http.MethodPost, "/api/builder/nl/sessions", strings.NewReader(`{}`))
		createReq = withReadUser(createReq, "reader-owner")
		createRecorder := httptest.NewRecorder()
		NLAuthoringSessionsHandler(&config.Config{ReadonlyMode: false}, service).ServeHTTP(createRecorder, createReq)

		var createResponse nlauthor.SessionCreateResponse
		if err := json.Unmarshal(createRecorder.Body.Bytes(), &createResponse); err != nil {
			t.Fatalf("decode create response error = %v", err)
		}

		turnReq := httptest.NewRequest(http.MethodPost, "/api/builder/nl/sessions/"+createResponse.SessionID+"/turns", strings.NewReader(`{"prompt":"Add route"}`))
		turnReq = withReadUser(turnReq, "reader-other")
		turnRecorder := httptest.NewRecorder()
		NLAuthoringSessionTurnsHandler(&config.Config{ReadonlyMode: false}, service).ServeHTTP(turnRecorder, turnReq)

		if turnRecorder.Code != http.StatusForbidden {
			t.Fatalf("status = %d, want %d", turnRecorder.Code, http.StatusForbidden)
		}
	})
}

func containsType(entries []nlauthor.TypeSchemaEntry, typeName string) bool {
	for _, entry := range entries {
		if entry.TypeName == typeName {
			return true
		}
	}
	return false
}

func containsConstruct(entries []nlauthor.ConstructKind, construct nlauthor.ConstructKind) bool {
	for _, entry := range entries {
		if entry == construct {
			return true
		}
	}
	return false
}

func containsStringValue(values []string, target string) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}
