package handlers

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestStaticFileServerServesSPADeepLinks(t *testing.T) {
	t.Helper()

	staticDir := t.TempDir()
	indexPath := filepath.Join(staticDir, "index.html")
	indexHTML := "<!doctype html><html><body>semantic-router-dashboard</body></html>"

	if err := os.WriteFile(indexPath, []byte(indexHTML), 0o644); err != nil {
		t.Fatalf("write index.html: %v", err)
	}

	handler := StaticFileServer(staticDir)

	for _, route := range []string{"/dashboard", "/login"} {
		req := httptest.NewRequest(http.MethodGet, route, nil)
		recorder := httptest.NewRecorder()

		handler.ServeHTTP(recorder, req)

		if recorder.Code != http.StatusOK {
			t.Fatalf("route %s: expected 200, got %d", route, recorder.Code)
		}

		body := recorder.Body.String()
		if !strings.Contains(body, "semantic-router-dashboard") {
			t.Fatalf("route %s: expected SPA index fallback, got %q", route, body)
		}
	}
}

func TestStaticFileServerRejectsAPIPaths(t *testing.T) {
	t.Helper()

	staticDir := t.TempDir()
	indexPath := filepath.Join(staticDir, "index.html")
	indexHTML := "<!doctype html><html><body>semantic-router-dashboard</body></html>"

	if err := os.WriteFile(indexPath, []byte(indexHTML), 0o644); err != nil {
		t.Fatalf("write index.html: %v", err)
	}

	handler := StaticFileServer(staticDir)
	req := httptest.NewRequest(http.MethodGet, "/api/auth/login", nil)
	recorder := httptest.NewRecorder()

	handler.ServeHTTP(recorder, req)

	if recorder.Code != http.StatusBadGateway {
		t.Fatalf("expected 502 for API path, got %d", recorder.Code)
	}

	if !strings.Contains(recorder.Body.String(), "Route not found") {
		t.Fatalf("expected proxy miss body, got %q", recorder.Body.String())
	}
}
