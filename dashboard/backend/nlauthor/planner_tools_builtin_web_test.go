package nlauthor

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestBuiltinWebToolSourceFetchRawURL(t *testing.T) {
	t.Parallel()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("hello from remote"))
	}))
	defer server.Close()

	registry := NewPlannerToolRegistry(NewBuiltinWebToolSource())
	policy := DefaultToolPolicy()
	policy.AllowedSources = append(policy.AllowedSources, PlannerToolSourceBuiltinWeb)

	result, err := registry.Invoke(
		t.Context(),
		Session{},
		TurnRequest{},
		"fetch_raw_url",
		json.RawMessage(`{"url":"`+server.URL+`"}`),
		policy,
	)
	if err != nil {
		t.Fatalf("Invoke error = %v", err)
	}
	if result.Content == "" {
		t.Fatal("tool content is empty")
	}
}
