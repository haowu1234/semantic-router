package nlauthor

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestPlannerToolRegistryListsAndInvokesBuiltinBuilderTools(t *testing.T) {
	t.Parallel()

	registry := NewPlannerToolRegistry(
		NewBuiltinBuilderToolSource(DefaultSchemaManifest(), NewPreviewPlanner(DefaultSchemaManifest()).Support()),
	)
	session := Session{
		Context: SessionContext{
			BaseDSL: `SIGNAL keyword urgent_signal {
  operator: "any"
  keywords: ["urgent", "asap"]
}
ROUTE support_route {
  PRIORITY 100
  MODEL "gpt-4o-mini"
}`,
			Symbols: &SymbolSnapshot{Routes: []string{"support_route"}},
		},
	}

	definitions := registry.List(session, TurnRequest{}, DefaultToolPolicy())
	if len(definitions) == 0 {
		t.Fatal("registry definitions are empty")
	}

	result, err := registry.Invoke(
		t.Context(),
		session,
		TurnRequest{},
		"get_signal",
		json.RawMessage(`{"name":"urgent_signal"}`),
		DefaultToolPolicy(),
	)
	if err != nil {
		t.Fatalf("Invoke error = %v", err)
	}
	if !strings.Contains(result.Content, "urgent_signal") {
		t.Fatalf("tool content = %q, want urgent_signal snippet", result.Content)
	}
}

func TestPlannerToolRegistryRejectsDisallowedSource(t *testing.T) {
	t.Parallel()

	registry := NewPlannerToolRegistry(
		NewBuiltinBuilderToolSource(DefaultSchemaManifest(), NewPreviewPlanner(DefaultSchemaManifest()).Support()),
	)
	_, err := registry.Invoke(
		t.Context(),
		Session{},
		TurnRequest{},
		"list_symbols",
		nil,
		ToolPolicy{AllowedSources: []string{"mcp"}},
	)
	if err == nil {
		t.Fatal("Invoke error = nil, want unavailable tool error")
	}
}
