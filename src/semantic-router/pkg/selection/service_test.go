//go:build !windows && cgo

package selection

import (
	"context"
	"testing"
	"time"
)

func TestEmbeddingCache(t *testing.T) {
	cache := NewEmbeddingCache(100, time.Hour)

	// Test set and get
	embedding := []float32{0.1, 0.2, 0.3}
	cache.Set("test_query", embedding)

	got, ok := cache.Get("test_query")
	if !ok {
		t.Fatal("expected cache hit")
	}
	if len(got) != len(embedding) {
		t.Errorf("expected %d elements, got %d", len(embedding), len(got))
	}

	// Test cache miss
	_, ok = cache.Get("nonexistent")
	if ok {
		t.Error("expected cache miss")
	}

	// Test cache size
	if cache.Size() != 1 {
		t.Errorf("expected size 1, got %d", cache.Size())
	}
}

func TestEmbeddingCacheTTL(t *testing.T) {
	// Create cache with very short TTL
	cache := NewEmbeddingCache(100, 10*time.Millisecond)

	embedding := []float32{0.1, 0.2, 0.3}
	cache.Set("test_query", embedding)

	// Should hit immediately
	_, ok := cache.Get("test_query")
	if !ok {
		t.Fatal("expected cache hit")
	}

	// Wait for TTL to expire
	time.Sleep(20 * time.Millisecond)

	// Should miss after TTL
	_, ok = cache.Get("test_query")
	if ok {
		t.Error("expected cache miss after TTL")
	}
}

func TestEmbeddingCacheEviction(t *testing.T) {
	cache := NewEmbeddingCache(10, time.Hour)

	// Fill cache beyond capacity
	for i := 0; i < 15; i++ {
		embedding := []float32{float32(i)}
		cache.Set("query_"+string(rune('a'+i)), embedding)
	}

	// Some entries should have been evicted
	if cache.Size() > 10 {
		t.Errorf("expected size <= 10 after eviction, got %d", cache.Size())
	}
}

func TestMultiDecisionStats(t *testing.T) {
	stats := MultiDecisionStats{
		TotalDecisions:      3,
		TotalSelections:     100,
		TotalFeedbacks:      50,
		TotalSamplesLearned: 30,
		DecisionStats:       make(map[string]*DecisionStat),
	}

	stats.DecisionStats["coding"] = &DecisionStat{
		DecisionName:        "coding",
		TotalSelections:     50,
		TotalFeedbacks:      25,
		TotalSamplesLearned: 15,
		ValidCandidates:     []string{"gpt-4", "claude-3"},
	}

	if stats.TotalDecisions != 3 {
		t.Errorf("expected 3 decisions, got %d", stats.TotalDecisions)
	}

	ds := stats.DecisionStats["coding"]
	if ds.DecisionName != "coding" {
		t.Errorf("expected decision name 'coding', got '%s'", ds.DecisionName)
	}
	if len(ds.ValidCandidates) != 2 {
		t.Errorf("expected 2 candidates, got %d", len(ds.ValidCandidates))
	}
}

func TestFeedbackRequest(t *testing.T) {
	feedback := FeedbackRequest{
		DecisionName:  "coding_assistant",
		Query:         "How to implement quicksort?",
		SelectedModel: "gpt-4",
		Rating:        0.9,
		RequestID:     "req-123",
	}

	if feedback.DecisionName != "coding_assistant" {
		t.Errorf("expected decision name 'coding_assistant', got '%s'", feedback.DecisionName)
	}
	if feedback.Query != "How to implement quicksort?" {
		t.Errorf("expected query 'How to implement quicksort?', got '%s'", feedback.Query)
	}
	if feedback.Rating != 0.9 {
		t.Errorf("expected rating 0.9, got %f", feedback.Rating)
	}
}

func TestSanitizeFileName(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"simple", "simple"},
		{"with space", "with_space"},
		{"with/slash", "with_slash"},
		{"with:colon", "with_colon"},
		{"CamelCase", "CamelCase"},
		{"with-dash", "with-dash"},
		{"with_underscore", "with_underscore"},
		{"123numbers", "123numbers"},
		{"///", "___"},
		{"", "default"},
	}

	for _, tt := range tests {
		result := sanitizeFileName(tt.input)
		if result != tt.expected {
			t.Errorf("sanitizeFileName(%q) = %q, want %q", tt.input, result, tt.expected)
		}
	}
}

func TestDecisionSelectionServiceFilterCandidates(t *testing.T) {
	ds := &DecisionSelectionService{
		decisionName:    "test",
		validCandidates: []string{"model-a", "model-b"},
	}

	// Should filter to only valid candidates
	candidates := []string{"model-a", "model-b", "model-c"}
	filtered := ds.filterValidCandidates(candidates)

	if len(filtered) != 2 {
		t.Errorf("expected 2 filtered candidates, got %d", len(filtered))
	}

	// Should include model-a and model-b
	foundA, foundB := false, false
	for _, c := range filtered {
		if c == "model-a" {
			foundA = true
		}
		if c == "model-b" {
			foundB = true
		}
	}
	if !foundA || !foundB {
		t.Error("expected both model-a and model-b in filtered results")
	}
}

func TestDecisionSelectionServiceEmptyCandidates(t *testing.T) {
	ds := &DecisionSelectionService{
		decisionName:    "test",
		validCandidates: []string{}, // Empty valid candidates
	}

	// Should return all candidates when no valid candidates defined
	candidates := []string{"model-a", "model-b", "model-c"}
	filtered := ds.filterValidCandidates(candidates)

	if len(filtered) != 3 {
		t.Errorf("expected 3 candidates (all), got %d", len(filtered))
	}
}

func TestGlobalMultiDecisionService(t *testing.T) {
	// Clear global service
	SetGlobalMultiDecisionService(nil)

	// Should be nil initially
	svc := GetGlobalMultiDecisionService()
	if svc != nil {
		t.Error("expected nil global service")
	}

	// Create and set service
	config := ServiceConfig{
		ModelSelectionConfig: ModelSelectionConfig{
			Method: "knn",
			KNN:    DefaultKNNConfig(),
		},
		AutoSaveEnabled:    false,
		EmbeddingCacheSize: 100,
		EmbeddingCacheTTL:  time.Hour,
	}

	newSvc, err := NewMultiDecisionSelectionService(config)
	if err != nil {
		t.Fatalf("failed to create service: %v", err)
	}
	defer newSvc.Stop()

	SetGlobalMultiDecisionService(newSvc)

	// Should return the service now
	svc = GetGlobalMultiDecisionService()
	if svc == nil {
		t.Error("expected non-nil global service")
	}

	// Cleanup
	SetGlobalMultiDecisionService(nil)
}

func TestMultiDecisionServiceGetOrCreate(t *testing.T) {
	config := ServiceConfig{
		ModelSelectionConfig: ModelSelectionConfig{
			Method: "knn",
			KNN:    DefaultKNNConfig(),
		},
		AutoSaveEnabled:    false,
		EmbeddingCacheSize: 100,
		EmbeddingCacheTTL:  time.Hour,
	}

	svc, err := NewMultiDecisionSelectionService(config)
	if err != nil {
		t.Fatalf("failed to create service: %v", err)
	}
	defer svc.Stop()

	// Create first decision
	ds1, err := svc.GetOrCreateDecisionService("decision1", []string{"model-a", "model-b"})
	if err != nil {
		t.Fatalf("failed to create decision service: %v", err)
	}
	if ds1.decisionName != "decision1" {
		t.Errorf("expected decision name 'decision1', got '%s'", ds1.decisionName)
	}

	// Get same decision again should return same instance
	ds1Again, err := svc.GetOrCreateDecisionService("decision1", []string{"model-a", "model-b"})
	if err != nil {
		t.Fatalf("failed to get decision service: %v", err)
	}
	if ds1 != ds1Again {
		t.Error("expected same decision service instance")
	}

	// Create second decision
	ds2, err := svc.GetOrCreateDecisionService("decision2", []string{"model-c"})
	if err != nil {
		t.Fatalf("failed to create second decision service: %v", err)
	}
	if ds2.decisionName != "decision2" {
		t.Errorf("expected decision name 'decision2', got '%s'", ds2.decisionName)
	}

	// List decisions
	decisions := svc.ListDecisions()
	if len(decisions) != 2 {
		t.Errorf("expected 2 decisions, got %d", len(decisions))
	}
}

func TestMultiDecisionServiceStats(t *testing.T) {
	config := ServiceConfig{
		ModelSelectionConfig: ModelSelectionConfig{
			Method: "knn",
			KNN:    DefaultKNNConfig(),
		},
		AutoSaveEnabled:    false,
		EmbeddingCacheSize: 100,
		EmbeddingCacheTTL:  time.Hour,
	}

	svc, err := NewMultiDecisionSelectionService(config)
	if err != nil {
		t.Fatalf("failed to create service: %v", err)
	}
	defer svc.Stop()

	// Create decisions
	_, _ = svc.GetOrCreateDecisionService("coding", []string{"gpt-4", "claude"})
	_, _ = svc.GetOrCreateDecisionService("math", []string{"gpt-4"})

	stats := svc.GetStats()

	if stats.TotalDecisions != 2 {
		t.Errorf("expected 2 decisions, got %d", stats.TotalDecisions)
	}

	// Check decision-specific stats exist
	if _, ok := stats.DecisionStats["coding"]; !ok {
		t.Error("expected 'coding' decision stats")
	}
	if _, ok := stats.DecisionStats["math"]; !ok {
		t.Error("expected 'math' decision stats")
	}

	// Get specific decision stats
	codingStats, err := svc.GetDecisionStats("coding")
	if err != nil {
		t.Fatalf("failed to get coding stats: %v", err)
	}
	if codingStats.DecisionName != "coding" {
		t.Errorf("expected decision name 'coding', got '%s'", codingStats.DecisionName)
	}
}

func TestBackwardCompatibilitySelectionService(t *testing.T) {
	config := ServiceConfig{
		ModelSelectionConfig: ModelSelectionConfig{
			Method: "knn",
			KNN:    DefaultKNNConfig(),
		},
		AutoSaveEnabled:    false,
		EmbeddingCacheSize: 100,
		EmbeddingCacheTTL:  time.Hour,
	}

	// Use backward-compatible SelectionService
	svc, err := NewSelectionService(config)
	if err != nil {
		t.Fatalf("failed to create selection service: %v", err)
	}
	defer svc.Stop()

	// Should have underlying multi-decision service
	if svc.multi == nil {
		t.Error("expected underlying multi-decision service")
	}

	// GetStats should work
	stats := svc.GetStats()
	if stats.TotalSelections < 0 {
		t.Error("expected non-negative total selections")
	}

	// GetMultiDecisionService should return the underlying service
	multi := svc.GetMultiDecisionService()
	if multi == nil {
		t.Error("expected non-nil multi-decision service")
	}
}

func TestSelectWithContext(t *testing.T) {
	config := ServiceConfig{
		ModelSelectionConfig: ModelSelectionConfig{
			Method: "knn",
			KNN:    DefaultKNNConfig(),
		},
		AutoSaveEnabled:    false,
		EmbeddingCacheSize: 100,
		EmbeddingCacheTTL:  time.Hour,
	}

	svc, err := NewMultiDecisionSelectionService(config)
	if err != nil {
		t.Fatalf("failed to create service: %v", err)
	}
	defer svc.Stop()

	// Test with cancelled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	// Selection should still work (context not deeply used in KNN)
	// but we're testing the interface accepts context
	_, _ = svc.Select(ctx, "test_decision", "test query", []string{"model-a"})
}
