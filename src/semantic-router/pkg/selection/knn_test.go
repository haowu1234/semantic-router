//go:build !windows && cgo

package selection

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

func TestKNNSelector_BasicSelection(t *testing.T) {
	config := DefaultKNNConfig()
	config.K = 3
	config.UseHNSW = false // Use brute force for deterministic testing

	selector := NewKNNSelector(config)

	// Add training samples
	// Cluster 1: queries that should route to "gpt-4"
	selector.AddTrainingSample([]float32{0.9, 0.1, 0.0}, "gpt-4")
	selector.AddTrainingSample([]float32{0.85, 0.15, 0.0}, "gpt-4")
	selector.AddTrainingSample([]float32{0.8, 0.2, 0.0}, "gpt-4")

	// Cluster 2: queries that should route to "claude-3"
	selector.AddTrainingSample([]float32{0.1, 0.9, 0.0}, "claude-3")
	selector.AddTrainingSample([]float32{0.15, 0.85, 0.0}, "claude-3")
	selector.AddTrainingSample([]float32{0.2, 0.8, 0.0}, "claude-3")

	// Cluster 3: queries that should route to "llama-3"
	selector.AddTrainingSample([]float32{0.0, 0.1, 0.9}, "llama-3")
	selector.AddTrainingSample([]float32{0.0, 0.15, 0.85}, "llama-3")
	selector.AddTrainingSample([]float32{0.0, 0.2, 0.8}, "llama-3")

	ctx := context.Background()

	tests := []struct {
		name          string
		embedding     []float32
		candidates    []string
		expectedModel string
	}{
		{
			name:          "query similar to gpt-4 cluster",
			embedding:     []float32{0.88, 0.12, 0.0},
			candidates:    []string{"gpt-4", "claude-3", "llama-3"},
			expectedModel: "gpt-4",
		},
		{
			name:          "query similar to claude-3 cluster",
			embedding:     []float32{0.12, 0.88, 0.0},
			candidates:    []string{"gpt-4", "claude-3", "llama-3"},
			expectedModel: "claude-3",
		},
		{
			name:          "query similar to llama-3 cluster",
			embedding:     []float32{0.0, 0.12, 0.88},
			candidates:    []string{"gpt-4", "claude-3", "llama-3"},
			expectedModel: "llama-3",
		},
		{
			name:          "limited candidates - only gpt-4 available",
			embedding:     []float32{0.12, 0.88, 0.0}, // Would normally choose claude-3
			candidates:    []string{"gpt-4"},
			expectedModel: "gpt-4",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, err := selector.Select(ctx, &SelectionContext{
				QueryEmbedding: tt.embedding,
				Candidates:     tt.candidates,
			})
			if err != nil {
				t.Fatalf("Select failed: %v", err)
			}
			if result.SelectedModel != tt.expectedModel {
				t.Errorf("expected %s, got %s (scores: %v)", tt.expectedModel, result.SelectedModel, result.Scores)
			}
			if result.SelectorName != "knn" {
				t.Errorf("expected selector name 'knn', got %s", result.SelectorName)
			}
		})
	}
}

func TestKNNSelector_WithHNSW(t *testing.T) {
	config := DefaultKNNConfig()
	config.K = 3
	config.UseHNSW = true

	selector := NewKNNSelector(config)

	// Add training samples
	for i := 0; i < 100; i++ {
		// Create samples in different clusters
		if i < 33 {
			selector.AddTrainingSample([]float32{0.9, 0.1, float32(i) * 0.001}, "model-a")
		} else if i < 66 {
			selector.AddTrainingSample([]float32{0.1, 0.9, float32(i) * 0.001}, "model-b")
		} else {
			selector.AddTrainingSample([]float32{0.5, 0.5, float32(i) * 0.001}, "model-c")
		}
	}

	ctx := context.Background()

	result, err := selector.Select(ctx, &SelectionContext{
		QueryEmbedding: []float32{0.85, 0.15, 0.05},
		Candidates:     []string{"model-a", "model-b", "model-c"},
	})
	if err != nil {
		t.Fatalf("Select failed: %v", err)
	}

	if result.SelectedModel != "model-a" {
		t.Errorf("expected model-a, got %s", result.SelectedModel)
	}

	// Check metadata
	if _, ok := result.Metadata["use_hnsw"]; !ok {
		t.Error("expected use_hnsw in metadata")
	}
}

func TestKNNSelector_SaveAndLoad(t *testing.T) {
	// Create temp directory
	tmpDir, err := os.MkdirTemp("", "knn_test")
	if err != nil {
		t.Fatalf("failed to create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	modelPath := filepath.Join(tmpDir, "knn_model.json")

	// Create and train selector
	config := DefaultKNNConfig()
	config.K = 3
	selector := NewKNNSelector(config)

	selector.AddTrainingSample([]float32{0.9, 0.1, 0.0}, "gpt-4")
	selector.AddTrainingSample([]float32{0.1, 0.9, 0.0}, "claude-3")
	selector.AddTrainingSample([]float32{0.0, 0.1, 0.9}, "llama-3")

	// Save model
	if err := selector.SaveToFile(modelPath); err != nil {
		t.Fatalf("SaveToFile failed: %v", err)
	}

	// Load into new selector
	loadedSelector := NewKNNSelector(config)
	if err := loadedSelector.LoadFromFile(modelPath); err != nil {
		t.Fatalf("LoadFromFile failed: %v", err)
	}

	// Verify loaded data
	if loadedSelector.Size() != 3 {
		t.Errorf("expected 3 samples, got %d", loadedSelector.Size())
	}

	candidates := loadedSelector.Candidates()
	if len(candidates) != 3 {
		t.Errorf("expected 3 candidates, got %d", len(candidates))
	}

	// Test selection with loaded model
	ctx := context.Background()
	result, err := loadedSelector.Select(ctx, &SelectionContext{
		QueryEmbedding: []float32{0.88, 0.12, 0.0},
		Candidates:     []string{"gpt-4", "claude-3", "llama-3"},
	})
	if err != nil {
		t.Fatalf("Select failed: %v", err)
	}

	if result.SelectedModel != "gpt-4" {
		t.Errorf("expected gpt-4, got %s", result.SelectedModel)
	}
}

func TestKNNSelector_Update(t *testing.T) {
	config := DefaultKNNConfig()
	selector := NewKNNSelector(config)

	ctx := context.Background()

	// Update with feedback
	err := selector.Update(ctx, &SelectionFeedback{
		QueryEmbedding: []float32{0.9, 0.1, 0.0},
		Winner:         "gpt-4",
	})
	if err != nil {
		t.Fatalf("Update failed: %v", err)
	}

	if selector.Size() != 1 {
		t.Errorf("expected 1 sample after update, got %d", selector.Size())
	}

	// Update with positive rating (no winner)
	err = selector.Update(ctx, &SelectionFeedback{
		QueryEmbedding: []float32{0.1, 0.9, 0.0},
		SelectedModel:  "claude-3",
		Rating:         0.8,
	})
	if err != nil {
		t.Fatalf("Update failed: %v", err)
	}

	if selector.Size() != 2 {
		t.Errorf("expected 2 samples after update, got %d", selector.Size())
	}
}

func TestKNNSelector_NoTrainingData(t *testing.T) {
	config := DefaultKNNConfig()
	selector := NewKNNSelector(config)

	ctx := context.Background()

	// Should fall back to first candidate
	result, err := selector.Select(ctx, &SelectionContext{
		QueryEmbedding: []float32{0.5, 0.5, 0.0},
		Candidates:     []string{"model-a", "model-b"},
	})
	if err != nil {
		t.Fatalf("Select failed: %v", err)
	}

	if result.SelectedModel != "model-a" {
		t.Errorf("expected fallback to model-a, got %s", result.SelectedModel)
	}

	if result.Metadata["fallback"] != "no_training_data" {
		t.Error("expected fallback metadata")
	}
}

func TestKNNSelector_EmptyEmbedding(t *testing.T) {
	config := DefaultKNNConfig()
	selector := NewKNNSelector(config)
	selector.AddTrainingSample([]float32{0.5, 0.5}, "model-a")

	ctx := context.Background()

	_, err := selector.Select(ctx, &SelectionContext{
		QueryEmbedding: []float32{},
		Candidates:     []string{"model-a"},
	})
	if err == nil {
		t.Error("expected error for empty embedding")
	}
}

func TestKNNSelector_NoCandidates(t *testing.T) {
	config := DefaultKNNConfig()
	selector := NewKNNSelector(config)

	ctx := context.Background()

	_, err := selector.Select(ctx, &SelectionContext{
		QueryEmbedding: []float32{0.5, 0.5},
		Candidates:     []string{},
	})
	if err == nil {
		t.Error("expected error for no candidates")
	}
}

func TestCosineSimilarity(t *testing.T) {
	tests := []struct {
		name     string
		a, b     []float32
		expected float32
		epsilon  float32
	}{
		{
			name:     "identical vectors",
			a:        []float32{1, 0, 0},
			b:        []float32{1, 0, 0},
			expected: 1.0,
			epsilon:  0.01,
		},
		{
			name:     "orthogonal vectors",
			a:        []float32{1, 0, 0},
			b:        []float32{0, 1, 0},
			expected: 0.0,
			epsilon:  0.01,
		},
		{
			name:     "similar vectors",
			a:        []float32{0.9, 0.1, 0.0},
			b:        []float32{0.85, 0.15, 0.0},
			expected: 0.9978, // approximately
			epsilon:  0.01,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := cosineSimilarity(tt.a, tt.b)
			diff := result - tt.expected
			if diff < 0 {
				diff = -diff
			}
			if diff > tt.epsilon {
				t.Errorf("expected ~%.4f, got %.4f", tt.expected, result)
			}
		})
	}
}
