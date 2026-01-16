//go:build windows || !cgo

package selection

import (
	"context"
	"fmt"
)

// KNNConfig contains configuration for the KNN selector (stub for non-cgo builds).
type KNNConfig struct {
	K                  int    `yaml:"k" json:"k"`
	UseHNSW            bool   `yaml:"use_hnsw" json:"use_hnsw"`
	HNSWM              int    `yaml:"hnsw_m" json:"hnsw_m"`
	HNSWEfConstruction int    `yaml:"hnsw_ef_construction" json:"hnsw_ef_construction"`
	HNSWEfSearch       int    `yaml:"hnsw_ef_search" json:"hnsw_ef_search"`
	WeightByDistance   bool   `yaml:"weight_by_distance" json:"weight_by_distance"`
	ModelPath          string `yaml:"model_path" json:"model_path"`
	LabelsPath         string `yaml:"labels_path" json:"labels_path"`
}

// DefaultKNNConfig returns a KNNConfig with sensible defaults (stub).
func DefaultKNNConfig() KNNConfig {
	return KNNConfig{
		K:                  5,
		UseHNSW:            false,
		HNSWM:              16,
		HNSWEfConstruction: 200,
		HNSWEfSearch:       50,
		WeightByDistance:   true,
	}
}

// KNNSelector stub for non-cgo builds - falls back to static selection.
type KNNSelector struct {
	config KNNConfig
}

// NewKNNSelector creates a stub KNN selector that falls back to static selection.
func NewKNNSelector(config KNNConfig) *KNNSelector {
	return &KNNSelector{config: config}
}

// LoadFromFile is a no-op stub.
func (s *KNNSelector) LoadFromFile(modelPath string) error {
	return fmt.Errorf("KNN selector requires cgo build")
}

// AddTrainingSample is a no-op stub.
func (s *KNNSelector) AddTrainingSample(embedding []float32, bestModel string) {
	// No-op in stub
}

// Select falls back to first candidate in stub mode.
func (s *KNNSelector) Select(ctx context.Context, selCtx *SelectionContext) (*SelectionResult, error) {
	if len(selCtx.Candidates) == 0 {
		return nil, fmt.Errorf("no candidates provided")
	}
	return &SelectionResult{
		SelectedModel: selCtx.Candidates[0],
		Confidence:    0.5,
		SelectorName:  s.Name(),
		Metadata: map[string]interface{}{
			"stub": true,
		},
	}, nil
}

// Name returns the selector name.
func (s *KNNSelector) Name() string {
	return "knn"
}

// Update is a no-op stub.
func (s *KNNSelector) Update(ctx context.Context, feedback *SelectionFeedback) error {
	return nil
}

// SaveToFile is a no-op stub.
func (s *KNNSelector) SaveToFile(modelPath string) error {
	return fmt.Errorf("KNN selector requires cgo build")
}

// Size returns 0 in stub mode.
func (s *KNNSelector) Size() int {
	return 0
}

// Candidates returns empty slice in stub mode.
func (s *KNNSelector) Candidates() []string {
	return nil
}
