//go:build !windows && cgo

package selection

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/hnsw"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// KNNConfig contains configuration for the KNN selector.
type KNNConfig struct {
	// K is the number of neighbors to consider for voting (default: 5)
	K int `yaml:"k" json:"k"`

	// UseHNSW enables HNSW index for O(log n) search (default: true)
	UseHNSW bool `yaml:"use_hnsw" json:"use_hnsw"`

	// HNSWConfig contains HNSW-specific parameters
	HNSWM              int `yaml:"hnsw_m" json:"hnsw_m"`
	HNSWEfConstruction int `yaml:"hnsw_ef_construction" json:"hnsw_ef_construction"`
	HNSWEfSearch       int `yaml:"hnsw_ef_search" json:"hnsw_ef_search"`

	// WeightByDistance enables distance-weighted voting (default: true)
	// When true, closer neighbors have more influence on the vote
	WeightByDistance bool `yaml:"weight_by_distance" json:"weight_by_distance"`

	// ModelPath is the path to the pre-trained KNN model data
	ModelPath string `yaml:"model_path" json:"model_path"`

	// LabelsPath is the path to the model labels JSON file
	LabelsPath string `yaml:"labels_path" json:"labels_path"`
}

// DefaultKNNConfig returns a KNNConfig with sensible defaults.
func DefaultKNNConfig() KNNConfig {
	return KNNConfig{
		K:                  5,
		UseHNSW:            true,
		HNSWM:              16,
		HNSWEfConstruction: 200,
		HNSWEfSearch:       50,
		WeightByDistance:   true,
	}
}

// KNNSelector implements K-Nearest Neighbors model selection.
// It routes queries to the model that was most successful for similar historical queries.
type KNNSelector struct {
	mu     sync.RWMutex
	config KNNConfig

	// HNSW index for fast approximate nearest neighbor search
	index *hnsw.Index

	// modelLabels maps embedding index to the best model for that query
	modelLabels []string

	// embeddings stores the training query embeddings (for brute-force fallback)
	embeddings [][]float32

	// candidates is the list of all known model names
	candidates []string

	// nodeCount tracks the number of training samples
	nodeCount int
}

// TrainingData represents a single training sample for KNN.
type TrainingData struct {
	// Embedding is the query embedding vector
	Embedding []float32 `json:"embedding"`

	// BestModel is the model that performed best for this query
	BestModel string `json:"best_model"`

	// Performance is the quality score for the best model (optional)
	Performance float32 `json:"performance,omitempty"`

	// Query is the original query text (optional, for debugging)
	Query string `json:"query,omitempty"`
}

// KNNModelData represents the serialized KNN model.
type KNNModelData struct {
	// Version is the model format version
	Version string `json:"version"`

	// Samples contains all training samples
	Samples []TrainingData `json:"samples"`

	// Candidates is the list of all model names
	Candidates []string `json:"candidates"`

	// Config contains the training configuration
	Config KNNConfig `json:"config"`
}

// NewKNNSelector creates a new KNN selector with the given configuration.
func NewKNNSelector(config KNNConfig) *KNNSelector {
	// Apply defaults for zero values
	if config.K <= 0 {
		config.K = 5
	}
	if config.HNSWM <= 0 {
		config.HNSWM = 16
	}
	if config.HNSWEfConstruction <= 0 {
		config.HNSWEfConstruction = 200
	}
	if config.HNSWEfSearch <= 0 {
		config.HNSWEfSearch = 50
	}

	selector := &KNNSelector{
		config:      config,
		modelLabels: make([]string, 0),
		embeddings:  make([][]float32, 0),
		candidates:  make([]string, 0),
	}

	// Initialize HNSW index if enabled
	if config.UseHNSW {
		selector.index = hnsw.NewIndex(hnsw.Config{
			M:              config.HNSWM,
			EfConstruction: config.HNSWEfConstruction,
			EfSearch:       config.HNSWEfSearch,
		})
	}

	return selector
}

// LoadFromFile loads a pre-trained KNN model from a JSON file.
func (s *KNNSelector) LoadFromFile(modelPath string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	data, err := os.ReadFile(modelPath)
	if err != nil {
		return fmt.Errorf("failed to read model file: %w", err)
	}

	var modelData KNNModelData
	if err := json.Unmarshal(data, &modelData); err != nil {
		return fmt.Errorf("failed to parse model file: %w", err)
	}

	// Reset state
	s.modelLabels = make([]string, 0, len(modelData.Samples))
	s.embeddings = make([][]float32, 0, len(modelData.Samples))
	s.candidates = modelData.Candidates

	// Reinitialize HNSW index
	if s.config.UseHNSW {
		s.index = hnsw.NewIndex(hnsw.Config{
			M:              s.config.HNSWM,
			EfConstruction: s.config.HNSWEfConstruction,
			EfSearch:       s.config.HNSWEfSearch,
		})
	}

	// Add training samples
	for i, sample := range modelData.Samples {
		s.modelLabels = append(s.modelLabels, sample.BestModel)
		s.embeddings = append(s.embeddings, sample.Embedding)

		if s.index != nil {
			s.index.Add(i, sample.Embedding)
		}
	}

	s.nodeCount = len(modelData.Samples)
	logging.Infof("[KNNSelector] Loaded %d training samples with %d candidates", s.nodeCount, len(s.candidates))

	return nil
}

// AddTrainingSample adds a new training sample to the KNN model.
// This allows for online learning from feedback.
func (s *KNNSelector) AddTrainingSample(embedding []float32, bestModel string) {
	s.mu.Lock()
	defer s.mu.Unlock()

	idx := len(s.modelLabels)
	s.modelLabels = append(s.modelLabels, bestModel)
	s.embeddings = append(s.embeddings, embedding)

	if s.index != nil {
		s.index.Add(idx, embedding)
	}

	s.nodeCount++

	// Track new candidates
	found := false
	for _, c := range s.candidates {
		if c == bestModel {
			found = true
			break
		}
	}
	if !found {
		s.candidates = append(s.candidates, bestModel)
	}
}

// Select chooses the best model using K-Nearest Neighbors voting.
func (s *KNNSelector) Select(ctx context.Context, selCtx *SelectionContext) (*SelectionResult, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	// Validate input
	if len(selCtx.QueryEmbedding) == 0 {
		return nil, fmt.Errorf("KNN selector requires query embedding")
	}

	if len(selCtx.Candidates) == 0 {
		return nil, fmt.Errorf("no candidates provided")
	}

	// If no training data, fall back to first candidate
	if s.nodeCount == 0 {
		logging.Warnf("[KNNSelector] No training data, falling back to first candidate")
		return &SelectionResult{
			SelectedModel: selCtx.Candidates[0],
			Confidence:    0.5,
			Scores:        nil,
			SelectorName:  s.Name(),
			Metadata: map[string]interface{}{
				"fallback": "no_training_data",
			},
		}, nil
	}

	// Find K nearest neighbors
	var neighbors []neighborInfo
	if s.index != nil && s.index.Size() > 0 {
		neighbors = s.searchHNSW(selCtx.QueryEmbedding, s.config.K)
	} else {
		neighbors = s.searchBruteForce(selCtx.QueryEmbedding, s.config.K)
	}

	// Vote for best model
	votes := make(map[string]float32)
	for _, neighbor := range neighbors {
		model := s.modelLabels[neighbor.ID]

		// Check if model is in candidates list
		isCandidate := false
		for _, c := range selCtx.Candidates {
			if c == model {
				isCandidate = true
				break
			}
		}
		if !isCandidate {
			continue
		}

		if s.config.WeightByDistance {
			// Weight by inverse distance (higher similarity = more weight)
			weight := neighbor.Similarity
			if weight < 0.01 {
				weight = 0.01 // Prevent division issues
			}
			votes[model] += weight
		} else {
			votes[model] += 1.0
		}
	}

	// Find the model with most votes
	var bestModel string
	var maxVotes float32
	for model, vote := range votes {
		if vote > maxVotes {
			maxVotes = vote
			bestModel = model
		}
	}

	// If no valid votes, fall back to first candidate
	if bestModel == "" {
		bestModel = selCtx.Candidates[0]
		maxVotes = 0
	}

	// Normalize scores
	totalVotes := float32(0)
	for _, v := range votes {
		totalVotes += v
	}

	scores := make(map[string]float32)
	for _, c := range selCtx.Candidates {
		if totalVotes > 0 {
			scores[c] = votes[c] / totalVotes
		} else {
			scores[c] = 0
		}
	}

	// Calculate confidence
	confidence := float32(0.5)
	if totalVotes > 0 && len(neighbors) > 0 {
		// Confidence based on vote margin and neighbor similarity
		confidence = maxVotes / totalVotes
		avgSimilarity := float32(0)
		for _, n := range neighbors {
			avgSimilarity += n.Similarity
		}
		avgSimilarity /= float32(len(neighbors))
		confidence = confidence * avgSimilarity
	}

	logging.Debugf("[KNNSelector] Selected %s with confidence %.4f (votes: %v)", bestModel, confidence, votes)

	return &SelectionResult{
		SelectedModel: bestModel,
		Confidence:    confidence,
		Scores:        scores,
		SelectorName:  s.Name(),
		Metadata: map[string]interface{}{
			"k":               s.config.K,
			"neighbors_found": len(neighbors),
			"votes":           votes,
			"use_hnsw":        s.config.UseHNSW,
		},
	}, nil
}

// neighborInfo holds information about a neighbor.
type neighborInfo struct {
	ID         int
	Similarity float32
}

// searchHNSW performs approximate nearest neighbor search using HNSW.
func (s *KNNSelector) searchHNSW(queryEmbedding []float32, k int) []neighborInfo {
	results := s.index.Search(queryEmbedding, k)
	neighbors := make([]neighborInfo, 0, len(results))
	for _, r := range results {
		neighbors = append(neighbors, neighborInfo{
			ID:         r.ID,
			Similarity: r.Similarity,
		})
	}
	return neighbors
}

// searchBruteForce performs exact nearest neighbor search.
func (s *KNNSelector) searchBruteForce(queryEmbedding []float32, k int) []neighborInfo {
	type scoredNeighbor struct {
		id         int
		similarity float32
	}

	// Calculate similarity to all training samples
	scored := make([]scoredNeighbor, 0, len(s.embeddings))
	for i, emb := range s.embeddings {
		sim := cosineSimilarity(queryEmbedding, emb)
		scored = append(scored, scoredNeighbor{id: i, similarity: sim})
	}

	// Sort by similarity (descending)
	for i := 0; i < len(scored)-1; i++ {
		for j := i + 1; j < len(scored); j++ {
			if scored[j].similarity > scored[i].similarity {
				scored[i], scored[j] = scored[j], scored[i]
			}
		}
	}

	// Return top k
	if len(scored) > k {
		scored = scored[:k]
	}

	neighbors := make([]neighborInfo, 0, len(scored))
	for _, sn := range scored {
		neighbors = append(neighbors, neighborInfo{
			ID:         sn.id,
			Similarity: sn.similarity,
		})
	}
	return neighbors
}

// cosineSimilarity calculates cosine similarity between two vectors.
func cosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) {
		return 0
	}

	var dotProduct, normA, normB float32
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / (sqrt32(normA) * sqrt32(normB))
}

// sqrt32 is a simple float32 square root.
func sqrt32(x float32) float32 {
	if x <= 0 {
		return 0
	}
	// Newton's method
	z := x / 2
	for i := 0; i < 10; i++ {
		z = (z + x/z) / 2
	}
	return z
}

// Name returns the selector name.
func (s *KNNSelector) Name() string {
	return "knn"
}

// Update adds feedback as a new training sample.
func (s *KNNSelector) Update(ctx context.Context, feedback *SelectionFeedback) error {
	if len(feedback.QueryEmbedding) == 0 {
		return fmt.Errorf("feedback requires query embedding for KNN update")
	}

	// Use the winner model or the selected model based on positive rating
	bestModel := feedback.Winner
	if bestModel == "" && feedback.Rating > 0 {
		bestModel = feedback.SelectedModel
	}

	if bestModel == "" {
		return nil // No positive feedback to learn from
	}

	s.AddTrainingSample(feedback.QueryEmbedding, bestModel)
	logging.Debugf("[KNNSelector] Added training sample for model %s", bestModel)

	return nil
}

// SaveToFile saves the KNN model to a JSON file.
func (s *KNNSelector) SaveToFile(modelPath string) error {
	s.mu.RLock()
	defer s.mu.RUnlock()

	samples := make([]TrainingData, 0, len(s.modelLabels))
	for i, model := range s.modelLabels {
		samples = append(samples, TrainingData{
			Embedding: s.embeddings[i],
			BestModel: model,
		})
	}

	modelData := KNNModelData{
		Version:    "1.0",
		Samples:    samples,
		Candidates: s.candidates,
		Config:     s.config,
	}

	data, err := json.MarshalIndent(modelData, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal model data: %w", err)
	}

	if err := os.WriteFile(modelPath, data, 0644); err != nil {
		return fmt.Errorf("failed to write model file: %w", err)
	}

	logging.Infof("[KNNSelector] Saved model with %d samples to %s", len(samples), modelPath)
	return nil
}

// Size returns the number of training samples.
func (s *KNNSelector) Size() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.nodeCount
}

// Candidates returns the list of known model candidates.
func (s *KNNSelector) Candidates() []string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	result := make([]string, len(s.candidates))
	copy(result, s.candidates)
	return result
}
