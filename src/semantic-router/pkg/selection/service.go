//go:build !windows && cgo

package selection

import (
	"context"
	"fmt"
	"path/filepath"
	"sync"
	"time"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// Global multi-decision selection service instance
var (
	globalMultiDecisionService *MultiDecisionSelectionService
	globalServiceMu            sync.RWMutex
)

// MultiDecisionSelectionService manages multiple decision-specific selectors.
// Each Decision has its own isolated KNN model and training data.
type MultiDecisionSelectionService struct {
	mu sync.RWMutex

	// decisions maps decision name to its dedicated SelectionService
	decisions map[string]*DecisionSelectionService

	// baseConfig is the shared configuration template
	baseConfig ServiceConfig

	// embeddingCache is shared across all decisions to avoid redundant computation
	embeddingCache *EmbeddingCache

	// basePath is the directory for storing decision-specific models
	basePath string

	// autoSave management
	autoSaveEnabled  bool
	autoSaveInterval time.Duration
	autoSaveCancel   context.CancelFunc

	// aggregated stats
	stats MultiDecisionStats
}

// DecisionSelectionService manages model selection for a single Decision.
type DecisionSelectionService struct {
	mu sync.RWMutex

	// decisionName is the unique identifier for this decision
	decisionName string

	// selector is the underlying model selector (e.g., KNN)
	selector Selector

	// config holds the model selection configuration
	config ModelSelectionConfig

	// modelPath is the path for persisting this decision's model
	modelPath string

	// stats for this decision
	stats SelectionStats

	// candidates are the valid model refs for this decision
	validCandidates []string
}

// MultiDecisionStats tracks aggregated statistics.
type MultiDecisionStats struct {
	mu sync.Mutex

	TotalDecisions       int                      `json:"total_decisions"`
	TotalSelections      int64                    `json:"total_selections"`
	TotalFeedbacks       int64                    `json:"total_feedbacks"`
	TotalSamplesLearned  int64                    `json:"total_samples_learned"`
	LastSaveTime         time.Time                `json:"last_save_time,omitempty"`
	CacheHits            int64                    `json:"cache_hits"`
	CacheMisses          int64                    `json:"cache_misses"`
	DecisionStats        map[string]*DecisionStat `json:"decision_stats,omitempty"`
}

// DecisionStat contains stats for a single decision.
type DecisionStat struct {
	DecisionName        string    `json:"decision_name"`
	TotalSelections     int64     `json:"total_selections"`
	TotalFeedbacks      int64     `json:"total_feedbacks"`
	PositiveFeedbacks   int64     `json:"positive_feedbacks"`
	NegativeFeedbacks   int64     `json:"negative_feedbacks"`
	TotalSamplesLearned int64     `json:"total_samples_learned"`
	TrainingSamples     int       `json:"training_samples"`
	ValidCandidates     []string  `json:"valid_candidates"`
	LastSelectionTime   time.Time `json:"last_selection_time,omitempty"`
	LastFeedbackTime    time.Time `json:"last_feedback_time,omitempty"`
}

// SelectionStats tracks service statistics for a single decision.
type SelectionStats struct {
	mu sync.Mutex

	TotalSelections     int64     `json:"total_selections"`
	TotalFeedbacks      int64     `json:"total_feedbacks"`
	PositiveFeedbacks   int64     `json:"positive_feedbacks"`
	NegativeFeedbacks   int64     `json:"negative_feedbacks"`
	LastSelectionTime   time.Time `json:"last_selection_time,omitempty"`
	LastFeedbackTime    time.Time `json:"last_feedback_time,omitempty"`
	LastSaveTime        time.Time `json:"last_save_time,omitempty"`
	TotalSamplesLearned int64     `json:"total_samples_learned"`
	CacheHits           int64     `json:"cache_hits"`
	CacheMisses         int64     `json:"cache_misses"`
	AvgSelectionTimeMs  float64   `json:"avg_selection_time_ms"`
	selectionTimeSum    float64
	selectionTimeCount  int64
}

// EmbeddingCache caches query embeddings.
type EmbeddingCache struct {
	mu      sync.RWMutex
	cache   map[string]embeddingEntry
	maxSize int
	ttl     time.Duration
}

type embeddingEntry struct {
	embedding []float32
	createdAt time.Time
}

// ServiceConfig contains configuration for the selection service.
type ServiceConfig struct {
	// ModelSelectionConfig is the selector configuration
	ModelSelectionConfig ModelSelectionConfig

	// AutoSaveEnabled enables periodic model saving
	AutoSaveEnabled bool `yaml:"auto_save_enabled" json:"auto_save_enabled"`

	// AutoSaveInterval is the interval between auto-saves
	AutoSaveInterval time.Duration `yaml:"auto_save_interval" json:"auto_save_interval"`

	// EmbeddingCacheSize is the max number of cached embeddings
	EmbeddingCacheSize int `yaml:"embedding_cache_size" json:"embedding_cache_size"`

	// EmbeddingCacheTTL is the TTL for cached embeddings
	EmbeddingCacheTTL time.Duration `yaml:"embedding_cache_ttl" json:"embedding_cache_ttl"`

	// BasePath is the directory for storing decision-specific models
	BasePath string `yaml:"base_path" json:"base_path"`
}

// DefaultServiceConfig returns default service configuration.
func DefaultServiceConfig() ServiceConfig {
	return ServiceConfig{
		AutoSaveEnabled:    true,
		AutoSaveInterval:   5 * time.Minute,
		EmbeddingCacheSize: 10000,
		EmbeddingCacheTTL:  1 * time.Hour,
		BasePath:           "config/models",
	}
}

// NewMultiDecisionSelectionService creates a new multi-decision selection service.
func NewMultiDecisionSelectionService(config ServiceConfig) (*MultiDecisionSelectionService, error) {
	svc := &MultiDecisionSelectionService{
		decisions:        make(map[string]*DecisionSelectionService),
		baseConfig:       config,
		basePath:         config.BasePath,
		autoSaveEnabled:  config.AutoSaveEnabled,
		autoSaveInterval: config.AutoSaveInterval,
		stats: MultiDecisionStats{
			DecisionStats: make(map[string]*DecisionStat),
		},
	}

	// Initialize shared embedding cache
	if config.EmbeddingCacheSize > 0 {
		svc.embeddingCache = NewEmbeddingCache(config.EmbeddingCacheSize, config.EmbeddingCacheTTL)
	}

	// Start auto-save if enabled
	if svc.autoSaveEnabled && svc.basePath != "" {
		svc.startAutoSave()
	}

	return svc, nil
}

// GetOrCreateDecisionService gets or creates a SelectionService for a specific decision.
func (svc *MultiDecisionSelectionService) GetOrCreateDecisionService(
	decisionName string,
	validCandidates []string,
) (*DecisionSelectionService, error) {
	svc.mu.Lock()
	defer svc.mu.Unlock()

	// Return existing service
	if ds, ok := svc.decisions[decisionName]; ok {
		// Update candidates if changed
		ds.updateCandidates(validCandidates)
		return ds, nil
	}

	// Create new decision service
	ds, err := svc.createDecisionService(decisionName, validCandidates)
	if err != nil {
		return nil, err
	}

	svc.decisions[decisionName] = ds
	svc.stats.TotalDecisions = len(svc.decisions)
	svc.stats.DecisionStats[decisionName] = &DecisionStat{
		DecisionName:    decisionName,
		ValidCandidates: validCandidates,
	}

	logging.Infof("[MultiDecisionService] Created selection service for decision '%s' with %d candidates",
		decisionName, len(validCandidates))

	return ds, nil
}

// createDecisionService creates a new DecisionSelectionService.
func (svc *MultiDecisionSelectionService) createDecisionService(
	decisionName string,
	validCandidates []string,
) (*DecisionSelectionService, error) {
	// Create selector from base config
	selector, err := NewSelectorFromConfig(svc.baseConfig.ModelSelectionConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create selector for decision %s: %w", decisionName, err)
	}

	// Determine model path for this decision
	modelPath := ""
	if svc.basePath != "" {
		// Sanitize decision name for file path
		safeName := sanitizeFileName(decisionName)
		modelPath = filepath.Join(svc.basePath, safeName+"_knn_model.json")
	}

	ds := &DecisionSelectionService{
		decisionName:    decisionName,
		selector:        selector,
		config:          svc.baseConfig.ModelSelectionConfig,
		modelPath:       modelPath,
		validCandidates: validCandidates,
	}

	// Try to load existing model if path is configured
	if modelPath != "" {
		if knn, ok := selector.(*KNNSelector); ok {
			if err := knn.LoadFromFile(modelPath); err != nil {
				logging.Debugf("[DecisionService] No existing model for decision '%s': %v", decisionName, err)
			} else {
				logging.Infof("[DecisionService] Loaded existing model for decision '%s' from %s",
					decisionName, modelPath)
			}
		}
	}

	return ds, nil
}

// updateCandidates updates the valid candidates for a decision.
func (ds *DecisionSelectionService) updateCandidates(candidates []string) {
	ds.mu.Lock()
	defer ds.mu.Unlock()
	ds.validCandidates = candidates
}

// Select performs model selection for a specific decision.
func (svc *MultiDecisionSelectionService) Select(
	ctx context.Context,
	decisionName string,
	query string,
	candidates []string,
) (*SelectionResult, error) {
	// Get or create decision service
	ds, err := svc.GetOrCreateDecisionService(decisionName, candidates)
	if err != nil {
		return nil, err
	}

	// Get or compute embedding using shared cache
	embedding, cached, err := svc.getOrComputeEmbedding(query)
	if err != nil {
		return nil, fmt.Errorf("failed to get embedding: %w", err)
	}

	// Update cache stats
	svc.stats.mu.Lock()
	if cached {
		svc.stats.CacheHits++
	} else {
		svc.stats.CacheMisses++
	}
	svc.stats.mu.Unlock()

	// Perform selection
	result, err := ds.selectWithEmbedding(ctx, query, embedding, candidates)
	if err != nil {
		return nil, err
	}

	// Update aggregated stats
	svc.stats.mu.Lock()
	svc.stats.TotalSelections++
	if stat, ok := svc.stats.DecisionStats[decisionName]; ok {
		stat.TotalSelections++
		stat.LastSelectionTime = time.Now()
	}
	svc.stats.mu.Unlock()

	return result, nil
}

// selectWithEmbedding performs selection with a pre-computed embedding.
func (ds *DecisionSelectionService) selectWithEmbedding(
	ctx context.Context,
	query string,
	embedding []float32,
	candidates []string,
) (*SelectionResult, error) {
	start := time.Now()

	ds.mu.RLock()
	selector := ds.selector
	ds.mu.RUnlock()

	if selector == nil {
		return nil, fmt.Errorf("selector not initialized for decision %s", ds.decisionName)
	}

	// Filter candidates to only those valid for this decision
	validCandidates := ds.filterValidCandidates(candidates)
	if len(validCandidates) == 0 {
		validCandidates = candidates // Fallback to all if none valid
	}

	// Build selection context
	selCtx := &SelectionContext{
		Query:          query,
		QueryEmbedding: embedding,
		Candidates:     validCandidates,
	}

	// Perform selection
	result, err := selector.Select(ctx, selCtx)
	if err != nil {
		return nil, err
	}

	// Add decision name to metadata
	if result.Metadata == nil {
		result.Metadata = make(map[string]interface{})
	}
	result.Metadata["decision"] = ds.decisionName

	// Update decision stats
	elapsed := time.Since(start)
	ds.stats.mu.Lock()
	ds.stats.TotalSelections++
	ds.stats.LastSelectionTime = time.Now()
	ds.stats.selectionTimeSum += float64(elapsed.Milliseconds())
	ds.stats.selectionTimeCount++
	ds.stats.AvgSelectionTimeMs = ds.stats.selectionTimeSum / float64(ds.stats.selectionTimeCount)
	ds.stats.mu.Unlock()

	return result, nil
}

// filterValidCandidates returns only candidates that are valid for this decision.
func (ds *DecisionSelectionService) filterValidCandidates(candidates []string) []string {
	ds.mu.RLock()
	defer ds.mu.RUnlock()

	if len(ds.validCandidates) == 0 {
		return candidates
	}

	validSet := make(map[string]bool)
	for _, c := range ds.validCandidates {
		validSet[c] = true
	}

	result := make([]string, 0, len(candidates))
	for _, c := range candidates {
		if validSet[c] {
			result = append(result, c)
		}
	}
	return result
}

// SubmitFeedback processes feedback for a specific decision.
func (svc *MultiDecisionSelectionService) SubmitFeedback(
	ctx context.Context,
	decisionName string,
	feedback *FeedbackRequest,
) error {
	// Get decision service (create if needed with empty candidates)
	ds, err := svc.GetOrCreateDecisionService(decisionName, nil)
	if err != nil {
		return err
	}

	// Get or compute embedding if not provided
	var embedding []float32
	if len(feedback.QueryEmbedding) > 0 {
		embedding = feedback.QueryEmbedding
	} else if feedback.Query != "" {
		embedding, _, err = svc.getOrComputeEmbedding(feedback.Query)
		if err != nil {
			return fmt.Errorf("failed to compute embedding: %w", err)
		}
	} else {
		return fmt.Errorf("feedback requires either query text or query_embedding")
	}

	// Submit to decision service
	if err := ds.submitFeedback(ctx, embedding, feedback); err != nil {
		return err
	}

	// Update aggregated stats
	svc.stats.mu.Lock()
	svc.stats.TotalFeedbacks++
	if feedback.Rating > 0 || feedback.Winner != "" {
		svc.stats.TotalSamplesLearned++
	}
	if stat, ok := svc.stats.DecisionStats[decisionName]; ok {
		stat.TotalFeedbacks++
		stat.LastFeedbackTime = time.Now()
		if feedback.Rating > 0 || feedback.Winner != "" {
			stat.PositiveFeedbacks++
			stat.TotalSamplesLearned++
		} else if feedback.Rating < 0 {
			stat.NegativeFeedbacks++
		}
	}
	svc.stats.mu.Unlock()

	return nil
}

// submitFeedback processes feedback for this decision.
func (ds *DecisionSelectionService) submitFeedback(
	ctx context.Context,
	embedding []float32,
	feedback *FeedbackRequest,
) error {
	ds.mu.RLock()
	selector := ds.selector
	ds.mu.RUnlock()

	if selector == nil {
		return fmt.Errorf("selector not initialized for decision %s", ds.decisionName)
	}

	// Build selection feedback
	selFeedback := &SelectionFeedback{
		Query:          feedback.Query,
		QueryEmbedding: embedding,
		SelectedModel:  feedback.SelectedModel,
		Rating:         feedback.Rating,
		Winner:         feedback.Winner,
		Loser:          feedback.Loser,
		Metadata:       feedback.Metadata,
	}

	// Submit to selector
	if err := selector.Update(ctx, selFeedback); err != nil {
		return fmt.Errorf("selector update failed: %w", err)
	}

	// Update stats
	ds.stats.mu.Lock()
	ds.stats.TotalFeedbacks++
	ds.stats.LastFeedbackTime = time.Now()
	if feedback.Rating > 0 || feedback.Winner != "" {
		ds.stats.PositiveFeedbacks++
		ds.stats.TotalSamplesLearned++
	} else if feedback.Rating < 0 {
		ds.stats.NegativeFeedbacks++
	}
	ds.stats.mu.Unlock()

	logging.Debugf("[DecisionService:%s] Processed feedback for model %s (rating: %.2f, winner: %s)",
		ds.decisionName, feedback.SelectedModel, feedback.Rating, feedback.Winner)

	return nil
}

// getOrComputeEmbedding retrieves embedding from shared cache or computes it.
func (svc *MultiDecisionSelectionService) getOrComputeEmbedding(query string) ([]float32, bool, error) {
	// Check cache first
	if svc.embeddingCache != nil {
		if emb, ok := svc.embeddingCache.Get(query); ok {
			return emb, true, nil
		}
	}

	// Compute embedding
	embOutput, err := candle_binding.GetEmbeddingWithModelType(query, "qwen3", 768)
	if err != nil {
		return nil, false, err
	}

	embedding := embOutput.Embedding

	// Cache the result
	if svc.embeddingCache != nil {
		svc.embeddingCache.Set(query, embedding)
	}

	return embedding, false, nil
}

// SaveAllModels persists all decision models to disk.
func (svc *MultiDecisionSelectionService) SaveAllModels() error {
	svc.mu.RLock()
	defer svc.mu.RUnlock()

	var lastErr error
	savedCount := 0

	for name, ds := range svc.decisions {
		if err := ds.saveModel(); err != nil {
			logging.Warnf("[MultiDecisionService] Failed to save model for decision '%s': %v", name, err)
			lastErr = err
		} else {
			savedCount++
		}
	}

	svc.stats.mu.Lock()
	svc.stats.LastSaveTime = time.Now()
	svc.stats.mu.Unlock()

	logging.Infof("[MultiDecisionService] Saved %d/%d decision models", savedCount, len(svc.decisions))

	return lastErr
}

// saveModel persists this decision's model to disk.
func (ds *DecisionSelectionService) saveModel() error {
	ds.mu.RLock()
	defer ds.mu.RUnlock()

	if ds.modelPath == "" {
		return nil // No path configured
	}

	knn, ok := ds.selector.(*KNNSelector)
	if !ok {
		return nil // Selector doesn't support persistence
	}

	if err := knn.SaveToFile(ds.modelPath); err != nil {
		return err
	}

	ds.stats.mu.Lock()
	ds.stats.LastSaveTime = time.Now()
	ds.stats.mu.Unlock()

	return nil
}

// startAutoSave starts the periodic auto-save goroutine.
func (svc *MultiDecisionSelectionService) startAutoSave() {
	if svc.autoSaveInterval <= 0 {
		svc.autoSaveInterval = 5 * time.Minute
	}

	ctx, cancel := context.WithCancel(context.Background())
	svc.autoSaveCancel = cancel

	go func() {
		ticker := time.NewTicker(svc.autoSaveInterval)
		defer ticker.Stop()

		for {
			select {
			case <-ctx.Done():
				// Final save on shutdown
				if err := svc.SaveAllModels(); err != nil {
					logging.Warnf("[MultiDecisionService] Final save failed: %v", err)
				}
				return
			case <-ticker.C:
				if err := svc.SaveAllModels(); err != nil {
					logging.Warnf("[MultiDecisionService] Auto-save failed: %v", err)
				}
			}
		}
	}()

	logging.Infof("[MultiDecisionService] Auto-save enabled with interval %v", svc.autoSaveInterval)
}

// Stop gracefully stops the selection service.
func (svc *MultiDecisionSelectionService) Stop() error {
	if svc.autoSaveCancel != nil {
		svc.autoSaveCancel()
	}
	return nil
}

// GetStats returns aggregated statistics.
func (svc *MultiDecisionSelectionService) GetStats() MultiDecisionStats {
	svc.mu.RLock()
	defer svc.mu.RUnlock()

	svc.stats.mu.Lock()
	defer svc.stats.mu.Unlock()

	// Update decision-specific stats
	for name, ds := range svc.decisions {
		ds.stats.mu.Lock()
		if stat, ok := svc.stats.DecisionStats[name]; ok {
			if knn, ok := ds.selector.(*KNNSelector); ok {
				stat.TrainingSamples = knn.Size()
			}
		}
		ds.stats.mu.Unlock()
	}

	// Return a copy
	statsCopy := MultiDecisionStats{
		TotalDecisions:      svc.stats.TotalDecisions,
		TotalSelections:     svc.stats.TotalSelections,
		TotalFeedbacks:      svc.stats.TotalFeedbacks,
		TotalSamplesLearned: svc.stats.TotalSamplesLearned,
		LastSaveTime:        svc.stats.LastSaveTime,
		CacheHits:           svc.stats.CacheHits,
		CacheMisses:         svc.stats.CacheMisses,
		DecisionStats:       make(map[string]*DecisionStat),
	}

	for k, v := range svc.stats.DecisionStats {
		statsCopy.DecisionStats[k] = &DecisionStat{
			DecisionName:        v.DecisionName,
			TotalSelections:     v.TotalSelections,
			TotalFeedbacks:      v.TotalFeedbacks,
			PositiveFeedbacks:   v.PositiveFeedbacks,
			NegativeFeedbacks:   v.NegativeFeedbacks,
			TotalSamplesLearned: v.TotalSamplesLearned,
			TrainingSamples:     v.TrainingSamples,
			ValidCandidates:     v.ValidCandidates,
			LastSelectionTime:   v.LastSelectionTime,
			LastFeedbackTime:    v.LastFeedbackTime,
		}
	}

	return statsCopy
}

// GetDecisionStats returns statistics for a specific decision.
func (svc *MultiDecisionSelectionService) GetDecisionStats(decisionName string) (*DecisionStat, error) {
	svc.mu.RLock()
	ds, ok := svc.decisions[decisionName]
	svc.mu.RUnlock()

	if !ok {
		return nil, fmt.Errorf("decision '%s' not found", decisionName)
	}

	ds.stats.mu.Lock()
	defer ds.stats.mu.Unlock()

	trainingSamples := 0
	if knn, ok := ds.selector.(*KNNSelector); ok {
		trainingSamples = knn.Size()
	}

	return &DecisionStat{
		DecisionName:        decisionName,
		TotalSelections:     ds.stats.TotalSelections,
		TotalFeedbacks:      ds.stats.TotalFeedbacks,
		PositiveFeedbacks:   ds.stats.PositiveFeedbacks,
		NegativeFeedbacks:   ds.stats.NegativeFeedbacks,
		TotalSamplesLearned: ds.stats.TotalSamplesLearned,
		TrainingSamples:     trainingSamples,
		ValidCandidates:     ds.validCandidates,
		LastSelectionTime:   ds.stats.LastSelectionTime,
		LastFeedbackTime:    ds.stats.LastFeedbackTime,
	}, nil
}

// ListDecisions returns all decision names.
func (svc *MultiDecisionSelectionService) ListDecisions() []string {
	svc.mu.RLock()
	defer svc.mu.RUnlock()

	decisions := make([]string, 0, len(svc.decisions))
	for name := range svc.decisions {
		decisions = append(decisions, name)
	}
	return decisions
}

// GetDecisionSelector returns the selector for a specific decision.
func (svc *MultiDecisionSelectionService) GetDecisionSelector(decisionName string) (Selector, error) {
	svc.mu.RLock()
	defer svc.mu.RUnlock()

	ds, ok := svc.decisions[decisionName]
	if !ok {
		return nil, fmt.Errorf("decision '%s' not found", decisionName)
	}

	return ds.selector, nil
}

// FeedbackRequest represents an API request for feedback submission.
type FeedbackRequest struct {
	// DecisionName specifies which decision this feedback is for (required)
	DecisionName string `json:"decision_name"`

	// Query is the original query text (used to compute embedding if not provided)
	Query string `json:"query,omitempty"`

	// QueryEmbedding is the pre-computed embedding (optional, saves computation)
	QueryEmbedding []float32 `json:"query_embedding,omitempty"`

	// SelectedModel is the model that was actually used
	SelectedModel string `json:"selected_model"`

	// Rating is the quality score (-1.0 to 1.0)
	// Positive: model performed well
	// Negative: model performed poorly
	// Zero: neutral feedback
	Rating float32 `json:"rating,omitempty"`

	// Winner is the model that performed better (for A/B comparison)
	Winner string `json:"winner,omitempty"`

	// Loser is the model that performed worse (for A/B comparison)
	Loser string `json:"loser,omitempty"`

	// RequestID links to the original request for audit trail
	RequestID string `json:"request_id,omitempty"`

	// Metadata contains additional context
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// FeedbackResponse represents the API response for feedback submission.
type FeedbackResponse struct {
	Success       bool   `json:"success"`
	Message       string `json:"message,omitempty"`
	DecisionName  string `json:"decision_name"`
	SamplesCount  int    `json:"samples_count"`
	LearnedSample bool   `json:"learned_sample"`
}

// NewEmbeddingCache creates a new embedding cache.
func NewEmbeddingCache(maxSize int, ttl time.Duration) *EmbeddingCache {
	if maxSize <= 0 {
		maxSize = 10000
	}
	if ttl <= 0 {
		ttl = 1 * time.Hour
	}
	return &EmbeddingCache{
		cache:   make(map[string]embeddingEntry),
		maxSize: maxSize,
		ttl:     ttl,
	}
}

// Get retrieves an embedding from cache.
func (c *EmbeddingCache) Get(key string) ([]float32, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	entry, ok := c.cache[key]
	if !ok {
		return nil, false
	}

	// Check TTL
	if time.Since(entry.createdAt) > c.ttl {
		return nil, false
	}

	return entry.embedding, true
}

// Set stores an embedding in cache.
func (c *EmbeddingCache) Set(key string, embedding []float32) {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Evict oldest entries if at capacity
	if len(c.cache) >= c.maxSize {
		c.evictOldest()
	}

	c.cache[key] = embeddingEntry{
		embedding: embedding,
		createdAt: time.Now(),
	}
}

// evictOldest removes the oldest 10% of entries.
func (c *EmbeddingCache) evictOldest() {
	toEvict := c.maxSize / 10
	if toEvict < 1 {
		toEvict = 1
	}

	// Simple eviction: remove entries with oldest timestamps
	type entry struct {
		key       string
		createdAt time.Time
	}
	entries := make([]entry, 0, len(c.cache))
	for k, v := range c.cache {
		entries = append(entries, entry{k, v.createdAt})
	}

	// Sort by creation time (simple bubble sort for small eviction count)
	for i := 0; i < toEvict && i < len(entries)-1; i++ {
		for j := i + 1; j < len(entries); j++ {
			if entries[j].createdAt.Before(entries[i].createdAt) {
				entries[i], entries[j] = entries[j], entries[i]
			}
		}
		delete(c.cache, entries[i].key)
	}
}

// Size returns the current cache size.
func (c *EmbeddingCache) Size() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return len(c.cache)
}

// SetGlobalMultiDecisionService sets the global multi-decision selection service.
func SetGlobalMultiDecisionService(svc *MultiDecisionSelectionService) {
	globalServiceMu.Lock()
	defer globalServiceMu.Unlock()
	globalMultiDecisionService = svc
}

// GetGlobalMultiDecisionService returns the global multi-decision selection service.
func GetGlobalMultiDecisionService() *MultiDecisionSelectionService {
	globalServiceMu.RLock()
	defer globalServiceMu.RUnlock()
	return globalMultiDecisionService
}

// sanitizeFileName removes/replaces characters not safe for file names.
func sanitizeFileName(name string) string {
	result := make([]byte, 0, len(name))
	for i := 0; i < len(name); i++ {
		c := name[i]
		if (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '-' {
			result = append(result, c)
		} else if c == ' ' || c == '/' || c == '\\' || c == ':' {
			result = append(result, '_')
		}
	}
	if len(result) == 0 {
		return "default"
	}
	return string(result)
}

// ============================================================================
// Backward Compatibility: SelectionService wraps MultiDecisionSelectionService
// ============================================================================

// SelectionService is provided for backward compatibility.
// It wraps MultiDecisionSelectionService using a default decision.
type SelectionService struct {
	multi           *MultiDecisionSelectionService
	defaultDecision string
}

// NewSelectionService creates a SelectionService for backward compatibility.
func NewSelectionService(config ServiceConfig) (*SelectionService, error) {
	multi, err := NewMultiDecisionSelectionService(config)
	if err != nil {
		return nil, err
	}

	return &SelectionService{
		multi:           multi,
		defaultDecision: "default",
	}, nil
}

// Select performs model selection using the default decision.
func (svc *SelectionService) Select(ctx context.Context, query string, candidates []string) (*SelectionResult, error) {
	return svc.multi.Select(ctx, svc.defaultDecision, query, candidates)
}

// SubmitFeedback processes feedback using the default decision.
func (svc *SelectionService) SubmitFeedback(ctx context.Context, feedback *FeedbackRequest) error {
	if feedback.DecisionName == "" {
		feedback.DecisionName = svc.defaultDecision
	}
	return svc.multi.SubmitFeedback(ctx, feedback.DecisionName, feedback)
}

// SaveModel saves all models.
func (svc *SelectionService) SaveModel() error {
	return svc.multi.SaveAllModels()
}

// Stop stops the service.
func (svc *SelectionService) Stop() error {
	return svc.multi.Stop()
}

// GetStats returns aggregated stats.
func (svc *SelectionService) GetStats() SelectionStats {
	multiStats := svc.multi.GetStats()

	return SelectionStats{
		TotalSelections:     multiStats.TotalSelections,
		TotalFeedbacks:      multiStats.TotalFeedbacks,
		TotalSamplesLearned: multiStats.TotalSamplesLearned,
		CacheHits:           multiStats.CacheHits,
		CacheMisses:         multiStats.CacheMisses,
		LastSaveTime:        multiStats.LastSaveTime,
	}
}

// GetSelector returns the default decision's selector.
func (svc *SelectionService) GetSelector() Selector {
	selector, _ := svc.multi.GetDecisionSelector(svc.defaultDecision)
	return selector
}

// GetMultiDecisionService returns the underlying multi-decision service.
func (svc *SelectionService) GetMultiDecisionService() *MultiDecisionSelectionService {
	return svc.multi
}

// SetGlobalSelectionService sets the global selection service (backward compatibility).
func SetGlobalSelectionService(svc *SelectionService) {
	if svc != nil && svc.multi != nil {
		SetGlobalMultiDecisionService(svc.multi)
	}
}

// GetGlobalSelectionService returns the global selection service (backward compatibility).
func GetGlobalSelectionService() *SelectionService {
	multi := GetGlobalMultiDecisionService()
	if multi == nil {
		return nil
	}
	return &SelectionService{
		multi:           multi,
		defaultDecision: "default",
	}
}
