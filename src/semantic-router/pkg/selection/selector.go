// Package selection provides model selection algorithms for intelligent routing.
// It enables the semantic router to choose the best LLM from multiple candidates
// based on learned preferences, query similarity, and various optimization strategies.
package selection

import (
	"context"
	"fmt"
	"sync"
)

// Selector is the interface that all model selection algorithms must implement.
type Selector interface {
	// Select chooses the best model from candidates based on the selection context.
	// Returns the selected model reference and any metadata about the selection.
	Select(ctx context.Context, selCtx *SelectionContext) (*SelectionResult, error)

	// Name returns the name of the selector for logging and metrics.
	Name() string

	// Update allows the selector to learn from feedback (e.g., user ratings, quality scores).
	// Not all selectors support updates - those that don't should return nil.
	Update(ctx context.Context, feedback *SelectionFeedback) error
}

// SelectionContext contains all information needed for model selection.
type SelectionContext struct {
	// Query is the user's input text
	Query string

	// QueryEmbedding is the embedding vector for the query (optional, computed on demand)
	QueryEmbedding []float32

	// Candidates is the list of model names to choose from
	Candidates []string

	// Metadata contains additional context (e.g., user ID, session ID, category)
	Metadata map[string]interface{}
}

// SelectionResult contains the outcome of a model selection.
type SelectionResult struct {
	// SelectedModel is the name of the chosen model
	SelectedModel string

	// Confidence is the selector's confidence in this choice (0.0-1.0)
	Confidence float32

	// Scores contains per-model scores (for debugging/explainability)
	Scores map[string]float32

	// Metadata contains additional information about the selection process
	Metadata map[string]interface{}

	// SelectorName identifies which selector made this decision
	SelectorName string
}

// SelectionFeedback contains information for updating selector state.
type SelectionFeedback struct {
	// Query is the original query text
	Query string

	// QueryEmbedding is the embedding vector (if available)
	QueryEmbedding []float32

	// SelectedModel is the model that was used
	SelectedModel string

	// Rating is the quality rating (-1.0 to 1.0, or specific scale per selector)
	Rating float32

	// Winner is the model that should have been selected (for pairwise comparison)
	Winner string

	// Loser is the model that should not have been selected
	Loser string

	// Metadata contains additional feedback context
	Metadata map[string]interface{}
}

// Registry manages available selectors and provides factory methods.
type Registry struct {
	mu        sync.RWMutex
	selectors map[string]Selector
	default_  string
}

// NewRegistry creates a new selector registry.
func NewRegistry() *Registry {
	return &Registry{
		selectors: make(map[string]Selector),
		default_:  "static",
	}
}

// Register adds a selector to the registry.
func (r *Registry) Register(name string, selector Selector) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.selectors[name] = selector
}

// Get returns a selector by name.
func (r *Registry) Get(name string) (Selector, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	s, ok := r.selectors[name]
	return s, ok
}

// GetDefault returns the default selector.
func (r *Registry) GetDefault() Selector {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if s, ok := r.selectors[r.default_]; ok {
		return s
	}
	// Fallback to static if default not found
	if s, ok := r.selectors["static"]; ok {
		return s
	}
	return nil
}

// SetDefault sets the default selector name.
func (r *Registry) SetDefault(name string) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.selectors[name]; !ok {
		return fmt.Errorf("selector %q not registered", name)
	}
	r.default_ = name
	return nil
}

// List returns all registered selector names.
func (r *Registry) List() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	names := make([]string, 0, len(r.selectors))
	for name := range r.selectors {
		names = append(names, name)
	}
	return names
}

// Global registry instance
var globalRegistry = NewRegistry()

// GlobalRegistry returns the global selector registry.
func GlobalRegistry() *Registry {
	return globalRegistry
}
