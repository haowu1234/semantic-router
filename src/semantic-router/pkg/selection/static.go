package selection

import (
	"context"
)

// StaticSelector always returns the first candidate model.
// This provides backwards-compatible behavior for existing deployments.
type StaticSelector struct{}

// NewStaticSelector creates a new static selector.
func NewStaticSelector() *StaticSelector {
	return &StaticSelector{}
}

// Select returns the first candidate model (backwards compatible behavior).
func (s *StaticSelector) Select(ctx context.Context, selCtx *SelectionContext) (*SelectionResult, error) {
	if len(selCtx.Candidates) == 0 {
		return &SelectionResult{
			SelectedModel: "",
			Confidence:    0,
			Scores:        nil,
			SelectorName:  s.Name(),
		}, nil
	}

	// Always select the first candidate
	selected := selCtx.Candidates[0]

	// Build scores map (all equal weight)
	scores := make(map[string]float32)
	for _, c := range selCtx.Candidates {
		if c == selected {
			scores[c] = 1.0
		} else {
			scores[c] = 0.0
		}
	}

	return &SelectionResult{
		SelectedModel: selected,
		Confidence:    1.0,
		Scores:        scores,
		SelectorName:  s.Name(),
		Metadata: map[string]interface{}{
			"method": "first_candidate",
		},
	}, nil
}

// Name returns the selector name.
func (s *StaticSelector) Name() string {
	return "static"
}

// Update is a no-op for static selector.
func (s *StaticSelector) Update(ctx context.Context, feedback *SelectionFeedback) error {
	// Static selector doesn't learn from feedback
	return nil
}

func init() {
	// Register static selector as default
	GlobalRegistry().Register("static", NewStaticSelector())
}
