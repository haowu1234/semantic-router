package selection

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// ModelSelectionConfig represents the configuration for model selection.
type ModelSelectionConfig struct {
	// Method specifies the selection algorithm: "static", "knn", "elo", "router_dc", "automix", "hybrid"
	Method string `yaml:"method" json:"method"`

	// KNN configuration
	KNN KNNConfig `yaml:"knn" json:"knn,omitempty"`

	// Elo configuration (placeholder for future implementation)
	Elo struct {
		InitialRating float64 `yaml:"initial_rating" json:"initial_rating"`
		KFactor       float64 `yaml:"k_factor" json:"k_factor"`
	} `yaml:"elo" json:"elo,omitempty"`

	// AutoMix configuration (placeholder for future implementation)
	AutoMix struct {
		CostQualityTradeoff float64 `yaml:"cost_quality_tradeoff" json:"cost_quality_tradeoff"`
	} `yaml:"automix" json:"automix,omitempty"`

	// Hybrid configuration (placeholder for future implementation)
	Hybrid struct {
		EloWeight      float64 `yaml:"elo_weight" json:"elo_weight"`
		RouterDCWeight float64 `yaml:"router_dc_weight" json:"router_dc_weight"`
		AutoMixWeight  float64 `yaml:"automix_weight" json:"automix_weight"`
		CostWeight     float64 `yaml:"cost_weight" json:"cost_weight"`
	} `yaml:"hybrid" json:"hybrid,omitempty"`
}

// NewSelectorFromConfig creates a selector based on configuration.
func NewSelectorFromConfig(config ModelSelectionConfig) (Selector, error) {
	switch config.Method {
	case "", "static":
		logging.Infof("[Selection] Using static selector (backwards compatible)")
		return NewStaticSelector(), nil

	case "knn":
		logging.Infof("[Selection] Initializing KNN selector with K=%d", config.KNN.K)
		selector := NewKNNSelector(config.KNN)

		// Load model if path is specified
		if config.KNN.ModelPath != "" {
			if err := selector.LoadFromFile(config.KNN.ModelPath); err != nil {
				return nil, fmt.Errorf("failed to load KNN model: %w", err)
			}
		}

		return selector, nil

	case "elo":
		// Placeholder for Elo selector (from PR #1089)
		logging.Warnf("[Selection] Elo selector not yet implemented, falling back to static")
		return NewStaticSelector(), nil

	case "router_dc":
		// Placeholder for RouterDC selector (from PR #1089)
		logging.Warnf("[Selection] RouterDC selector not yet implemented, falling back to static")
		return NewStaticSelector(), nil

	case "automix":
		// Placeholder for AutoMix selector (from PR #1089)
		logging.Warnf("[Selection] AutoMix selector not yet implemented, falling back to static")
		return NewStaticSelector(), nil

	case "hybrid":
		// Placeholder for Hybrid selector (from PR #1089)
		logging.Warnf("[Selection] Hybrid selector not yet implemented, falling back to static")
		return NewStaticSelector(), nil

	default:
		return nil, fmt.Errorf("unknown selection method: %s", config.Method)
	}
}

// InitializeSelectors initializes and registers selectors based on configuration.
func InitializeSelectors(config ModelSelectionConfig) error {
	registry := GlobalRegistry()

	// Always register static selector
	registry.Register("static", NewStaticSelector())

	// Create and register the configured selector
	selector, err := NewSelectorFromConfig(config)
	if err != nil {
		return err
	}

	registry.Register(selector.Name(), selector)

	// Set default based on config
	method := config.Method
	if method == "" {
		method = "static"
	}

	if err := registry.SetDefault(method); err != nil {
		// Fall back to static if the configured method isn't available
		logging.Warnf("[Selection] Failed to set default to %s, falling back to static: %v", method, err)
		registry.SetDefault("static")
	}

	logging.Infof("[Selection] Initialized with default selector: %s", method)
	return nil
}
