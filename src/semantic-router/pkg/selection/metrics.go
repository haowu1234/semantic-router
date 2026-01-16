package selection

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	// selectionsTotal tracks total model selections by selector and model
	selectionsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: "semantic_router",
			Subsystem: "selection",
			Name:      "total",
			Help:      "Total number of model selections by selector and selected model",
		},
		[]string{"selector", "model"},
	)

	// selectionLatency tracks selection operation latency
	selectionLatency = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: "semantic_router",
			Subsystem: "selection",
			Name:      "latency_seconds",
			Help:      "Latency of model selection operations",
			Buckets:   []float64{0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1},
		},
		[]string{"selector"},
	)

	// selectionConfidence tracks selection confidence scores
	selectionConfidence = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Namespace: "semantic_router",
			Subsystem: "selection",
			Name:      "confidence",
			Help:      "Confidence scores for model selections",
			Buckets:   []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0},
		},
		[]string{"selector"},
	)

	// knnTrainingSamples tracks the number of KNN training samples
	knnTrainingSamples = promauto.NewGauge(
		prometheus.GaugeOpts{
			Namespace: "semantic_router",
			Subsystem: "selection",
			Name:      "knn_training_samples",
			Help:      "Number of training samples in the KNN selector",
		},
	)

	// feedbackTotal tracks feedback events by type
	feedbackTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: "semantic_router",
			Subsystem: "selection",
			Name:      "feedback_total",
			Help:      "Total number of feedback events by selector and type",
		},
		[]string{"selector", "type"},
	)
)

// RecordSelection records metrics for a model selection.
func RecordSelection(selectorName, modelName string, confidence float32, latencySeconds float64) {
	selectionsTotal.WithLabelValues(selectorName, modelName).Inc()
	selectionLatency.WithLabelValues(selectorName).Observe(latencySeconds)
	selectionConfidence.WithLabelValues(selectorName).Observe(float64(confidence))
}

// RecordKNNTrainingSamples updates the KNN training samples gauge.
func RecordKNNTrainingSamples(count int) {
	knnTrainingSamples.Set(float64(count))
}

// RecordFeedback records a feedback event.
func RecordFeedback(selectorName, feedbackType string) {
	feedbackTotal.WithLabelValues(selectorName, feedbackType).Inc()
}
