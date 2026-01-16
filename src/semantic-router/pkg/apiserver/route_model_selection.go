//go:build !windows && cgo

package apiserver

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

// ModelSelectionFeedbackRequest represents the request body for feedback submission.
type ModelSelectionFeedbackRequest struct {
	// DecisionName specifies which decision this feedback is for (required)
	DecisionName string `json:"decision_name"`

	// Query is the original query text (required if query_embedding not provided)
	Query string `json:"query,omitempty"`

	// QueryEmbedding is the pre-computed embedding (optional, saves computation)
	QueryEmbedding []float32 `json:"query_embedding,omitempty"`

	// SelectedModel is the model that was actually used (required)
	SelectedModel string `json:"selected_model"`

	// Rating is the quality score (-1.0 to 1.0)
	Rating float32 `json:"rating,omitempty"`

	// Winner is the model that performed better (for A/B comparison)
	Winner string `json:"winner,omitempty"`

	// Loser is the model that performed worse (for A/B comparison)
	Loser string `json:"loser,omitempty"`

	// RequestID links to the original request
	RequestID string `json:"request_id,omitempty"`

	// Metadata contains additional context
	Metadata map[string]interface{} `json:"metadata,omitempty"`
}

// ModelSelectionFeedbackResponse represents the response for feedback submission.
type ModelSelectionFeedbackResponse struct {
	Success       bool   `json:"success"`
	Message       string `json:"message,omitempty"`
	DecisionName  string `json:"decision_name"`
	SamplesCount  int    `json:"samples_count"`
	LearnedSample bool   `json:"learned_sample"`
	ProcessedAt   string `json:"processed_at"`
}

// ModelSelectionStatsResponse represents the response for stats endpoint.
type ModelSelectionStatsResponse struct {
	TotalDecisions       int                              `json:"total_decisions"`
	TotalSelections      int64                            `json:"total_selections"`
	TotalFeedbacks       int64                            `json:"total_feedbacks"`
	TotalSamplesLearned  int64                            `json:"total_samples_learned"`
	CacheHits            int64                            `json:"cache_hits"`
	CacheMisses          int64                            `json:"cache_misses"`
	CacheHitRate         float64                          `json:"cache_hit_rate"`
	LastSaveTime         string                           `json:"last_save_time,omitempty"`
	AutoSaveEnabled      bool                             `json:"auto_save_enabled"`
	Decisions            map[string]*DecisionStatsDetail  `json:"decisions,omitempty"`
}

// DecisionStatsDetail contains detailed stats for a single decision.
type DecisionStatsDetail struct {
	DecisionName        string   `json:"decision_name"`
	TotalSelections     int64    `json:"total_selections"`
	TotalFeedbacks      int64    `json:"total_feedbacks"`
	PositiveFeedbacks   int64    `json:"positive_feedbacks"`
	NegativeFeedbacks   int64    `json:"negative_feedbacks"`
	TotalSamplesLearned int64    `json:"total_samples_learned"`
	TrainingSamples     int      `json:"training_samples"`
	ValidCandidates     []string `json:"valid_candidates"`
	LastSelectionTime   string   `json:"last_selection_time,omitempty"`
	LastFeedbackTime    string   `json:"last_feedback_time,omitempty"`
}

// ModelSelectionSelectRequest represents a request for model selection.
type ModelSelectionSelectRequest struct {
	DecisionName string   `json:"decision_name"`
	Query        string   `json:"query"`
	Candidates   []string `json:"candidates"`
}

// ModelSelectionSelectResponse represents the response for model selection.
type ModelSelectionSelectResponse struct {
	DecisionName     string             `json:"decision_name"`
	SelectedModel    string             `json:"selected_model"`
	Confidence       float32            `json:"confidence"`
	Scores           map[string]float32 `json:"scores,omitempty"`
	SelectorName     string             `json:"selector_name"`
	ProcessingTimeMs int64              `json:"processing_time_ms"`
}

// handleModelSelectionFeedback handles POST /api/v1/model_selection/feedback
func (s *ClassificationAPIServer) handleModelSelectionFeedback(w http.ResponseWriter, r *http.Request) {
	// Get global selection service
	svc := selection.GetGlobalMultiDecisionService()
	if svc == nil {
		s.writeErrorResponse(w, http.StatusServiceUnavailable, "SERVICE_UNAVAILABLE",
			"Model selection service not initialized")
		return
	}

	// Parse request
	var req ModelSelectionFeedbackRequest
	if err := s.parseJSONRequest(r, &req); err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_REQUEST", err.Error())
		return
	}

	// Validate request
	if req.DecisionName == "" {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_REQUEST",
			"'decision_name' is required")
		return
	}

	if req.Query == "" && len(req.QueryEmbedding) == 0 {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_REQUEST",
			"Either 'query' or 'query_embedding' must be provided")
		return
	}

	if req.SelectedModel == "" && req.Winner == "" {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_REQUEST",
			"Either 'selected_model' or 'winner' must be provided")
		return
	}

	// Build feedback request
	feedbackReq := &selection.FeedbackRequest{
		DecisionName:   req.DecisionName,
		Query:          req.Query,
		QueryEmbedding: req.QueryEmbedding,
		SelectedModel:  req.SelectedModel,
		Rating:         req.Rating,
		Winner:         req.Winner,
		Loser:          req.Loser,
		RequestID:      req.RequestID,
		Metadata:       req.Metadata,
	}

	// Submit feedback
	if err := svc.SubmitFeedback(r.Context(), req.DecisionName, feedbackReq); err != nil {
		logging.Errorf("Failed to process feedback for decision '%s': %v", req.DecisionName, err)
		s.writeErrorResponse(w, http.StatusInternalServerError, "PROCESSING_FAILED", err.Error())
		return
	}

	// Determine if sample was learned
	learnedSample := req.Winner != "" || req.Rating > 0

	// Get current sample count for this decision
	decisionStats, _ := svc.GetDecisionStats(req.DecisionName)
	samplesCount := 0
	if decisionStats != nil {
		samplesCount = int(decisionStats.TotalSamplesLearned)
	}

	// Build response
	response := ModelSelectionFeedbackResponse{
		Success:       true,
		Message:       "Feedback processed successfully",
		DecisionName:  req.DecisionName,
		SamplesCount:  samplesCount,
		LearnedSample: learnedSample,
		ProcessedAt:   time.Now().UTC().Format(time.RFC3339),
	}

	s.writeJSONResponse(w, http.StatusOK, response)
}

// handleModelSelectionStats handles GET /api/v1/model_selection/stats
// Optional query parameter: decision_name (if provided, returns stats for that decision only)
func (s *ClassificationAPIServer) handleModelSelectionStats(w http.ResponseWriter, r *http.Request) {
	// Get global selection service
	svc := selection.GetGlobalMultiDecisionService()
	if svc == nil {
		s.writeErrorResponse(w, http.StatusServiceUnavailable, "SERVICE_UNAVAILABLE",
			"Model selection service not initialized")
		return
	}

	// Check if specific decision requested
	decisionName := r.URL.Query().Get("decision_name")
	if decisionName != "" {
		// Return stats for specific decision
		decisionStats, err := svc.GetDecisionStats(decisionName)
		if err != nil {
			s.writeErrorResponse(w, http.StatusNotFound, "NOT_FOUND",
				fmt.Sprintf("Decision '%s' not found", decisionName))
			return
		}

		detail := &DecisionStatsDetail{
			DecisionName:        decisionStats.DecisionName,
			TotalSelections:     decisionStats.TotalSelections,
			TotalFeedbacks:      decisionStats.TotalFeedbacks,
			PositiveFeedbacks:   decisionStats.PositiveFeedbacks,
			NegativeFeedbacks:   decisionStats.NegativeFeedbacks,
			TotalSamplesLearned: decisionStats.TotalSamplesLearned,
			TrainingSamples:     decisionStats.TrainingSamples,
			ValidCandidates:     decisionStats.ValidCandidates,
		}
		if !decisionStats.LastSelectionTime.IsZero() {
			detail.LastSelectionTime = decisionStats.LastSelectionTime.UTC().Format(time.RFC3339)
		}
		if !decisionStats.LastFeedbackTime.IsZero() {
			detail.LastFeedbackTime = decisionStats.LastFeedbackTime.UTC().Format(time.RFC3339)
		}

		s.writeJSONResponse(w, http.StatusOK, detail)
		return
	}

	// Return aggregated stats
	stats := svc.GetStats()

	// Calculate cache hit rate
	var cacheHitRate float64
	totalCacheOps := stats.CacheHits + stats.CacheMisses
	if totalCacheOps > 0 {
		cacheHitRate = float64(stats.CacheHits) / float64(totalCacheOps)
	}

	// Build response
	response := ModelSelectionStatsResponse{
		TotalDecisions:      stats.TotalDecisions,
		TotalSelections:     stats.TotalSelections,
		TotalFeedbacks:      stats.TotalFeedbacks,
		TotalSamplesLearned: stats.TotalSamplesLearned,
		CacheHits:           stats.CacheHits,
		CacheMisses:         stats.CacheMisses,
		CacheHitRate:        cacheHitRate,
		AutoSaveEnabled:     true,
		Decisions:           make(map[string]*DecisionStatsDetail),
	}

	// Format timestamps
	if !stats.LastSaveTime.IsZero() {
		response.LastSaveTime = stats.LastSaveTime.UTC().Format(time.RFC3339)
	}

	// Add decision-specific stats
	for name, dstat := range stats.DecisionStats {
		detail := &DecisionStatsDetail{
			DecisionName:        dstat.DecisionName,
			TotalSelections:     dstat.TotalSelections,
			TotalFeedbacks:      dstat.TotalFeedbacks,
			PositiveFeedbacks:   dstat.PositiveFeedbacks,
			NegativeFeedbacks:   dstat.NegativeFeedbacks,
			TotalSamplesLearned: dstat.TotalSamplesLearned,
			TrainingSamples:     dstat.TrainingSamples,
			ValidCandidates:     dstat.ValidCandidates,
		}
		if !dstat.LastSelectionTime.IsZero() {
			detail.LastSelectionTime = dstat.LastSelectionTime.UTC().Format(time.RFC3339)
		}
		if !dstat.LastFeedbackTime.IsZero() {
			detail.LastFeedbackTime = dstat.LastFeedbackTime.UTC().Format(time.RFC3339)
		}
		response.Decisions[name] = detail
	}

	s.writeJSONResponse(w, http.StatusOK, response)
}

// handleModelSelectionSelect handles POST /api/v1/model_selection/select
func (s *ClassificationAPIServer) handleModelSelectionSelect(w http.ResponseWriter, r *http.Request) {
	start := time.Now()

	// Get global selection service
	svc := selection.GetGlobalMultiDecisionService()
	if svc == nil {
		s.writeErrorResponse(w, http.StatusServiceUnavailable, "SERVICE_UNAVAILABLE",
			"Model selection service not initialized")
		return
	}

	// Parse request
	var req ModelSelectionSelectRequest
	if err := s.parseJSONRequest(r, &req); err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_REQUEST", err.Error())
		return
	}

	// Validate request
	if req.DecisionName == "" {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_REQUEST", "'decision_name' is required")
		return
	}
	if req.Query == "" {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_REQUEST", "Query is required")
		return
	}
	if len(req.Candidates) == 0 {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_REQUEST", "At least one candidate is required")
		return
	}

	// Perform selection
	result, err := svc.Select(r.Context(), req.DecisionName, req.Query, req.Candidates)
	if err != nil {
		logging.Errorf("Model selection failed for decision '%s': %v", req.DecisionName, err)
		s.writeErrorResponse(w, http.StatusInternalServerError, "SELECTION_FAILED", err.Error())
		return
	}

	// Build response
	response := ModelSelectionSelectResponse{
		DecisionName:     req.DecisionName,
		SelectedModel:    result.SelectedModel,
		Confidence:       result.Confidence,
		Scores:           result.Scores,
		SelectorName:     result.SelectorName,
		ProcessingTimeMs: time.Since(start).Milliseconds(),
	}

	s.writeJSONResponse(w, http.StatusOK, response)
}

// handleModelSelectionSave handles POST /api/v1/model_selection/save
func (s *ClassificationAPIServer) handleModelSelectionSave(w http.ResponseWriter, _ *http.Request) {
	// Get global selection service
	svc := selection.GetGlobalMultiDecisionService()
	if svc == nil {
		s.writeErrorResponse(w, http.StatusServiceUnavailable, "SERVICE_UNAVAILABLE",
			"Model selection service not initialized")
		return
	}

	// Save all models
	if err := svc.SaveAllModels(); err != nil {
		logging.Errorf("Failed to save models: %v", err)
		s.writeErrorResponse(w, http.StatusInternalServerError, "SAVE_FAILED", err.Error())
		return
	}

	// Get decision list
	decisions := svc.ListDecisions()

	// Build response
	response := map[string]interface{}{
		"success":         true,
		"message":         "All models saved successfully",
		"decisions_saved": len(decisions),
		"decision_names":  decisions,
		"saved_at":        time.Now().UTC().Format(time.RFC3339),
	}

	s.writeJSONResponse(w, http.StatusOK, response)
}

// handleModelSelectionCandidates handles GET /api/v1/model_selection/candidates
// Optional query parameter: decision_name (if provided, returns candidates for that decision)
func (s *ClassificationAPIServer) handleModelSelectionCandidates(w http.ResponseWriter, r *http.Request) {
	// Get global selection service
	svc := selection.GetGlobalMultiDecisionService()
	if svc == nil {
		s.writeErrorResponse(w, http.StatusServiceUnavailable, "SERVICE_UNAVAILABLE",
			"Model selection service not initialized")
		return
	}

	// Check if specific decision requested
	decisionName := r.URL.Query().Get("decision_name")
	if decisionName != "" {
		// Return candidates for specific decision
		selector, err := svc.GetDecisionSelector(decisionName)
		if err != nil {
			s.writeErrorResponse(w, http.StatusNotFound, "NOT_FOUND",
				fmt.Sprintf("Decision '%s' not found", decisionName))
			return
		}

		candidates := []string{}
		if knn, ok := selector.(*selection.KNNSelector); ok {
			candidates = knn.Candidates()
		}

		response := map[string]interface{}{
			"decision":   decisionName,
			"selector":   selector.Name(),
			"candidates": candidates,
			"count":      len(candidates),
		}

		s.writeJSONResponse(w, http.StatusOK, response)
		return
	}

	// Return all decisions and their candidates
	decisions := svc.ListDecisions()
	result := make(map[string]interface{})

	for _, name := range decisions {
		selector, err := svc.GetDecisionSelector(name)
		if err != nil {
			continue
		}

		candidates := []string{}
		if knn, ok := selector.(*selection.KNNSelector); ok {
			candidates = knn.Candidates()
		}

		result[name] = map[string]interface{}{
			"selector":   selector.Name(),
			"candidates": candidates,
			"count":      len(candidates),
		}
	}

	response := map[string]interface{}{
		"total_decisions": len(decisions),
		"decisions":       result,
	}

	s.writeJSONResponse(w, http.StatusOK, response)
}

// handleModelSelectionBatchFeedback handles POST /api/v1/model_selection/feedback/batch
func (s *ClassificationAPIServer) handleModelSelectionBatchFeedback(w http.ResponseWriter, r *http.Request) {
	// Get global selection service
	svc := selection.GetGlobalMultiDecisionService()
	if svc == nil {
		s.writeErrorResponse(w, http.StatusServiceUnavailable, "SERVICE_UNAVAILABLE",
			"Model selection service not initialized")
		return
	}

	// Parse request (array of feedback items)
	var feedbacks []ModelSelectionFeedbackRequest
	body, err := readRequestBody(r)
	if err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_REQUEST", err.Error())
		return
	}

	if err := json.Unmarshal(body, &feedbacks); err != nil {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_REQUEST", "Invalid JSON array: "+err.Error())
		return
	}

	if len(feedbacks) == 0 {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_REQUEST", "Empty feedback array")
		return
	}

	// Process each feedback
	successCount := 0
	failureCount := 0
	learnedCount := 0
	decisionCounts := make(map[string]int)
	var errors []string

	for i, req := range feedbacks {
		// Validate
		if req.DecisionName == "" {
			errors = append(errors, formatBatchError(i, "missing decision_name"))
			failureCount++
			continue
		}
		if req.Query == "" && len(req.QueryEmbedding) == 0 {
			errors = append(errors, formatBatchError(i, "missing query or query_embedding"))
			failureCount++
			continue
		}
		if req.SelectedModel == "" && req.Winner == "" {
			errors = append(errors, formatBatchError(i, "missing selected_model or winner"))
			failureCount++
			continue
		}

		// Build feedback
		feedbackReq := &selection.FeedbackRequest{
			DecisionName:   req.DecisionName,
			Query:          req.Query,
			QueryEmbedding: req.QueryEmbedding,
			SelectedModel:  req.SelectedModel,
			Rating:         req.Rating,
			Winner:         req.Winner,
			Loser:          req.Loser,
			RequestID:      req.RequestID,
			Metadata:       req.Metadata,
		}

		// Submit
		if err := svc.SubmitFeedback(r.Context(), req.DecisionName, feedbackReq); err != nil {
			errors = append(errors, formatBatchError(i, err.Error()))
			failureCount++
			continue
		}

		successCount++
		decisionCounts[req.DecisionName]++
		if req.Winner != "" || req.Rating > 0 {
			learnedCount++
		}
	}

	// Build response
	response := map[string]interface{}{
		"total":           len(feedbacks),
		"success_count":   successCount,
		"failure_count":   failureCount,
		"learned_count":   learnedCount,
		"decision_counts": decisionCounts,
		"processed_at":    time.Now().UTC().Format(time.RFC3339),
	}

	if len(errors) > 0 {
		response["errors"] = errors
	}

	s.writeJSONResponse(w, http.StatusOK, response)
}

// handleModelSelectionDecisions handles GET /api/v1/model_selection/decisions
func (s *ClassificationAPIServer) handleModelSelectionDecisions(w http.ResponseWriter, _ *http.Request) {
	// Get global selection service
	svc := selection.GetGlobalMultiDecisionService()
	if svc == nil {
		s.writeErrorResponse(w, http.StatusServiceUnavailable, "SERVICE_UNAVAILABLE",
			"Model selection service not initialized")
		return
	}

	// Get all decisions
	decisions := svc.ListDecisions()

	response := map[string]interface{}{
		"decisions": decisions,
		"count":     len(decisions),
	}

	s.writeJSONResponse(w, http.StatusOK, response)
}

// formatBatchError formats an error message for batch processing.
func formatBatchError(index int, message string) string {
	return "item " + strconv.Itoa(index) + ": " + message
}

// readRequestBody reads the request body into a byte slice.
func readRequestBody(r *http.Request) ([]byte, error) {
	if r.Body == nil {
		return nil, nil
	}
	defer r.Body.Close()

	// Read with size limit
	const maxBodySize = 10 * 1024 * 1024 // 10 MB
	return limitedRead(r.Body, maxBodySize)
}

// limitedRead reads up to maxBytes from reader.
func limitedRead(reader interface{ Read([]byte) (int, error) }, maxBytes int64) ([]byte, error) {
	buf := make([]byte, 0, 4096)
	for int64(len(buf)) < maxBytes {
		readSize := 4096
		if int64(len(buf)+readSize) > maxBytes {
			readSize = int(maxBytes - int64(len(buf)))
		}
		tmp := make([]byte, readSize)
		n, err := reader.Read(tmp)
		if n > 0 {
			buf = append(buf, tmp[:n]...)
		}
		if err != nil {
			if err.Error() == "EOF" {
				break
			}
			return buf, err
		}
	}
	return buf, nil
}
