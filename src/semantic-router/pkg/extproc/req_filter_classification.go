package extproc

import (
	"context"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/promptcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

// performDecisionEvaluation performs decision evaluation using DecisionEngine
// Returns (decisionName, confidence, reasoningDecision, selectedModel)
// This is the new approach that uses Decision-based routing with AND/OR rule combinations
// Decision evaluation is ALWAYS performed when decisions are configured (for plugin features like
// hallucination detection), but model selection only happens for auto models.
func (r *OpenAIRouter) performDecisionEvaluation(originalModel string, userContent string, nonUserMessages []string, ctx *RequestContext) (string, float64, entropy.ReasoningDecision, string, error) {
	if len(nonUserMessages) == 0 && userContent == "" {
		return "", 0.0, entropy.ReasoningDecision{}, "", nil
	}

	if len(r.Config.Decisions) == 0 {
		if r.Config.IsAutoModelName(originalModel) {
			logging.Warnf("No decisions configured, using default model")
			return "", 0.0, entropy.ReasoningDecision{}, r.Config.DefaultModel, nil
		}
		return "", 0.0, entropy.ReasoningDecision{}, "", nil
	}

	evaluationText, allMessagesText, ok := prepareDecisionEvaluationTexts(userContent, nonUserMessages)
	if !ok {
		return "", 0.0, entropy.ReasoningDecision{}, "", nil
	}

	signals, authzErr := r.evaluateSignalsForDecision(evaluationText, allMessagesText, nonUserMessages, ctx)
	if authzErr != nil {
		return "", 0, entropy.ReasoningDecision{}, "", authzErr
	}

	r.processUserFeedbackForElo(signals.MatchedUserFeedbackRules, originalModel, ctx)

	result, err := r.evaluateDecisionMatch(signals, ctx)
	if err != nil {
		if r.Config.IsAutoModelName(originalModel) {
			return "", 0.0, entropy.ReasoningDecision{}, r.Config.DefaultModel, nil
		}
		return "", 0.0, entropy.ReasoningDecision{}, "", nil
	}

	if result == nil || result.Decision == nil {
		if r.Config.IsAutoModelName(originalModel) {
			return "", 0.0, entropy.ReasoningDecision{}, r.Config.DefaultModel, nil
		}
		return "", 0.0, entropy.ReasoningDecision{}, "", nil
	}

	decisionName, evaluationConfidence, categoryName := r.applyMatchedDecisionContext(result, ctx)

	if !r.Config.IsAutoModelName(originalModel) {
		logging.Infof("Model %s explicitly specified, keeping original model (decision %s plugins will be applied)",
			originalModel, decisionName)
		return decisionName, evaluationConfidence, entropy.ReasoningDecision{}, "", nil
	}

	selectedModel := r.selectDecisionModel(result, decisionName, userContent, nonUserMessages, categoryName, ctx)
	return decisionName, evaluationConfidence, entropy.ReasoningDecision{}, selectedModel, nil
}

// getSelectionMethod determines which selection algorithm to use.
// Per-decision algorithm is the primary configuration (aligned with looper pattern).
// Defaults to static selection if no algorithm is specified.
func (r *OpenAIRouter) getSelectionMethod(algorithm *config.AlgorithmConfig) selection.SelectionMethod {
	// Check per-decision algorithm (aligned with looper pattern)
	if algorithm != nil && algorithm.Type != "" {
		switch algorithm.Type {
		case "elo":
			return selection.MethodElo
		case "router_dc":
			return selection.MethodRouterDC
		case "automix":
			return selection.MethodAutoMix
		case "hybrid":
			return selection.MethodHybrid
		case "rl_driven":
			return selection.MethodRLDriven
		case "gmtrouter":
			return selection.MethodGMTRouter
		case "latency_aware":
			return selection.MethodLatencyAware
		case "static":
			return selection.MethodStatic
		case "knn":
			return selection.MethodKNN
		case "kmeans":
			return selection.MethodKMeans
		case "svm":
			return selection.MethodSVM
		case "confidence", "ratings":
			// These are looper algorithms, not selection algorithms
			// Fall through to default
		}
	}

	// Default to static selection (use first model)
	return selection.MethodStatic
}

// getSelectionWeights returns cost and quality weights based on algorithm config.
// Uses per-decision config only (aligned with looper pattern).
func (r *OpenAIRouter) getSelectionWeights(algorithm *config.AlgorithmConfig) (float64, float64) {
	// Check per-decision algorithm config
	if algorithm != nil {
		if algorithm.AutoMix != nil && algorithm.AutoMix.CostQualityTradeoff > 0 {
			cost := algorithm.AutoMix.CostQualityTradeoff
			return cost, 1.0 - cost
		}
		if algorithm.Hybrid != nil && algorithm.Hybrid.CostWeight > 0 {
			cost := algorithm.Hybrid.CostWeight
			return cost, 1.0 - cost
		}
	}

	// Default: equal weighting (0.5 cost, 0.5 quality)
	return 0.5, 0.5
}

// getLatencyAwarePercentiles extracts TPOT/TTFT percentile settings for latency_aware selection.
// Returns (0, 0) when latency_aware is not configured for the decision.
func (r *OpenAIRouter) getLatencyAwarePercentiles(algorithm *config.AlgorithmConfig) (int, int) {
	if algorithm == nil || algorithm.LatencyAware == nil {
		return 0, 0
	}
	return algorithm.LatencyAware.TPOTPercentile, algorithm.LatencyAware.TTFTPercentile
}

// processUserFeedbackForElo automatically updates Elo ratings based on detected user feedback signals.
// This implements "automatic scoring by signals" - when the FeedbackDetector classifies user
// follow-up messages as "satisfied" or "wrong_answer", we automatically update Elo ratings.
//
// Signal mapping:
// - "satisfied" → Model performed well, record as implicit win
// - "wrong_answer" → Model performed poorly, record as implicit loss
// - "need_clarification" / "want_different" → Neutral, no Elo update
//
// For single-model feedback (no comparison), we use a "virtual opponent" approach:
// - The model competes against an expected baseline (rating 1500)
// - "satisfied" = win against baseline
// - "wrong_answer" = loss against baseline
func (r *OpenAIRouter) processUserFeedbackForElo(userFeedbackSignals []string, model string, ctx *RequestContext) {
	if len(userFeedbackSignals) == 0 || model == "" {
		return
	}

	// Get Elo selector from registry
	if r.ModelSelector == nil {
		return
	}

	eloSelector, ok := r.ModelSelector.Get(selection.MethodElo)
	if !ok || eloSelector == nil {
		return
	}

	// Process each feedback signal
	// Get decision name safely
	decisionName := ""
	if ctx.VSRSelectedDecision != nil {
		decisionName = ctx.VSRSelectedDecision.Name
	}

	for _, signal := range userFeedbackSignals {
		var feedback *selection.Feedback

		switch signal {
		case "satisfied":
			// Model performed well - record as win against virtual baseline
			feedback = &selection.Feedback{
				Query:        ctx.RequestQuery,
				WinnerModel:  model,
				LoserModel:   "", // Empty = self-feedback mode
				DecisionName: decisionName,
				Tie:          false,
			}
			logging.Infof("[AutoFeedback] User satisfied with %s, recording positive Elo feedback", model)

		case "wrong_answer":
			// Model performed poorly - record as loss against virtual baseline
			feedback = &selection.Feedback{
				Query:        ctx.RequestQuery,
				WinnerModel:  "", // Empty = model loses
				LoserModel:   model,
				DecisionName: decisionName,
				Tie:          false,
			}
			logging.Infof("[AutoFeedback] User reported wrong answer from %s, recording negative Elo feedback", model)

		default:
			// "need_clarification" and "want_different" are neutral - no Elo update
			logging.Debugf("[AutoFeedback] Neutral feedback signal %s, no Elo update", signal)
			continue
		}

		// Submit feedback to Elo selector
		if err := eloSelector.UpdateFeedback(context.Background(), feedback); err != nil {
			logging.Warnf("[AutoFeedback] Failed to update Elo: %v", err)
		}
	}
}

// buildCompressionConfig translates the YAML config into the promptcompression
// package's Config struct, applying defaults for omitted fields.
func buildCompressionConfig(pc config.PromptCompressionConfig) promptcompression.Config {
	cfg := promptcompression.DefaultConfig(pc.MaxTokens)
	if pc.TextRankWeight > 0 {
		cfg.TextRankWeight = pc.TextRankWeight
	}
	if pc.PositionWeight > 0 {
		cfg.PositionWeight = pc.PositionWeight
	}
	if pc.TFIDFWeight > 0 {
		cfg.TFIDFWeight = pc.TFIDFWeight
	}
	if pc.PositionDepth > 0 {
		cfg.PositionDepth = pc.PositionDepth
	}
	return cfg
}
