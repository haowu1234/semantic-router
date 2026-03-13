package extproc

import (
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/decision"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/promptcompression"
)

func prepareDecisionEvaluationTexts(userContent string, nonUserMessages []string) (string, string, bool) {
	evaluationText := userContent
	if evaluationText == "" && len(nonUserMessages) > 0 {
		evaluationText = strings.Join(nonUserMessages, " ")
	}
	if evaluationText == "" {
		return "", "", false
	}

	allMessages := append([]string{}, nonUserMessages...)
	if userContent != "" {
		allMessages = append(allMessages, userContent)
	}
	return evaluationText, strings.Join(allMessages, " "), true
}

func (r *OpenAIRouter) evaluateSignalsForDecision(
	evaluationText string,
	allMessagesText string,
	nonUserMessages []string,
	ctx *RequestContext,
) (*classification.SignalResults, error) {
	compressedText, skipCompressionSignals := r.compressDecisionEvaluationText(evaluationText)

	signalStart := time.Now()
	signalCtx, signalSpan := tracing.StartSpan(ctx.TraceContext, tracing.SpanSignalEvaluation)

	signals, authzErr := r.Classifier.EvaluateAllSignalsWithHeaders(
		compressedText,
		allMessagesText,
		nonUserMessages,
		ctx.Headers,
		false,
		ctx.RequestImageURL,
		evaluationText,
		skipCompressionSignals,
	)
	if authzErr != nil {
		signalSpan.End()
		logging.Errorf("[Signal Evaluation] Authz evaluation failed: %v", authzErr)
		return nil, authzErr
	}

	signalLatency := time.Since(signalStart).Milliseconds()
	r.applySignalResultsToContext(ctx, signals)
	logging.Infof("Signal evaluation results: keyword=%v, embedding=%v, domain=%v, fact_check=%v, user_feedback=%v, preference=%v, language=%v, modality=%v, jailbreak=%v, pii=%v",
		signals.MatchedKeywordRules, signals.MatchedEmbeddingRules, signals.MatchedDomainRules,
		signals.MatchedFactCheckRules, signals.MatchedUserFeedbackRules, signals.MatchedPreferenceRules,
		signals.MatchedLanguageRules, signals.MatchedModalityRules, signals.MatchedJailbreakRules, signals.MatchedPIIRules)

	tracing.EndSignalSpan(signalSpan, matchedSignalRules(signals), 1.0, signalLatency)
	ctx.TraceContext = signalCtx
	return signals, nil
}

func (r *OpenAIRouter) compressDecisionEvaluationText(evaluationText string) (string, map[string]bool) {
	compressedText := evaluationText
	var skipCompressionSignals map[string]bool
	if !r.Config.PromptCompression.Enabled || r.Config.PromptCompression.MaxTokens <= 0 {
		return compressedText, skipCompressionSignals
	}

	cfg := buildCompressionConfig(r.Config.PromptCompression)
	origTokens := promptcompression.CountTokensApprox(evaluationText)
	if r.Config.PromptCompression.MinLength > 0 && len(evaluationText) <= r.Config.PromptCompression.MinLength {
		logging.Infof("[PromptCompression] Skipped: %d chars <= min_length threshold %d", len(evaluationText), r.Config.PromptCompression.MinLength)
		return compressedText, skipCompressionSignals
	}
	if origTokens <= cfg.MaxTokens {
		return compressedText, skipCompressionSignals
	}

	result := promptcompression.Compress(evaluationText, cfg)
	logging.Infof("[PromptCompression] Compressed evaluationText: %d -> %d tokens (ratio=%.2f, kept %d sentences)",
		result.OriginalTokens, result.CompressedTokens, result.Ratio, len(result.KeptIndices))
	return result.Compressed, r.Config.PromptCompression.SkipSignalsSet()
}

func (r *OpenAIRouter) applySignalResultsToContext(ctx *RequestContext, signals *classification.SignalResults) {
	ctx.VSRMatchedKeywords = signals.MatchedKeywordRules
	ctx.VSRMatchedEmbeddings = signals.MatchedEmbeddingRules
	ctx.VSRMatchedDomains = signals.MatchedDomainRules
	ctx.VSRMatchedFactCheck = signals.MatchedFactCheckRules
	ctx.VSRMatchedUserFeedback = signals.MatchedUserFeedbackRules
	ctx.VSRMatchedPreference = signals.MatchedPreferenceRules
	ctx.VSRMatchedLanguage = signals.MatchedLanguageRules
	ctx.VSRMatchedContext = signals.MatchedContextRules
	ctx.VSRContextTokenCount = signals.TokenCount
	ctx.VSRMatchedComplexity = signals.MatchedComplexityRules
	ctx.VSRMatchedModality = signals.MatchedModalityRules
	ctx.VSRMatchedAuthz = signals.MatchedAuthzRules
	ctx.VSRMatchedJailbreak = signals.MatchedJailbreakRules
	ctx.VSRMatchedPII = signals.MatchedPIIRules

	if signals.JailbreakDetected {
		ctx.JailbreakDetected = signals.JailbreakDetected
		ctx.JailbreakType = signals.JailbreakType
		ctx.JailbreakConfidence = signals.JailbreakConfidence
	}
	if signals.PIIDetected {
		ctx.PIIDetected = signals.PIIDetected
		ctx.PIIEntities = signals.PIIEntities
	}

	r.setFactCheckFromSignals(ctx, signals.MatchedFactCheckRules)
	r.setModalityFromSignals(ctx, signals.MatchedModalityRules)
}

func matchedSignalRules(signals *classification.SignalResults) []string {
	if signals == nil {
		return nil
	}

	allMatchedRules := []string{}
	allMatchedRules = append(allMatchedRules, signals.MatchedKeywordRules...)
	allMatchedRules = append(allMatchedRules, signals.MatchedEmbeddingRules...)
	allMatchedRules = append(allMatchedRules, signals.MatchedDomainRules...)
	allMatchedRules = append(allMatchedRules, signals.MatchedFactCheckRules...)
	allMatchedRules = append(allMatchedRules, signals.MatchedUserFeedbackRules...)
	allMatchedRules = append(allMatchedRules, signals.MatchedPreferenceRules...)
	allMatchedRules = append(allMatchedRules, signals.MatchedLanguageRules...)
	allMatchedRules = append(allMatchedRules, signals.MatchedModalityRules...)
	allMatchedRules = append(allMatchedRules, signals.MatchedJailbreakRules...)
	allMatchedRules = append(allMatchedRules, signals.MatchedPIIRules...)
	return allMatchedRules
}

func (r *OpenAIRouter) evaluateDecisionMatch(
	signals *classification.SignalResults,
	ctx *RequestContext,
) (*decision.DecisionResult, error) {
	decisionStart := time.Now()
	decisionCtx, decisionSpan := tracing.StartDecisionSpan(ctx.TraceContext, "decision_evaluation")

	result, err := r.Classifier.EvaluateDecisionWithEngine(signals)
	decisionLatency := time.Since(decisionStart).Seconds()
	metrics.RecordDecisionEvaluation(decisionLatency)

	if err != nil {
		logging.Errorf("Decision evaluation error: %v", err)
		tracing.EndDecisionSpan(decisionSpan, 0.0, []string{}, r.Config.Strategy)
		ctx.TraceContext = decisionCtx
		return nil, err
	}

	confidence := 0.0
	matchedRules := []string{}
	if result != nil {
		confidence = result.Confidence
		matchedRules = result.MatchedRules
		if result.Decision != nil {
			metrics.RecordDecisionMatch(result.Decision.Name, result.Confidence)
		}
	}
	tracing.EndDecisionSpan(decisionSpan, confidence, matchedRules, r.Config.Strategy)
	ctx.TraceContext = decisionCtx
	return result, nil
}

func (r *OpenAIRouter) applyMatchedDecisionContext(
	result *decision.DecisionResult,
	ctx *RequestContext,
) (string, float64, string) {
	ctx.VSRSelectedDecision = result.Decision
	if pluginCfg := result.Decision.GetRouterReplayConfig(); pluginCfg != nil && pluginCfg.Enabled {
		ctx.RouterReplayPluginConfig = pluginCfg
	}

	categoryName := matchedDecisionCategory(result)
	ctx.VSRSelectedCategory = categoryName
	ctx.VSRSelectedDecisionConfidence = result.Confidence

	logging.Infof("Decision Evaluation Result: decision=%s, category=%s, confidence=%.3f, matched_rules=%v",
		result.Decision.Name, categoryName, result.Confidence, result.MatchedRules)
	return result.Decision.Name, result.Confidence, categoryName
}

func matchedDecisionCategory(result *decision.DecisionResult) string {
	if result == nil {
		return ""
	}
	for _, rule := range result.MatchedRules {
		if strings.HasPrefix(rule, "domain:") {
			return strings.TrimPrefix(rule, "domain:")
		}
	}
	return ""
}

func (r *OpenAIRouter) selectDecisionModel(
	result *decision.DecisionResult,
	decisionName string,
	userContent string,
	nonUserMessages []string,
	categoryName string,
	ctx *RequestContext,
) string {
	if len(result.Decision.ModelRefs) == 0 {
		ctx.VSRSelectedModel = r.Config.DefaultModel
		ctx.VSRSelectionMethod = "default"
		logging.Infof("No model refs in decision %s, using default model: %s", decisionName, r.Config.DefaultModel)
		return r.Config.DefaultModel
	}

	selectionOutcome := r.selectModelFromCandidates(
		ctx,
		result.Decision.ModelRefs,
		decisionName,
		userContent,
		nonUserMessages,
		result.Decision.Algorithm,
		categoryName,
	)
	if selectionOutcome == nil {
		selectedModel := routingModelName(result.Decision.ModelRefs[0])
		ctx.VSRSelectedModel = selectedModel
		ctx.VSRSelectionMethod = "fallback"
		logging.Warnf("Model selection returned no outcome for decision %s, using first model ref", decisionName)
		return selectedModel
	}

	ctx.routingSelection = selectionOutcome
	ctx.VSRSelectedModel = selectionOutcome.primaryModel
	ctx.VSRSelectionMethod = selectionOutcome.usedMethod
	logSelectedDecisionModel(decisionName, selectionOutcome)
	return selectionOutcome.primaryModel
}

func logSelectedDecisionModel(decisionName string, selectionOutcome *routingSelectionOutcome) {
	selectedModelRef, _ := selectionOutcome.primaryModelRef()
	if selectedModelRef != nil && selectedModelRef.LoRAName != "" {
		logging.Infof("Selected model from decision %s: %s (LoRA adapter for base model %s) using %s selection",
			decisionName, selectionOutcome.primaryModel, selectedModelRef.Model, selectionOutcome.usedMethod)
		return
	}
	logging.Infof("Selected model from decision %s: %s using %s selection",
		decisionName, selectionOutcome.primaryModel, selectionOutcome.usedMethod)
}
