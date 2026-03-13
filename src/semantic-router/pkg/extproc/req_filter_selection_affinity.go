package extproc

import (
	"context"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessionaffinity"
)

type routingSelectionOutcome struct {
	usedMethod         string
	selectorModel      string
	primaryModel       string
	orderedModels      []string
	modelRefsByName    map[string]config.ModelRef
	affinityEvaluation *sessionaffinity.Evaluation
}

func (outcome *routingSelectionOutcome) primaryModelRef() (*config.ModelRef, bool) {
	if outcome == nil || outcome.primaryModel == "" {
		return nil, false
	}
	ref, ok := outcome.modelRefsByName[outcome.primaryModel]
	if !ok {
		return nil, false
	}
	return &ref, true
}

func (outcome *routingSelectionOutcome) orderedModelRefs() []config.ModelRef {
	if outcome == nil {
		return nil
	}

	ordered := make([]config.ModelRef, 0, len(outcome.orderedModels))
	for _, name := range outcome.orderedModels {
		ref, ok := outcome.modelRefsByName[name]
		if ok {
			ordered = append(ordered, ref)
		}
	}
	return ordered
}

func routingModelName(modelRef config.ModelRef) string {
	if modelRef.LoRAName != "" {
		return modelRef.LoRAName
	}
	return modelRef.Model
}

func (r *OpenAIRouter) selectModelFromCandidates(
	reqCtx *RequestContext,
	modelRefs []config.ModelRef,
	decisionName string,
	query string,
	conversationHistory []string,
	algorithm *config.AlgorithmConfig,
	categoryName string,
) *routingSelectionOutcome {
	if len(modelRefs) == 0 {
		return nil
	}

	modelRefsByName := buildModelRefIndex(modelRefs)
	selectionCtx := r.buildSelectorSelectionContext(
		reqCtx,
		modelRefs,
		decisionName,
		query,
		conversationHistory,
		algorithm,
		categoryName,
	)
	reqCtx.VSRSessionID = selectionCtx.SessionID

	if len(modelRefs) == 1 {
		outcome := r.singleModelSelectionOutcome(modelRefs[0], modelRefsByName)
		r.applyAffinityToSelectionOutcome(reqCtx, decisionName, selectionCtx, modelRefs, nil, outcome)
		r.applySessionAffinityOutcome(reqCtx, outcome)
		return outcome
	}

	selectedRef, usedMethod, result := r.selectCandidateWithAlgorithm(selectionCtx, algorithm)
	outcome := buildRoutingSelectionOutcome(modelRefsByName, modelRefs, selectedRef, usedMethod)
	r.applyAffinityToSelectionOutcome(reqCtx, decisionName, selectionCtx, modelRefs, result, outcome)
	r.applySessionAffinityOutcome(reqCtx, outcome)
	return outcome
}

func buildModelRefIndex(modelRefs []config.ModelRef) map[string]config.ModelRef {
	modelRefsByName := make(map[string]config.ModelRef, len(modelRefs))
	for _, modelRef := range modelRefs {
		modelRefsByName[routingModelName(modelRef)] = modelRef
	}
	return modelRefsByName
}

func (r *OpenAIRouter) singleModelSelectionOutcome(
	modelRef config.ModelRef,
	modelRefsByName map[string]config.ModelRef,
) *routingSelectionOutcome {
	onlyModel := routingModelName(modelRef)
	return &routingSelectionOutcome{
		usedMethod:      "single",
		selectorModel:   onlyModel,
		primaryModel:    onlyModel,
		orderedModels:   []string{onlyModel},
		modelRefsByName: modelRefsByName,
	}
}

func (r *OpenAIRouter) buildSelectorSelectionContext(
	reqCtx *RequestContext,
	modelRefs []config.ModelRef,
	decisionName string,
	query string,
	conversationHistory []string,
	algorithm *config.AlgorithmConfig,
	categoryName string,
) *selection.SelectionContext {
	costWeight, qualityWeight := r.getSelectionWeights(algorithm)
	latencyAwareTPOTPercentile, latencyAwareTTFTPercentile := r.getLatencyAwarePercentiles(algorithm)
	return &selection.SelectionContext{
		Query:                      query,
		ConversationHistory:        r.buildSelectionConversationHistory(reqCtx, conversationHistory),
		DecisionName:               decisionName,
		CategoryName:               categoryName,
		CandidateModels:            modelRefs,
		CostWeight:                 costWeight,
		QualityWeight:              qualityWeight,
		UserID:                     extractUserID(reqCtx),
		SessionID:                  r.extractSessionAffinitySessionID(reqCtx),
		LatencyAwareTPOTPercentile: latencyAwareTPOTPercentile,
		LatencyAwareTTFTPercentile: latencyAwareTTFTPercentile,
	}
}

func (r *OpenAIRouter) selectCandidateWithAlgorithm(
	selectionCtx *selection.SelectionContext,
	algorithm *config.AlgorithmConfig,
) (*config.ModelRef, string, *selection.SelectionResult) {
	method := r.getSelectionMethod(algorithm)
	usedMethod := string(method)
	selectedRef := &selectionCtx.CandidateModels[0]

	var selector selection.Selector
	if r.ModelSelector != nil {
		selector, _ = r.ModelSelector.Get(method)
	}
	if selector == nil {
		logging.Warnf("[ModelSelection] No selector available for method %s, using first model", method)
		return selectedRef, usedMethod, nil
	}

	result, err := selector.Select(context.Background(), selectionCtx)
	if err != nil {
		logging.Warnf("[ModelSelection] Selection failed: %v, using first model", err)
		return selectedRef, usedMethod, nil
	}

	if matchedRef, ok := findModelRefBySelection(selectionCtx.CandidateModels, result.SelectedModel); ok {
		selectedRef = matchedRef
		logging.Infof("[ModelSelection] Selected %s (method=%s, score=%.4f, confidence=%.2f): %s",
			result.SelectedModel, method, result.Score, result.Confidence, result.Reasoning)
		selection.RecordSelection(string(method), selectionCtx.DecisionName, result.SelectedModel, result.Score)
		return selectedRef, usedMethod, result
	}

	logging.Warnf("[ModelSelection] Selected model %s not found in candidates, using first model", result.SelectedModel)
	return selectedRef, usedMethod, result
}

func buildRoutingSelectionOutcome(
	modelRefsByName map[string]config.ModelRef,
	modelRefs []config.ModelRef,
	selectedRef *config.ModelRef,
	usedMethod string,
) *routingSelectionOutcome {
	selectorModel := routingModelName(*selectedRef)
	return &routingSelectionOutcome{
		usedMethod:      usedMethod,
		selectorModel:   selectorModel,
		primaryModel:    selectorModel,
		orderedModels:   orderedRoutingModels(modelRefs, selectorModel),
		modelRefsByName: modelRefsByName,
	}
}

func (r *OpenAIRouter) applyAffinityToSelectionOutcome(
	reqCtx *RequestContext,
	decisionName string,
	selectionCtx *selection.SelectionContext,
	modelRefs []config.ModelRef,
	result *selection.SelectionResult,
	outcome *routingSelectionOutcome,
) {
	if r.SessionAffinity == nil || !r.SessionAffinity.Enabled() || selectionCtx == nil || outcome == nil {
		return
	}

	eval, err := r.SessionAffinity.Evaluate(sessionaffinity.Request{
		UserID:                selectionCtx.UserID,
		SessionID:             selectionCtx.SessionID,
		DecisionName:          decisionName,
		Mode:                  r.extractSessionAffinityMode(reqCtx),
		SelectorModel:         outcome.selectorModel,
		Candidates:            buildSessionAffinityCandidates(modelRefs, result),
		MatchedFeedbackSignal: reqCtx.VSRMatchedUserFeedback,
	})
	if err != nil {
		logging.Warnf("SessionAffinity: evaluation failed for decision=%s session=%s: %v", decisionName, selectionCtx.SessionID, err)
		return
	}
	if eval == nil || len(eval.PreferredModels) == 0 {
		return
	}

	outcome.affinityEvaluation = eval
	outcome.primaryModel = eval.PreferredModels[0]
	outcome.orderedModels = eval.PreferredModels
}

func findModelRefBySelection(modelRefs []config.ModelRef, selectedModel string) (*config.ModelRef, bool) {
	for i := range modelRefs {
		routingName := routingModelName(modelRefs[i])
		if modelRefs[i].Model == selectedModel || modelRefs[i].LoRAName == selectedModel || routingName == selectedModel {
			return &modelRefs[i], true
		}
	}
	return nil, false
}

func orderedRoutingModels(modelRefs []config.ModelRef, primary string) []string {
	ordered := make([]string, 0, len(modelRefs))
	seen := make(map[string]struct{}, len(modelRefs))
	appendModel := func(name string) {
		if name == "" {
			return
		}
		if _, ok := seen[name]; ok {
			return
		}
		seen[name] = struct{}{}
		ordered = append(ordered, name)
	}

	appendModel(primary)
	for _, modelRef := range modelRefs {
		appendModel(routingModelName(modelRef))
	}
	return ordered
}

func buildSessionAffinityCandidates(modelRefs []config.ModelRef, result *selection.SelectionResult) []sessionaffinity.Candidate {
	candidates := make([]sessionaffinity.Candidate, 0, len(modelRefs))
	for _, modelRef := range modelRefs {
		routingName := routingModelName(modelRef)
		candidates = append(candidates, sessionaffinity.Candidate{
			Model: routingName,
			Score: candidateScoreFromSelection(modelRef, routingName, result),
		})
	}
	return candidates
}

func candidateScoreFromSelection(modelRef config.ModelRef, routingName string, result *selection.SelectionResult) float64 {
	if result == nil {
		return 0
	}
	if scoreFromBase, ok := result.AllScores[modelRef.Model]; ok {
		return scoreFromBase
	}
	if scoreFromRoutingName, ok := result.AllScores[routingName]; ok {
		return scoreFromRoutingName
	}
	if routingName == result.SelectedModel {
		return result.Score
	}
	return 0
}

func (r *OpenAIRouter) buildSelectionConversationHistory(reqCtx *RequestContext, conversationHistory []string) []string {
	history := make([]string, 0, len(conversationHistory))
	for _, entry := range conversationHistory {
		if strings.TrimSpace(entry) != "" {
			history = append(history, entry)
		}
	}

	if reqCtx == nil || reqCtx.ResponseAPICtx == nil || len(reqCtx.ResponseAPICtx.ConversationHistory) == 0 {
		return history
	}

	for _, message := range convertStoredResponsesToMessages(reqCtx.ResponseAPICtx.ConversationHistory) {
		if strings.TrimSpace(message.Content) != "" {
			history = append(history, message.Content)
		}
	}
	return history
}

func (r *OpenAIRouter) extractSessionAffinitySessionID(reqCtx *RequestContext) string {
	if reqCtx == nil {
		return ""
	}
	if reqCtx.ResponseAPICtx != nil && reqCtx.ResponseAPICtx.ConversationID != "" {
		return reqCtx.ResponseAPICtx.ConversationID
	}
	headerName := headers.VSRSessionID
	if r != nil && r.Config != nil {
		headerName = r.Config.IntelligentRouting.SessionAffinity.EffectiveSessionIDHeader()
	}
	return reqCtx.Headers[headerName]
}

func (r *OpenAIRouter) extractSessionAffinityMode(reqCtx *RequestContext) sessionaffinity.Mode {
	if reqCtx != nil && reqCtx.ResponseAPICtx != nil && reqCtx.ResponseAPICtx.OriginalRequest != nil {
		if mode, ok := reqCtx.ResponseAPICtx.OriginalRequest.Metadata["vsr.affinity_mode"]; ok {
			return sessionaffinity.Mode(strings.ToLower(strings.TrimSpace(mode)))
		}
	}
	if reqCtx == nil {
		return sessionaffinity.ModeDefault
	}
	return sessionaffinity.Mode(strings.ToLower(strings.TrimSpace(reqCtx.Headers[headers.VSRSessionAffinityMode])))
}

func (r *OpenAIRouter) applySessionAffinityOutcome(reqCtx *RequestContext, outcome *routingSelectionOutcome) {
	if reqCtx == nil {
		return
	}

	reqCtx.VSRSessionAffinityAction = ""
	reqCtx.VSRSessionAffinityReason = ""
	reqCtx.VSRSessionPriorModel = ""

	if outcome == nil || outcome.affinityEvaluation == nil {
		return
	}

	reqCtx.VSRSessionPriorModel = outcome.affinityEvaluation.PriorModel
	reqCtx.VSRSessionAffinityAction = string(outcome.affinityEvaluation.Action)
	reqCtx.VSRSessionAffinityReason = outcome.affinityEvaluation.Reason
}
