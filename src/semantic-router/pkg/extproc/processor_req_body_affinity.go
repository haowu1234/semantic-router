package extproc

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessionaffinity"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/entropy"
)

type resolvedAutoRoutingTarget struct {
	modelRef        *config.ModelRef
	routingModel    string
	endpoint        string
	endpointName    string
	affinityAction  sessionaffinity.Action
	affinityReason  string
	availabilityHit bool
}

func (r *OpenAIRouter) resolveAutoRoutingTarget(
	ctx *RequestContext,
	selectedModel string,
) (*resolvedAutoRoutingTarget, error) {
	var (
		orderedRefs []config.ModelRef
		primary     = selectedModel
	)

	if ctx != nil && ctx.routingSelection != nil {
		orderedRefs = ctx.routingSelection.orderedModelRefs()
		primary = ctx.routingSelection.primaryModel
	}

	if len(orderedRefs) == 0 && selectedModel != "" {
		orderedRefs = append(orderedRefs, config.ModelRef{Model: selectedModel})
	}
	if len(orderedRefs) == 0 {
		return nil, fmt.Errorf("no auto-routing candidates available")
	}

	var lastErr error
	for _, modelRef := range orderedRefs {
		routingModel := routingModelName(modelRef)
		selectedEndpoint, selectedEndpointName, endpointErr := r.selectEndpointForModel(ctx, routingModel)
		if endpointErr != nil {
			lastErr = endpointErr
			logging.Warnf("Auto routing candidate unavailable: model=%s err=%v", routingModel, endpointErr)
			continue
		}

		result := &resolvedAutoRoutingTarget{
			modelRef:     cloneModelRef(modelRef),
			routingModel: routingModel,
			endpoint:     selectedEndpoint,
			endpointName: selectedEndpointName,
		}

		if ctx != nil && ctx.routingSelection != nil && ctx.routingSelection.affinityEvaluation != nil {
			result.affinityAction = ctx.routingSelection.affinityEvaluation.Action
			result.affinityReason = ctx.routingSelection.affinityEvaluation.Reason
		}
		if ctx != nil && ctx.routingSelection != nil && ctx.routingSelection.affinityEvaluation != nil &&
			primary != "" && routingModel != primary {
			result.affinityAction = sessionaffinity.ActionSwitch
			result.affinityReason = sessionaffinity.ReasonAvailabilityFallback
			result.availabilityHit = true
		}
		return result, nil
	}

	if lastErr != nil {
		return nil, fmt.Errorf("all auto-routing candidates unavailable: %w", lastErr)
	}
	return nil, fmt.Errorf("all auto-routing candidates unavailable")
}

func cloneModelRef(modelRef config.ModelRef) *config.ModelRef {
	cloned := modelRef
	return &cloned
}

func buildReasoningDecisionForModel(
	decisionName string,
	confidence float64,
	modelRef *config.ModelRef,
) entropy.ReasoningDecision {
	decision := entropy.ReasoningDecision{
		Confidence:       confidence,
		DecisionReason:   "decision_engine_evaluation",
		FallbackStrategy: "decision_based_routing",
	}
	if decisionName != "" {
		decision.TopCategories = []entropy.CategoryProbability{
			{
				Category:    decisionName,
				Probability: float32(confidence),
			},
		}
	}
	if modelRef != nil && modelRef.UseReasoning != nil {
		decision.UseReasoning = *modelRef.UseReasoning
	}
	return decision
}

func (r *OpenAIRouter) commitSessionAffinitySelection(
	ctx *RequestContext,
	actualModel string,
	action sessionaffinity.Action,
	reason string,
) {
	if r == nil || r.SessionAffinity == nil || ctx == nil || ctx.routingSelection == nil || ctx.routingSelection.affinityEvaluation == nil {
		return
	}
	if err := r.SessionAffinity.Commit(ctx.routingSelection.affinityEvaluation, actualModel, action, reason); err != nil {
		logging.Warnf("SessionAffinity: commit failed for session=%s model=%s: %v", ctx.VSRSessionID, actualModel, err)
	}
}

func sessionAffinityActionFromContext(ctx *RequestContext) sessionaffinity.Action {
	if ctx == nil {
		return ""
	}
	return sessionaffinity.Action(ctx.VSRSessionAffinityAction)
}
