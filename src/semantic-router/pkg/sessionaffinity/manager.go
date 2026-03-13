package sessionaffinity

import (
	"fmt"
	"time"
)

// Manager evaluates and persists session-affinity routing state.
type Manager struct {
	cfg   Config
	store Store
}

func NewManager(cfg Config, store Store) *Manager {
	if cfg.ImmediateUpgradeThreshold <= 0 {
		cfg.ImmediateUpgradeThreshold = 0.15
	}
	if cfg.ReleaseAfterPendingTurns <= 0 {
		cfg.ReleaseAfterPendingTurns = 2
	}
	if cfg.NegativeFeedbackSignals == nil {
		cfg.NegativeFeedbackSignals = map[string]struct{}{
			"wrong_answer":   {},
			"want_different": {},
		}
	}
	return &Manager{
		cfg:   cfg,
		store: store,
	}
}

func (m *Manager) Enabled() bool {
	return m != nil && m.cfg.Enabled && m.store != nil
}

func (m *Manager) Evaluate(req Request) (*Evaluation, error) {
	eval := newEvaluation(m, req)
	if disabled := m.applyDisabledState(eval); disabled {
		return eval, nil
	}
	if err := m.loadState(eval); err != nil {
		return nil, err
	}
	if overridden := m.applyModeOverride(eval); overridden {
		return eval, nil
	}
	return m.evaluateBoundState(eval), nil
}

// Commit persists the final routed model after runtime availability fallback resolves.
func (m *Manager) Commit(eval *Evaluation, actualModel string, action Action, reason string) error {
	if eval == nil || !eval.Enabled || actualModel == "" {
		return nil
	}
	if eval.Mode == ModeBypass {
		return nil
	}

	if action == "" {
		action = eval.Action
	}
	if reason == "" {
		reason = eval.Reason
	}

	if eval.Mode == ModeReset {
		if err := m.store.Delete(eval.Key); err != nil && err != ErrNotFound {
			return err
		}
	}

	now := time.Now().UTC()
	next := &State{
		Key:               eval.Key,
		UserID:            eval.UserID,
		SessionID:         eval.SessionID,
		BoundModel:        actualModel,
		CurrentDecision:   eval.Request.DecisionName,
		LastSelectorModel: eval.SelectorModel,
		LastAction:        string(action),
		LastReason:        reason,
		LastUsedAt:        now,
	}

	if eval.State != nil {
		next.TurnCount = eval.State.TurnCount + 1
	} else {
		next.TurnCount = 1
	}

	if action == ActionStick && eval.PriorModel == actualModel && reason == ReasonMomentumHold &&
		eval.SelectorModel != "" && eval.SelectorModel != actualModel {
		if eval.State != nil && eval.State.PendingModel == eval.SelectorModel {
			next.PendingTurns = eval.State.PendingTurns + 1
		} else {
			next.PendingTurns = 1
		}
		next.PendingModel = eval.SelectorModel
	}

	return m.store.Put(next, m.cfg.TTL)
}

func (m *Manager) Close() error {
	if m == nil || m.store == nil {
		return nil
	}
	return m.store.Close()
}

func newEvaluation(m *Manager, req Request) *Evaluation {
	return &Evaluation{
		Enabled:       m.Enabled(),
		UserID:        req.UserID,
		SessionID:     req.SessionID,
		Mode:          normalizeMode(req.Mode),
		SelectorModel: req.SelectorModel,
		Request:       req,
	}
}

func (m *Manager) applyDisabledState(eval *Evaluation) bool {
	if !eval.Enabled {
		populateEvaluation(eval, ActionDisabled, "disabled", orderedModels(eval.Request.Candidates, eval.SelectorModel))
		return true
	}
	if eval.Request.SessionID == "" {
		populateEvaluation(eval, ActionDisabled, ReasonDisabledMissingSession, orderedModels(eval.Request.Candidates, eval.SelectorModel))
		return true
	}
	if m.cfg.RequireTrustedUser && eval.Request.UserID == "" {
		populateEvaluation(eval, ActionDisabled, ReasonDisabledMissingUser, orderedModels(eval.Request.Candidates, eval.SelectorModel))
		return true
	}
	return false
}

func (m *Manager) loadState(eval *Evaluation) error {
	eval.Key = buildStateKey(eval.Request.UserID, eval.Request.SessionID)
	state, err := m.store.Get(eval.Key)
	if err != nil && err != ErrNotFound {
		return err
	}
	eval.State = state
	if state != nil {
		eval.PriorModel = state.BoundModel
	}
	return nil
}

func (m *Manager) applyModeOverride(eval *Evaluation) bool {
	switch eval.Mode {
	case ModeBypass:
		populateEvaluation(eval, ActionBypass, ReasonExplicitBypass, orderedModels(eval.Request.Candidates, eval.SelectorModel))
		return true
	case ModeRefresh:
		populateEvaluation(eval, ActionRefresh, ReasonExplicitRefresh, orderedModels(eval.Request.Candidates, eval.SelectorModel))
		return true
	case ModeReset:
		populateEvaluation(eval, ActionReset, ReasonExplicitReset, orderedModels(eval.Request.Candidates, eval.SelectorModel))
		return true
	default:
		return false
	}
}

func (m *Manager) evaluateBoundState(eval *Evaluation) *Evaluation {
	state := eval.State
	req := eval.Request

	switch {
	case state == nil || state.BoundModel == "":
		populateEvaluation(eval, ActionBind, ReasonNewSession, orderedModels(req.Candidates, req.SelectorModel))
	case !containsModel(req.Candidates, state.BoundModel):
		populateEvaluation(eval, ActionSwitch, ReasonBoundMissing, orderedModels(req.Candidates, req.SelectorModel))
	case hasNegativeFeedback(req.MatchedFeedbackSignal, m.cfg.NegativeFeedbackSignals) && req.SelectorModel != "":
		populateEvaluation(eval, ActionSwitch, ReasonNegativeFeedback, orderedModelsWithFallback(req.Candidates, req.SelectorModel, state.BoundModel))
	case req.SelectorModel == "" || req.SelectorModel == state.BoundModel:
		populateEvaluation(eval, ActionStick, ReasonSelectorMatch, orderedModelsWithFallback(req.Candidates, state.BoundModel, ""))
	case m.shouldUpgradeImmediately(req, state):
		populateEvaluation(eval, ActionSwitch, ReasonUpgradeThreshold, orderedModelsWithFallback(req.Candidates, req.SelectorModel, state.BoundModel))
	case m.shouldReleaseMomentum(req, state):
		populateEvaluation(eval, ActionSwitch, ReasonMomentumRelease, orderedModelsWithFallback(req.Candidates, req.SelectorModel, state.BoundModel))
	default:
		populateEvaluation(eval, ActionStick, ReasonMomentumHold, orderedModelsWithFallback(req.Candidates, state.BoundModel, req.SelectorModel))
	}

	return eval
}

func (m *Manager) shouldUpgradeImmediately(req Request, state *State) bool {
	selectorScore := scoreForModel(req.Candidates, req.SelectorModel)
	boundScore := scoreForModel(req.Candidates, state.BoundModel)
	return selectorScore-boundScore >= m.cfg.ImmediateUpgradeThreshold
}

func (m *Manager) shouldReleaseMomentum(req Request, state *State) bool {
	pendingTurns := 1
	if state.PendingModel == req.SelectorModel {
		pendingTurns = state.PendingTurns + 1
	}
	return pendingTurns >= m.cfg.ReleaseAfterPendingTurns
}

func populateEvaluation(eval *Evaluation, action Action, reason string, preferredModels []string) {
	eval.Action = action
	eval.Reason = reason
	eval.PreferredModels = preferredModels
}

func buildStateKey(userID, sessionID string) string {
	if userID == "" {
		return sessionID
	}
	return fmt.Sprintf("%s:%s", userID, sessionID)
}

func normalizeMode(mode Mode) Mode {
	switch mode {
	case ModeBypass, ModeRefresh, ModeReset:
		return mode
	default:
		return ModeDefault
	}
}

func orderedModels(candidates []Candidate, primary string) []string {
	return orderedModelsWithFallback(candidates, primary, "")
}

func orderedModelsWithFallback(candidates []Candidate, primary string, secondary string) []string {
	ordered := make([]string, 0, len(candidates))
	seen := make(map[string]struct{}, len(candidates))
	appendModel := func(model string) {
		if model == "" {
			return
		}
		if _, ok := seen[model]; ok {
			return
		}
		for _, candidate := range candidates {
			if candidate.Model == model {
				ordered = append(ordered, model)
				seen[model] = struct{}{}
				return
			}
		}
	}

	appendModel(primary)
	appendModel(secondary)
	for _, candidate := range candidates {
		appendModel(candidate.Model)
	}
	return ordered
}

func containsModel(candidates []Candidate, model string) bool {
	for _, candidate := range candidates {
		if candidate.Model == model {
			return true
		}
	}
	return false
}

func scoreForModel(candidates []Candidate, model string) float64 {
	for _, candidate := range candidates {
		if candidate.Model == model {
			return candidate.Score
		}
	}
	return 0
}

func hasNegativeFeedback(signals []string, allowed map[string]struct{}) bool {
	for _, signal := range signals {
		if _, ok := allowed[signal]; ok {
			return true
		}
	}
	return false
}
