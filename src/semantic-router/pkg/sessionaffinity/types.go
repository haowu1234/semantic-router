package sessionaffinity

import (
	"errors"
	"time"
)

// ErrNotFound indicates affinity state is not present for the key.
var ErrNotFound = errors.New("session affinity state not found")

type Mode string

const (
	ModeDefault Mode = "default"
	ModeBypass  Mode = "bypass"
	ModeRefresh Mode = "refresh"
	ModeReset   Mode = "reset"
)

type Action string

const (
	ActionDisabled Action = "disabled"
	ActionBind     Action = "bind"
	ActionStick    Action = "stick"
	ActionSwitch   Action = "switch"
	ActionBypass   Action = "bypass"
	ActionRefresh  Action = "refresh"
	ActionReset    Action = "reset"
)

const (
	ReasonDisabledMissingUser    = "missing_trusted_user"
	ReasonDisabledMissingSession = "missing_session_id"
	ReasonNewSession             = "new_session"
	ReasonExplicitBypass         = "explicit_bypass"
	ReasonExplicitRefresh        = "explicit_refresh"
	ReasonExplicitReset          = "explicit_reset"
	ReasonSelectorMatch          = "selector_match"
	ReasonBoundMissing           = "bound_model_missing"
	ReasonNegativeFeedback       = "negative_feedback_release"
	ReasonUpgradeThreshold       = "upgrade_threshold"
	ReasonMomentumHold           = "momentum_hold"
	ReasonMomentumRelease        = "momentum_release"
	ReasonAvailabilityFallback   = "availability_fallback"
)

// State stores the current session-affinity routing state for one trusted user/session pair.
type State struct {
	Key               string    `json:"key"`
	UserID            string    `json:"user_id"`
	SessionID         string    `json:"session_id"`
	BoundModel        string    `json:"bound_model"`
	PendingModel      string    `json:"pending_model,omitempty"`
	PendingTurns      int       `json:"pending_turns,omitempty"`
	TurnCount         int       `json:"turn_count"`
	CurrentDecision   string    `json:"current_decision,omitempty"`
	LastSelectorModel string    `json:"last_selector_model,omitempty"`
	LastAction        string    `json:"last_action,omitempty"`
	LastReason        string    `json:"last_reason,omitempty"`
	LastUsedAt        time.Time `json:"last_used_at"`
}

// Candidate is a selector-evaluated routing candidate.
type Candidate struct {
	Model string
	Score float64
}

// Request captures the information needed to evaluate an affinity decision.
type Request struct {
	UserID                string
	SessionID             string
	DecisionName          string
	Mode                  Mode
	SelectorModel         string
	Candidates            []Candidate
	MatchedFeedbackSignal []string
}

// Evaluation is the affinity manager's pre-routing recommendation.
type Evaluation struct {
	Enabled         bool
	Key             string
	UserID          string
	SessionID       string
	Mode            Mode
	SelectorModel   string
	PriorModel      string
	PreferredModels []string
	Action          Action
	Reason          string
	State           *State
	Request         Request
}

// Config controls affinity evaluation and persistence behavior.
type Config struct {
	Enabled                   bool
	RequireTrustedUser        bool
	TTL                       time.Duration
	ImmediateUpgradeThreshold float64
	ReleaseAfterPendingTurns  int
	NegativeFeedbackSignals   map[string]struct{}
}

// Store persists affinity state between turns.
type Store interface {
	Get(key string) (*State, error)
	Put(state *State, ttl time.Duration) error
	Delete(key string) error
	Close() error
}
