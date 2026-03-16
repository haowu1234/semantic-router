package nlauthor

import "context"

// Planner is the backend-owned seam for generating structured NL authoring plans.
type Planner interface {
	BackendName() string
	Available() bool
	SupportsClarification() bool
	Support() PlannerSupport
	Plan(ctx context.Context, session Session, request TurnRequest) (PlannerResult, error)
}

// UnavailablePlanner keeps the session API live while no model-backed planner is configured.
type UnavailablePlanner struct {
	backendName    string
	explanation    string
	warningCode    string
	warningMessage string
}

func NewUnavailablePlanner(backendName, explanation, warningCode, warningMessage string) UnavailablePlanner {
	return UnavailablePlanner{
		backendName:    backendName,
		explanation:    explanation,
		warningCode:    warningCode,
		warningMessage: warningMessage,
	}
}

func (p UnavailablePlanner) BackendName() string {
	if p.backendName == "" {
		return "unconfigured"
	}
	return p.backendName
}

func (UnavailablePlanner) Available() bool {
	return false
}

func (UnavailablePlanner) SupportsClarification() bool {
	return false
}

func (UnavailablePlanner) Support() PlannerSupport {
	return PlannerSupport{}
}

func (p UnavailablePlanner) Plan(_ context.Context, _ Session, _ TurnRequest) (PlannerResult, error) {
	explanation := p.explanation
	if explanation == "" {
		explanation = "NL planning is not configured on this dashboard yet."
	}
	warningCode := p.warningCode
	if warningCode == "" {
		warningCode = "planner_unavailable"
	}
	warningMessage := p.warningMessage
	if warningMessage == "" {
		warningMessage = "The Builder NL session API is live, but the planner backend is still disabled."
	}
	return PlannerResult{
		Status:      PlannerStatusUnsupported,
		Explanation: explanation,
		Warnings: []PlannerWarning{
			{
				Code:    warningCode,
				Message: warningMessage,
			},
		},
		Error: "planner backend is unavailable",
	}, nil
}
