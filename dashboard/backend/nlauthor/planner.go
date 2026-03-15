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
type UnavailablePlanner struct{}

func (UnavailablePlanner) BackendName() string {
	return "unconfigured"
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

func (UnavailablePlanner) Plan(_ context.Context, _ Session, _ TurnRequest) (PlannerResult, error) {
	return PlannerResult{
		Status:      PlannerStatusUnsupported,
		Explanation: "NL planning is not configured on this dashboard yet.",
		Warnings: []PlannerWarning{
			{
				Code:    "planner_unavailable",
				Message: "The Builder NL session API is live, but the planner backend is still disabled.",
			},
		},
		Error: "planner backend is unavailable",
	}, nil
}
