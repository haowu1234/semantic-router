package nlauthor

// PlannerSupport describes the backend-advertised NL capability subset for one planner.
type PlannerSupport struct {
	Operations     []OperationMode
	Constructs     []ConstructKind
	SignalTypes    []string
	PluginTypes    []string
	AlgorithmTypes []string
	BackendTypes   []string
}
