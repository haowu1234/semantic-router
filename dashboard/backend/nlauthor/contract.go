package nlauthor

// SchemaVersion identifies the current NL authoring contract and schema shape.
const SchemaVersion = "v1alpha1"

type FieldType string

const (
	FieldTypeString      FieldType = "string"
	FieldTypeNumber      FieldType = "number"
	FieldTypeBoolean     FieldType = "boolean"
	FieldTypeStringArray FieldType = "string[]"
	FieldTypeNumberArray FieldType = "number[]"
	FieldTypeSelect      FieldType = "select"
	FieldTypeJSON        FieldType = "json"
)

type ConstructKind string

const (
	ConstructRoute     ConstructKind = "route"
	ConstructSignal    ConstructKind = "signal"
	ConstructPlugin    ConstructKind = "plugin"
	ConstructAlgorithm ConstructKind = "algorithm"
	ConstructBackend   ConstructKind = "backend"
)

type OperationMode string

const (
	OperationGenerate OperationMode = "generate"
	OperationModify   OperationMode = "modify"
	OperationFix      OperationMode = "fix"
)

type PlannerResultStatus string

const (
	PlannerStatusReady              PlannerResultStatus = "ready"
	PlannerStatusNeedsClarification PlannerResultStatus = "needs_clarification"
	PlannerStatusUnsupported        PlannerResultStatus = "unsupported"
	PlannerStatusError              PlannerResultStatus = "error"
)

// FieldSchema describes one construct field the planner is allowed to use.
type FieldSchema struct {
	Key         string    `json:"key"`
	Label       string    `json:"label"`
	Type        FieldType `json:"type"`
	Options     []string  `json:"options,omitempty"`
	Required    bool      `json:"required,omitempty"`
	Placeholder string    `json:"placeholder,omitempty"`
	Description string    `json:"description,omitempty"`
}

// TypeSchemaEntry describes one allowed type within a construct category.
type TypeSchemaEntry struct {
	TypeName    string        `json:"typeName"`
	Description string        `json:"description"`
	Fields      []FieldSchema `json:"fields,omitempty"`
}

// SchemaManifest is the canonical NL authoring manifest exposed by the backend.
type SchemaManifest struct {
	Version    string            `json:"version"`
	Signals    []TypeSchemaEntry `json:"signals"`
	Plugins    []TypeSchemaEntry `json:"plugins"`
	Algorithms []TypeSchemaEntry `json:"algorithms"`
	Backends   []TypeSchemaEntry `json:"backends"`
}

// Capabilities reports which parts of the NL authoring surface are live.
type Capabilities struct {
	Enabled               bool            `json:"enabled"`
	Preview               bool            `json:"preview"`
	PlannerAvailable      bool            `json:"plannerAvailable"`
	PlannerBackend        string          `json:"plannerBackend"`
	SchemaVersion         string          `json:"schemaVersion"`
	SupportedOperations   []OperationMode `json:"supportedOperations"`
	SupportedConstructs   []ConstructKind `json:"supportedConstructs"`
	SupportedSignalTypes  []string        `json:"supportedSignalTypes,omitempty"`
	SupportedPluginTypes  []string        `json:"supportedPluginTypes,omitempty"`
	SupportedBackendTypes []string        `json:"supportedBackendTypes,omitempty"`
	SupportedAlgoTypes    []string        `json:"supportedAlgorithmTypes,omitempty"`
	SupportsClarification bool            `json:"supportsClarification"`
	SupportsSessionAPI    bool            `json:"supportsSessionApi"`
	SupportsStreaming     bool            `json:"supportsStreaming"`
	SupportsApply         bool            `json:"supportsApply"`
	ReadonlyMode          bool            `json:"readonlyMode"`
}

// SymbolInfo mirrors the Builder symbol table shape without coupling to WASM types.
type SymbolInfo struct {
	Name string `json:"name"`
	Type string `json:"type"`
}

// SymbolSnapshot captures the current Builder symbols passed to the planner.
type SymbolSnapshot struct {
	Signals  []SymbolInfo `json:"signals,omitempty"`
	Models   []string     `json:"models,omitempty"`
	Plugins  []string     `json:"plugins,omitempty"`
	Backends []SymbolInfo `json:"backends,omitempty"`
	Routes   []string     `json:"routes,omitempty"`
}

// DiagnosticSnapshot is the planner-facing diagnostic shape from the Builder.
type DiagnosticSnapshot struct {
	Level   string `json:"level"`
	Message string `json:"message"`
	Line    int    `json:"line,omitempty"`
	Column  int    `json:"column,omitempty"`
}

// SessionContext is the canonical payload shared across future NL session APIs.
type SessionContext struct {
	BaseDSL     string               `json:"baseDsl,omitempty"`
	Symbols     *SymbolSnapshot      `json:"symbols,omitempty"`
	Diagnostics []DiagnosticSnapshot `json:"diagnostics,omitempty"`
}

// IntentIR is the structured planner output consumed by deterministic Builder code.
type IntentIR struct {
	Version   string                   `json:"version"`
	Operation OperationMode            `json:"operation"`
	Intents   []map[string]interface{} `json:"intents"`
}

// ClarificationOption is one user-visible disambiguation choice.
type ClarificationOption struct {
	ID          string `json:"id"`
	Label       string `json:"label"`
	Description string `json:"description,omitempty"`
}

// Clarification asks the user to resolve planner ambiguity before draft creation.
type Clarification struct {
	Question string                `json:"question"`
	Options  []ClarificationOption `json:"options"`
}

// PlannerWarning carries non-fatal contract or planner concerns for review UI.
type PlannerWarning struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

// PlannerResult is the canonical planner response for future NL turn APIs.
type PlannerResult struct {
	Status        PlannerResultStatus `json:"status"`
	IntentIR      *IntentIR           `json:"intentIr,omitempty"`
	Explanation   string              `json:"explanation,omitempty"`
	Clarification *Clarification      `json:"clarification,omitempty"`
	Warnings      []PlannerWarning    `json:"warnings,omitempty"`
	Error         string              `json:"error,omitempty"`
}

// SessionCreateRequest reserves the session bootstrap contract for later phases.
type SessionCreateRequest struct {
	SchemaVersion string         `json:"schemaVersion,omitempty"`
	Context       SessionContext `json:"context,omitempty"`
}

// SessionCreateResponse reserves the session bootstrap response for later phases.
type SessionCreateResponse struct {
	SessionID     string       `json:"sessionId"`
	SchemaVersion string       `json:"schemaVersion"`
	ExpiresAt     string       `json:"expiresAt,omitempty"`
	Capabilities  Capabilities `json:"capabilities"`
}

// TurnRequest reserves the planner-turn contract for later phases.
type TurnRequest struct {
	Prompt        string         `json:"prompt"`
	ModeHint      OperationMode  `json:"modeHint,omitempty"`
	SchemaVersion string         `json:"schemaVersion,omitempty"`
	Context       SessionContext `json:"context,omitempty"`
}

// TurnResponse returns one planner turn plus session metadata for future UI flows.
type TurnResponse struct {
	SessionID     string        `json:"sessionId"`
	TurnID        string        `json:"turnId"`
	SchemaVersion string        `json:"schemaVersion"`
	ExpiresAt     string        `json:"expiresAt,omitempty"`
	Result        PlannerResult `json:"result"`
}

// DefaultCapabilities reports the currently shipped NL contract surface.
func DefaultCapabilities(readonlyMode bool) Capabilities {
	return Capabilities{
		Enabled:               true,
		Preview:               true,
		PlannerAvailable:      false,
		PlannerBackend:        "unconfigured",
		SchemaVersion:         SchemaVersion,
		SupportedOperations:   []OperationMode{},
		SupportedConstructs:   []ConstructKind{},
		SupportsClarification: false,
		SupportsSessionAPI:    true,
		SupportsStreaming:     false,
		SupportsApply:         !readonlyMode,
		ReadonlyMode:          readonlyMode,
	}
}
