package nlauthor

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
)

var (
	signalHeaderPattern  = regexp.MustCompile(`(?m)^SIGNAL\s+([a-zA-Z0-9_]+)\s+([a-zA-Z0-9_"-]+)\s*\{`)
	pluginHeaderPattern  = regexp.MustCompile(`(?m)^PLUGIN\s+([a-zA-Z0-9_"-]+)\s+([a-zA-Z0-9_]+)\s*\{`)
	backendHeaderPattern = regexp.MustCompile(`(?m)^BACKEND\s+([a-zA-Z0-9_]+)\s+([a-zA-Z0-9_"-]+)\s*\{`)
	routeHeaderExact     = regexp.MustCompile(`(?m)^ROUTE\s+([a-zA-Z0-9_"-]+)(?:\s+\(.*\))?\s*\{`)
)

func NewBuiltinBuilderToolSource(manifest SchemaManifest, support PlannerSupport) PlannerToolSource {
	return builtinBuilderToolSource{
		manifest: manifest,
		support:  support,
	}
}

type builtinBuilderToolSource struct {
	manifest SchemaManifest
	support  PlannerSupport
}

func (s builtinBuilderToolSource) SourceName() string {
	return PlannerToolSourceBuiltinBuilder
}

func (s builtinBuilderToolSource) Tools(_ Session, _ TurnRequest) []PlannerTool {
	return []PlannerTool{
		staticPlannerTool{
			definition: toolDefinition("get_capabilities", "Inspect the currently supported planner subset and operations.", emptyObjectSchema()),
			invokeFn: func(_ context.Context, _ Session, _ TurnRequest, _ json.RawMessage) (PlannerToolResult, error) {
				return marshalToolJSON(s.support)
			},
		},
		staticPlannerTool{
			definition: toolDefinition("get_schema_subset", "Look up supported schema entries for one construct type and optional concrete type.", objectSchema(requiredStringField("construct"), optionalStringField("typeName"))),
			invokeFn: func(_ context.Context, _ Session, _ TurnRequest, arguments json.RawMessage) (PlannerToolResult, error) {
				var payload struct {
					Construct string `json:"construct"`
					TypeName  string `json:"typeName"`
				}
				if err := json.Unmarshal(arguments, &payload); err != nil {
					return PlannerToolResult{}, fmt.Errorf("invalid schema lookup arguments: %w", err)
				}
				construct := strings.TrimSpace(payload.Construct)
				typeName := strings.TrimSpace(payload.TypeName)
				switch ConstructKind(construct) {
				case ConstructRoute:
					return marshalToolJSON(routeSchemaEntries(typeName, s.support))
				case ConstructSignal:
					return marshalToolJSON(filterManifestEntries(s.manifest.Signals, requestedTypes(typeName, s.support.SignalTypes)))
				case ConstructPlugin:
					return marshalToolJSON(filterManifestEntries(s.manifest.Plugins, requestedTypes(typeName, s.support.PluginTypes)))
				case ConstructBackend:
					return marshalToolJSON(filterManifestEntries(s.manifest.Backends, requestedTypes(typeName, s.support.BackendTypes)))
				case ConstructAlgorithm:
					return marshalToolJSON(filterManifestEntries(s.manifest.Algorithms, requestedTypes(typeName, s.support.AlgorithmTypes)))
				default:
					return PlannerToolResult{}, fmt.Errorf("unsupported construct %q", construct)
				}
			},
		},
		staticPlannerTool{
			definition: toolDefinition("list_symbols", "Inspect the current Builder symbol snapshot.", emptyObjectSchema()),
			invokeFn: func(_ context.Context, session Session, _ TurnRequest, _ json.RawMessage) (PlannerToolResult, error) {
				if session.Context.Symbols == nil {
					return marshalToolJSON(map[string]any{"symbols": nil})
				}
				return marshalToolJSON(session.Context.Symbols)
			},
		},
		staticPlannerTool{
			definition: toolDefinition("get_diagnostics", "Inspect current canonical Builder diagnostics.", emptyObjectSchema()),
			invokeFn: func(_ context.Context, session Session, _ TurnRequest, _ json.RawMessage) (PlannerToolResult, error) {
				return marshalToolJSON(session.Context.Diagnostics)
			},
		},
		staticPlannerTool{
			definition: toolDefinition("get_signal", "Look up one existing signal by name from the current DSL.", objectSchema(requiredStringField("name"))),
			invokeFn: func(_ context.Context, session Session, _ TurnRequest, arguments json.RawMessage) (PlannerToolResult, error) {
				return lookupDSLBlock(session.Context.BaseDSL, arguments, extractSignalBlock)
			},
		},
		staticPlannerTool{
			definition: toolDefinition("get_plugin", "Look up one existing plugin template by name from the current DSL.", objectSchema(requiredStringField("name"))),
			invokeFn: func(_ context.Context, session Session, _ TurnRequest, arguments json.RawMessage) (PlannerToolResult, error) {
				return lookupDSLBlock(session.Context.BaseDSL, arguments, extractPluginBlock)
			},
		},
		staticPlannerTool{
			definition: toolDefinition("get_backend", "Look up one existing backend by name from the current DSL.", objectSchema(requiredStringField("name"))),
			invokeFn: func(_ context.Context, session Session, _ TurnRequest, arguments json.RawMessage) (PlannerToolResult, error) {
				return lookupDSLBlock(session.Context.BaseDSL, arguments, extractBackendBlock)
			},
		},
		staticPlannerTool{
			definition: toolDefinition("get_route", "Look up one existing route by name from the current DSL.", objectSchema(requiredStringField("name"))),
			invokeFn: func(_ context.Context, session Session, _ TurnRequest, arguments json.RawMessage) (PlannerToolResult, error) {
				return lookupDSLBlock(session.Context.BaseDSL, arguments, extractRouteBlock)
			},
		},
		staticPlannerTool{
			definition: toolDefinition("validate_candidate_intent_ir", "Validate a candidate intent IR against the current manifest contract.", objectSchema(requiredAnyField("intentIr"))),
			invokeFn: func(_ context.Context, _ Session, _ TurnRequest, arguments json.RawMessage) (PlannerToolResult, error) {
				var payload struct {
					IntentIR *IntentIR `json:"intentIr"`
				}
				if err := json.Unmarshal(arguments, &payload); err != nil {
					return PlannerToolResult{}, fmt.Errorf("invalid validation arguments: %w", err)
				}
				result := PlannerResult{
					Status:      PlannerStatusReady,
					Explanation: "Candidate intent validation.",
					IntentIR:    payload.IntentIR,
				}
				err := validatePlannerResult(result, s.manifest)
				if err == nil {
					return marshalToolJSON(map[string]any{"valid": true})
				}
				return marshalToolJSON(map[string]any{"valid": false, "error": err.Error()})
			},
		},
	}
}

func requestedTypes(typeName string, fallback []string) []string {
	if typeName == "" {
		return fallback
	}
	return []string{typeName}
}

func toolDefinition(name, description string, schema json.RawMessage) PlannerToolDefinition {
	return PlannerToolDefinition{
		Name:        name,
		Description: description,
		InputSchema: schema,
		Readonly:    true,
		Source:      PlannerToolSourceBuiltinBuilder,
	}
}

func lookupDSLBlock(baseDSL string, arguments json.RawMessage, extractor func(string, string) (map[string]any, bool)) (PlannerToolResult, error) {
	name := parseOptionalName(arguments)
	if name == "" {
		return PlannerToolResult{}, fmt.Errorf("name is required")
	}
	result, ok := extractor(baseDSL, name)
	if !ok {
		return marshalToolJSON(map[string]any{
			"name":  name,
			"found": false,
		})
	}
	return marshalToolJSON(result)
}

func extractSignalBlock(baseDSL, name string) (map[string]any, bool) {
	return extractTopLevelBlock(baseDSL, func(header string) bool {
		match := signalHeaderPattern.FindStringSubmatch(header)
		return len(match) == 3 && strings.Trim(match[2], `"`) == name
	}, func(header string) map[string]any {
		match := signalHeaderPattern.FindStringSubmatch(header)
		return map[string]any{
			"found":      true,
			"construct":  "signal",
			"signalType": match[1],
			"name":       strings.Trim(match[2], `"`),
		}
	})
}

func extractPluginBlock(baseDSL, name string) (map[string]any, bool) {
	return extractTopLevelBlock(baseDSL, func(header string) bool {
		match := pluginHeaderPattern.FindStringSubmatch(header)
		return len(match) == 3 && strings.Trim(match[1], `"`) == name
	}, func(header string) map[string]any {
		match := pluginHeaderPattern.FindStringSubmatch(header)
		return map[string]any{
			"found":      true,
			"construct":  "plugin",
			"name":       strings.Trim(match[1], `"`),
			"pluginType": match[2],
		}
	})
}

func extractBackendBlock(baseDSL, name string) (map[string]any, bool) {
	return extractTopLevelBlock(baseDSL, func(header string) bool {
		match := backendHeaderPattern.FindStringSubmatch(header)
		return len(match) == 3 && strings.Trim(match[2], `"`) == name
	}, func(header string) map[string]any {
		match := backendHeaderPattern.FindStringSubmatch(header)
		return map[string]any{
			"found":       true,
			"construct":   "backend",
			"name":        strings.Trim(match[2], `"`),
			"backendType": match[1],
		}
	})
}

func extractRouteBlock(baseDSL, name string) (map[string]any, bool) {
	return extractTopLevelBlock(baseDSL, func(header string) bool {
		match := routeHeaderExact.FindStringSubmatch(header)
		return len(match) == 2 && strings.Trim(match[1], `"`) == name
	}, func(header string) map[string]any {
		match := routeHeaderExact.FindStringSubmatch(header)
		return map[string]any{
			"found":     true,
			"construct": "route",
			"name":      strings.Trim(match[1], `"`),
		}
	})
}

func extractTopLevelBlock(
	baseDSL string,
	headerMatches func(header string) bool,
	buildMetadata func(header string) map[string]any,
) (map[string]any, bool) {
	lines := strings.Split(baseDSL, "\n")
	for i := 0; i < len(lines); i++ {
		header := strings.TrimSpace(lines[i])
		if !headerMatches(header) {
			continue
		}
		braceBalance := strings.Count(lines[i], "{") - strings.Count(lines[i], "}")
		end := i
		for braceBalance > 0 && end+1 < len(lines) {
			end++
			braceBalance += strings.Count(lines[end], "{")
			braceBalance -= strings.Count(lines[end], "}")
		}
		metadata := buildMetadata(header)
		metadata["snippet"] = strings.Join(lines[i:end+1], "\n")
		return metadata, true
	}
	return nil, false
}

func emptyObjectSchema() json.RawMessage {
	return json.RawMessage(`{"type":"object","additionalProperties":false,"properties":{}}`)
}

type schemaField struct {
	name     string
	required bool
	schema   string
}

func objectSchema(fields ...schemaField) json.RawMessage {
	var properties strings.Builder
	properties.WriteString(`{"type":"object","additionalProperties":false,"properties":{`)

	required := make([]string, 0)
	for index, field := range fields {
		if index > 0 {
			properties.WriteByte(',')
		}
		properties.WriteString(fmt.Sprintf(`%q:%s`, field.name, field.schema))
		if field.required {
			required = append(required, field.name)
		}
	}
	properties.WriteString(`}`)
	if len(required) > 0 {
		properties.WriteString(`,"required":[`)
		for index, name := range required {
			if index > 0 {
				properties.WriteByte(',')
			}
			properties.WriteString(fmt.Sprintf(`%q`, name))
		}
		properties.WriteString(`]`)
	}
	properties.WriteString(`}`)
	return json.RawMessage(properties.String())
}

func requiredStringField(name string) schemaField {
	return schemaField{name: name, required: true, schema: `{"type":"string"}`}
}

func optionalStringField(name string) schemaField {
	return schemaField{name: name, schema: `{"type":"string"}`}
}

func requiredAnyField(name string) schemaField {
	return schemaField{name: name, required: true, schema: `{}`}
}
