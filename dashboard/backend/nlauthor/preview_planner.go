package nlauthor

import (
	"context"
	"fmt"
	"regexp"
	"slices"
	"strconv"
	"strings"
)

var (
	quotedStringPattern = regexp.MustCompile(`"([^"]+)"|'([^']+)'`)
	namePattern         = regexp.MustCompile(`(?i)\b(?:named|called)\s+([a-zA-Z0-9_-]+)\b`)
	modelPattern        = regexp.MustCompile(`(?i)\bmodel\s+["']?([a-zA-Z0-9_./:-]+)["']?`)
	hostPortPattern     = regexp.MustCompile(`\b([a-zA-Z0-9.-]+):(\d{2,5})\b`)
	priorityPattern     = regexp.MustCompile(`(?i)\bpriority\s+(\d+)\b`)
	numberPattern       = regexp.MustCompile(`\b\d+(?:\.\d+)?\b`)
	routeHeaderPattern  = regexp.MustCompile(`(?m)^ROUTE\s+([a-zA-Z0-9_-]+)(?:\s+\(description\s*=\s*"([^"]*)"\))?\s*\{`)
	routeWhenPattern    = regexp.MustCompile(`(?m)^\s*WHEN\s+([a-zA-Z0-9_]+)\("([^"]+)"\)\s*$`)
	routeModelLine      = regexp.MustCompile(`(?m)^\s*MODEL\s+"([^"]+)"(?:\s*\(([^)]*)\))?\s*$`)
	globalPromptGuard   = regexp.MustCompile(`(?s)prompt_guard:\s*\{(.*?)\}`)
	boolFieldPattern    = regexp.MustCompile(`(?m)^\s*enabled:\s*(true|false)\s*$`)
)

// PreviewPlanner is a bounded deterministic planner for the first NL preview.
type PreviewPlanner struct {
	manifest SchemaManifest
}

func NewPreviewPlanner(manifest SchemaManifest) PreviewPlanner {
	if manifest.Version == "" {
		manifest = DefaultSchemaManifest()
	}
	return PreviewPlanner{manifest: manifest}
}

func (p PreviewPlanner) BackendName() string {
	return "preview-rulebased"
}

func (p PreviewPlanner) Available() bool {
	return true
}

func (p PreviewPlanner) SupportsClarification() bool {
	return true
}

func (p PreviewPlanner) Support() PlannerSupport {
	return PlannerSupport{
		Operations:     []OperationMode{OperationGenerate, OperationModify, OperationFix},
		Constructs:     []ConstructKind{ConstructRoute, ConstructSignal, ConstructPlugin, ConstructBackend},
		SignalTypes:    []string{"keyword", "embedding"},
		PluginTypes:    []string{"semantic_cache", "memory", "fast_response", "system_prompt"},
		AlgorithmTypes: []string{},
		BackendTypes:   []string{"vllm_endpoint"},
	}
}

func (p PreviewPlanner) Plan(_ context.Context, session Session, request TurnRequest) (PlannerResult, error) {
	prompt := strings.TrimSpace(request.Prompt)
	lowerPrompt := strings.ToLower(prompt)
	if len(prompt) < 8 {
		return plannerClarification(
			"Do you want to create a new draft, modify the current config, or fix an existing issue?",
			[]ClarificationOption{
				{ID: "generate", Label: "Create draft", Description: "Start a new signal, route, plugin, or backend draft."},
				{ID: "modify", Label: "Modify config", Description: "Change or delete an existing construct."},
				{ID: "fix", Label: "Fix issue", Description: "Repair an existing DSL problem."},
			},
		), nil
	}

	if request.ModeHint == OperationFix || containsAny(lowerPrompt, "fix", "repair", "diagnostic", "compile error", "validation error") {
		if len(session.Context.Diagnostics) == 0 {
			return plannerClarification(
				"Fix mode needs current diagnostics. Should I treat this as a config change instead?",
				[]ClarificationOption{
					{ID: "modify", Label: "Modify config", Description: "Use this prompt to add, update, or delete config constructs."},
					{ID: "generate", Label: "Create draft", Description: "Create a new construct draft instead of repairing diagnostics."},
				},
			), nil
		}
		return p.planRepair(prompt, lowerPrompt, session.Context), nil
	}

	constructs := p.detectConstructs(lowerPrompt)
	if len(constructs) == 0 {
		return plannerClarification(
			"Which construct should I work on in Builder NL mode?",
			[]ClarificationOption{
				{ID: "route", Label: "Route", Description: "Create or delete a route."},
				{ID: "signal", Label: "Signal", Description: "Create or delete a signal."},
				{ID: "plugin", Label: "Plugin", Description: "Create or delete a plugin template."},
				{ID: "backend", Label: "Backend", Description: "Create or delete a backend declaration."},
			},
		), nil
	}
	if len(constructs) > 1 {
		options := make([]ClarificationOption, 0, len(constructs))
		for _, construct := range constructs {
			options = append(options, ClarificationOption{
				ID:          string(construct),
				Label:       strings.Title(string(construct)),
				Description: "Focus the preview planner on one construct at a time.",
			})
		}
		return plannerClarification(
			"This preview planner handles one construct per turn. Which construct should I focus on first?",
			options,
		), nil
	}

	action := detectAction(lowerPrompt, request.ModeHint, session.Context.BaseDSL != "")
	switch constructs[0] {
	case ConstructSignal:
		return p.planSignal(prompt, lowerPrompt, session.Context.Symbols, action), nil
	case ConstructPlugin:
		return p.planPlugin(prompt, lowerPrompt, session.Context.Symbols, action), nil
	case ConstructBackend:
		return p.planBackend(prompt, lowerPrompt, session.Context.Symbols, action), nil
	case ConstructRoute:
		return p.planRoute(prompt, lowerPrompt, session.Context.Symbols, action), nil
	default:
		return PlannerResult{
			Status:      PlannerStatusUnsupported,
			Explanation: "The preview planner could not map this request to a supported construct.",
			Error:       "unsupported construct",
		}, nil
	}
}

func (p PreviewPlanner) planRepair(prompt, lowerPrompt string, context SessionContext) PlannerResult {
	if result, ok := p.planThresholdRepairReady(lowerPrompt, context); ok {
		return result
	}
	if result, ok := p.planUndefinedSignalRepairReady(prompt, lowerPrompt, context); ok {
		return result
	}
	if result, ok := p.planConstraintRepair(context.Diagnostics); ok {
		return result
	}
	if result, ok := p.planUndefinedSignalRepair(context); ok {
		return result
	}
	if result, ok := p.planUnknownTypeRepair(context.Diagnostics); ok {
		return result
	}

	return PlannerResult{
		Status:      PlannerStatusUnsupported,
		Explanation: "Preview repair can only clarify undefined signal references, unknown construct types, or invalid threshold values.",
		Warnings: []PlannerWarning{
			{Code: "repair_pattern_unimplemented", Message: "Open in DSL for manual edits if the validator output does not match the preview repair subset."},
		},
		Error: "repair pattern is not supported yet",
	}
}

func (p PreviewPlanner) planThresholdRepairReady(lowerPrompt string, context SessionContext) (PlannerResult, bool) {
	if !hasThresholdDiagnostic(context.Diagnostics) {
		return PlannerResult{}, false
	}

	replacement, ok := extractNumberAfterWord(lowerPrompt, "use threshold")
	if !ok {
		replacement, ok = extractNumberAfterWord(lowerPrompt, "threshold")
	}
	if !ok || replacement < 0 || replacement > 1 {
		return PlannerResult{}, false
	}

	enabled, ok := extractPromptGuardEnabled(context.BaseDSL)
	if !ok {
		return PlannerResult{}, false
	}

	return plannerReady(IntentIR{
		Version:   "1.0",
		Operation: OperationFix,
		Intents: []map[string]interface{}{
			{
				"type": "global",
				"fields": map[string]interface{}{
					"prompt_guard": map[string]interface{}{
						"enabled":   enabled,
						"threshold": replacement,
					},
				},
			},
		},
	}, fmt.Sprintf("Repair prompt_guard threshold by setting it to %.2f.", replacement), nil), true
}

func (p PreviewPlanner) planUndefinedSignalRepairReady(prompt, lowerPrompt string, context SessionContext) (PlannerResult, bool) {
	if !hasUndefinedSignalDiagnostic(context.Diagnostics) {
		return PlannerResult{}, false
	}

	chosenSignalName, chosenSignalType, ok := matchExistingSignal(prompt, lowerPrompt, context.Symbols)
	if !ok {
		return PlannerResult{}, false
	}

	route, ok := extractSingleSimpleRoute(context.BaseDSL)
	if !ok {
		return PlannerResult{}, false
	}

	if chosenSignalType == "" {
		chosenSignalType = route.SignalType
	}

	return plannerReady(IntentIR{
		Version:   "1.0",
		Operation: OperationFix,
		Intents: []map[string]interface{}{
			{
				"type":        "route",
				"name":        route.Name,
				"description": route.Description,
				"priority":    route.Priority,
				"condition": map[string]interface{}{
					"op":          "SIGNAL_REF",
					"signal_type": chosenSignalType,
					"signal_name": chosenSignalName,
				},
				"models": []map[string]interface{}{
					{"model": route.Model},
				},
				"plugins": []map[string]interface{}{},
			},
		},
	}, fmt.Sprintf("Repair route %s by switching it to signal %s.", route.Name, chosenSignalName), nil), true
}

func (p PreviewPlanner) planConstraintRepair(diagnostics []DiagnosticSnapshot) (PlannerResult, bool) {
	for _, diagnostic := range diagnostics {
		lowerMessage := strings.ToLower(diagnostic.Message)
		if diagnostic.Level == "constraint" && strings.Contains(lowerMessage, "threshold") {
			return plannerClarification(
				"What threshold should replace the invalid value?",
				[]ClarificationOption{
					{ID: "0.7", Label: "0.7", Description: "Balanced threshold for most prompt-guard and embedding cases."},
					{ID: "0.8", Label: "0.8", Description: "Use a stricter threshold while staying within canonical limits."},
					{ID: "0.9", Label: "0.9", Description: "Use a very strict threshold near the upper bound."},
				},
			), true
		}
	}
	return PlannerResult{}, false
}

func (p PreviewPlanner) planUndefinedSignalRepair(context SessionContext) (PlannerResult, bool) {
	for _, diagnostic := range context.Diagnostics {
		lowerMessage := strings.ToLower(diagnostic.Message)
		if strings.Contains(lowerMessage, "signal") && strings.Contains(lowerMessage, "not defined") {
			return plannerClarification(
				"Which signal should this route reference instead?",
				buildSignalOptions(context.Symbols),
			), true
		}
	}
	return PlannerResult{}, false
}

func (p PreviewPlanner) planUnknownTypeRepair(diagnostics []DiagnosticSnapshot) (PlannerResult, bool) {
	for _, diagnostic := range diagnostics {
		lowerMessage := strings.ToLower(diagnostic.Message)
		switch {
		case strings.Contains(lowerMessage, "unknown signal type"):
			return plannerClarification(
				"What kind of signal should replace the unknown signal type?",
				buildTypeOptions("signal", p.manifest.Signals),
			), true
		case strings.Contains(lowerMessage, "unknown plugin type"):
			return plannerClarification(
				"What kind of plugin should replace the unknown plugin type?",
				buildTypeOptions("plugin", p.manifest.Plugins),
			), true
		case strings.Contains(lowerMessage, "unknown algorithm type"):
			return plannerClarification(
				"What kind of algorithm should replace the unknown algorithm type?",
				buildTypeOptions("algorithm", p.manifest.Algorithms),
			), true
		case strings.Contains(lowerMessage, "unknown backend type"):
			return plannerClarification(
				"What kind of backend should replace the unknown backend type?",
				buildTypeOptions("backend", p.manifest.Backends),
			), true
		}
	}
	return PlannerResult{}, false
}

func (p PreviewPlanner) detectConstructs(lowerPrompt string) []ConstructKind {
	seen := map[ConstructKind]bool{}
	add := func(kind ConstructKind) {
		if !seen[kind] {
			seen[kind] = true
		}
	}

	routeFocused := containsAny(lowerPrompt, "route")
	routeConditionFocused := routeFocused && containsAny(lowerPrompt, "signal", "when", "if", "condition")
	routePluginFocused := routeFocused && containsAny(lowerPrompt, "plugin", "plugins", "attach", "protect", "guard", "cache", "memory", "response")

	if routeFocused {
		add(ConstructRoute)
	}
	if containsAny(lowerPrompt, "signal") && !routeConditionFocused {
		add(ConstructSignal)
	}
	if containsAny(lowerPrompt, "plugin") && !routePluginFocused {
		add(ConstructPlugin)
	}
	if containsAny(lowerPrompt, "backend", "endpoint", "provider profile") {
		add(ConstructBackend)
	}

	for _, entry := range p.manifest.Signals {
		if strings.Contains(lowerPrompt, entry.TypeName) && !routeConditionFocused {
			add(ConstructSignal)
		}
	}
	for _, entry := range p.manifest.Plugins {
		if strings.Contains(lowerPrompt, entry.TypeName) && !routePluginFocused {
			add(ConstructPlugin)
		}
	}
	for _, entry := range p.manifest.Backends {
		if strings.Contains(lowerPrompt, entry.TypeName) {
			add(ConstructBackend)
		}
	}

	constructs := make([]ConstructKind, 0, len(seen))
	for _, kind := range []ConstructKind{ConstructRoute, ConstructSignal, ConstructPlugin, ConstructBackend} {
		if seen[kind] {
			constructs = append(constructs, kind)
		}
	}
	return constructs
}

func (p PreviewPlanner) planSignal(prompt, lowerPrompt string, symbols *SymbolSnapshot, action string) PlannerResult {
	if action == "delete" {
		name, signalType, ok := matchExistingSignal(prompt, lowerPrompt, symbols)
		if !ok {
			return plannerClarification(
				"Which existing signal should I delete?",
				buildSignalOptions(symbols),
			)
		}
		return plannerReady(IntentIR{
			Version:   "1.0",
			Operation: OperationModify,
			Intents: []map[string]interface{}{
				{
					"type":               "modify",
					"action":             "delete",
					"target_construct":   "signal",
					"target_name":        name,
					"target_signal_type": signalType,
				},
			},
		}, fmt.Sprintf("Delete the %s signal %s.", signalType, name), nil)
	}

	signalType, ok := matchManifestType(lowerPrompt, p.manifest.Signals)
	if !ok {
		return plannerClarification(
			"What kind of signal do you want to create?",
			p.buildSupportedTypeOptions(p.Support().SignalTypes, p.manifest.Signals),
		)
	}

	switch signalType {
	case "keyword":
		keywords := extractStringList(prompt)
		if len(keywords) == 0 {
			return plannerClarification(
				"Which keywords should the keyword signal match?",
				[]ClarificationOption{
					{ID: "quoted_keywords", Label: "Use quotes", Description: `Write the keywords in quotes, for example "urgent" and "asap".`},
					{ID: "comma_keywords", Label: "Use commas", Description: "List the keywords after the word keywords, separated by commas."},
				},
			)
		}
		name := firstNonEmpty(extractExplicitName(prompt), "keyword_"+sanitizeIdentifier(keywords[0]))
		operator := "any"
		warnings := []PlannerWarning{}
		if containsAny(lowerPrompt, "all keywords", "operator all") {
			operator = "all"
		} else {
			warnings = append(warnings, PlannerWarning{Code: "default_keyword_operator", Message: `Using operator "any" because the prompt did not request "all".`})
		}
		return plannerReady(IntentIR{
			Version:   "1.0",
			Operation: OperationGenerate,
			Intents: []map[string]interface{}{
				{
					"type":        "signal",
					"signal_type": "keyword",
					"name":        name,
					"fields": map[string]interface{}{
						"operator": operator,
						"keywords": keywords,
					},
				},
			},
		}, fmt.Sprintf("Create a keyword signal %s for %d keyword(s).", name, len(keywords)), warnings)
	case "embedding":
		candidates := extractStringList(prompt)
		if len(candidates) == 0 {
			return plannerClarification(
				"Which candidate phrases should the embedding signal compare against?",
				[]ClarificationOption{
					{ID: "quoted_candidates", Label: "Use quotes", Description: `Write candidate phrases in quotes, for example "billing" and "refund".`},
					{ID: "retry_with_examples", Label: "Add examples", Description: "Include example phrases the signal should detect."},
				},
			)
		}
		threshold, ok := extractNumberAfterWord(lowerPrompt, "threshold")
		if !ok {
			return plannerClarification(
				"What threshold should the embedding signal use?",
				[]ClarificationOption{
					{ID: "0.7", Label: "0.7", Description: "Balanced precision and recall."},
					{ID: "0.8", Label: "0.8", Description: "Stricter similarity matching."},
					{ID: "0.9", Label: "0.9", Description: "Very strict similarity matching."},
				},
			)
		}
		name := firstNonEmpty(extractExplicitName(prompt), "embedding_"+sanitizeIdentifier(candidates[0]))
		return plannerReady(IntentIR{
			Version:   "1.0",
			Operation: OperationGenerate,
			Intents: []map[string]interface{}{
				{
					"type":        "signal",
					"signal_type": "embedding",
					"name":        name,
					"fields": map[string]interface{}{
						"threshold":  threshold,
						"candidates": candidates,
					},
				},
			},
		}, fmt.Sprintf("Create an embedding signal %s.", name), nil)
	default:
		return PlannerResult{
			Status:      PlannerStatusUnsupported,
			Explanation: fmt.Sprintf("Signal type %s needs richer field extraction than the preview planner supports.", signalType),
			Warnings: []PlannerWarning{
				{Code: "signal_type_unimplemented", Message: "Preview planner currently supports keyword and embedding signal creation first."},
			},
			Error: "unsupported signal type for preview planner",
		}
	}
}

func (p PreviewPlanner) planPlugin(prompt, lowerPrompt string, symbols *SymbolSnapshot, action string) PlannerResult {
	if action == "delete" {
		pluginNames := []string(nil)
		if symbols != nil {
			pluginNames = symbols.Plugins
		}
		name, ok := matchExistingName(lowerPrompt, pluginNames)
		if !ok {
			return plannerClarification(
				"Which existing plugin should I delete?",
				buildStringOptions("plugin", pluginNames),
			)
		}
		return plannerReady(IntentIR{
			Version:   "1.0",
			Operation: OperationModify,
			Intents: []map[string]interface{}{
				{
					"type":             "modify",
					"action":           "delete",
					"target_construct": "plugin",
					"target_name":      name,
				},
			},
		}, fmt.Sprintf("Delete the plugin template %s.", name), nil)
	}

	pluginType, ok := matchManifestType(lowerPrompt, p.manifest.Plugins)
	if !ok {
		return plannerClarification(
			"What kind of plugin template do you want to create?",
			p.buildSupportedTypeOptions(p.Support().PluginTypes, p.manifest.Plugins),
		)
	}

	name := firstNonEmpty(extractExplicitName(prompt), pluginType+"_default")
	fields := map[string]interface{}{}
	warnings := []PlannerWarning{}
	switch pluginType {
	case "semantic_cache", "memory":
		fields["enabled"] = true
		warnings = append(warnings, PlannerWarning{Code: "default_enabled", Message: "Enabled the plugin template by default."})
	case "fast_response":
		message := firstQuotedString(prompt)
		if message == "" {
			return plannerClarification(
				"What response message should the fast response plugin return?",
				[]ClarificationOption{
					{ID: "quoted_message", Label: "Add quoted text", Description: `Include the static response in quotes, for example "I cannot help with that request."`},
				},
			)
		}
		fields["message"] = message
	case "system_prompt":
		systemPrompt := firstQuotedString(prompt)
		if systemPrompt == "" {
			return plannerClarification(
				"What system prompt should this plugin inject?",
				[]ClarificationOption{
					{ID: "quoted_prompt", Label: "Add quoted prompt", Description: `Include the system prompt in quotes.`},
				},
			)
		}
		fields["system_prompt"] = systemPrompt
	default:
		return PlannerResult{
			Status:      PlannerStatusUnsupported,
			Explanation: fmt.Sprintf("Plugin type %s needs richer field extraction than the preview planner supports.", pluginType),
			Warnings: []PlannerWarning{
				{Code: "plugin_type_unimplemented", Message: "Preview planner currently supports semantic_cache, memory, fast_response, and system_prompt creation."},
			},
			Error: "unsupported plugin type for preview planner",
		}
	}

	return plannerReady(IntentIR{
		Version:   "1.0",
		Operation: OperationGenerate,
		Intents: []map[string]interface{}{
			{
				"type":        "plugin_template",
				"name":        name,
				"plugin_type": pluginType,
				"fields":      fields,
			},
		},
	}, fmt.Sprintf("Create a %s plugin template named %s.", pluginType, name), warnings)
}

func (p PreviewPlanner) buildSupportedTypeOptions(typeNames []string, entries []TypeSchemaEntry) []ClarificationOption {
	if len(typeNames) == 0 {
		return nil
	}

	entryByType := make(map[string]TypeSchemaEntry, len(entries))
	for _, entry := range entries {
		entryByType[entry.TypeName] = entry
	}

	options := make([]ClarificationOption, 0, len(typeNames))
	for _, typeName := range typeNames {
		entry, ok := entryByType[typeName]
		if !ok {
			continue
		}
		options = append(options, ClarificationOption{
			ID:          typeName,
			Label:       humanizeTypeName(typeName),
			Description: entry.Description,
		})
	}
	return options
}

func humanizeTypeName(typeName string) string {
	parts := strings.Split(typeName, "_")
	for idx, part := range parts {
		if part == "" {
			continue
		}
		parts[idx] = strings.ToUpper(part[:1]) + part[1:]
	}
	return strings.Join(parts, " ")
}

func (p PreviewPlanner) planBackend(prompt, lowerPrompt string, symbols *SymbolSnapshot, action string) PlannerResult {
	if action == "delete" {
		name, backendType, ok := matchExistingBackend(prompt, lowerPrompt, symbols)
		if !ok {
			return plannerClarification(
				"Which existing backend should I delete?",
				buildBackendOptions(symbols),
			)
		}
		return plannerReady(IntentIR{
			Version:   "1.0",
			Operation: OperationModify,
			Intents: []map[string]interface{}{
				{
					"type":                "modify",
					"action":              "delete",
					"target_construct":    "backend",
					"target_name":         name,
					"target_backend_type": backendType,
				},
			},
		}, fmt.Sprintf("Delete the %s backend %s.", backendType, name), nil)
	}

	backendType := "vllm_endpoint"
	if matchedType, ok := matchManifestType(lowerPrompt, p.manifest.Backends); ok {
		backendType = matchedType
	}
	if backendType != "vllm_endpoint" {
		return PlannerResult{
			Status:      PlannerStatusUnsupported,
			Explanation: fmt.Sprintf("Backend type %s needs richer field extraction than the preview planner supports.", backendType),
			Warnings: []PlannerWarning{
				{Code: "backend_type_unimplemented", Message: "Preview planner currently supports vllm_endpoint backend creation first."},
			},
			Error: "unsupported backend type for preview planner",
		}
	}

	host, port, ok := extractHostPort(prompt)
	if !ok {
		return plannerClarification(
			"What address and port should the vLLM endpoint use?",
			[]ClarificationOption{
				{ID: "host_port", Label: "Add host:port", Description: `Include a host and port such as 127.0.0.1:8000 or localhost:8001.`},
			},
		)
	}

	model, _ := extractModel(prompt, lowerPrompt, symbols)
	name := firstNonEmpty(extractExplicitName(prompt), model, "endpoint_"+sanitizeIdentifier(host))
	fields := map[string]interface{}{
		"address": host,
		"port":    port,
	}
	if model != "" {
		fields["model"] = model
	}

	return plannerReady(IntentIR{
		Version:   "1.0",
		Operation: OperationGenerate,
		Intents: []map[string]interface{}{
			{
				"type":         "backend",
				"backend_type": "vllm_endpoint",
				"name":         name,
				"fields":       fields,
			},
		},
	}, fmt.Sprintf("Create a vLLM endpoint backend named %s.", name), nil)
}

func (p PreviewPlanner) planRoute(prompt, lowerPrompt string, symbols *SymbolSnapshot, action string) PlannerResult {
	if action == "delete" {
		routeNames := []string(nil)
		if symbols != nil {
			routeNames = symbols.Routes
		}
		name, ok := extractRouteName(prompt, lowerPrompt, symbols)
		if !ok {
			return plannerClarification(
				"Which existing route should I delete?",
				buildStringOptions("route", routeNames),
			)
		}
		return plannerReady(IntentIR{
			Version:   "1.0",
			Operation: OperationModify,
			Intents: []map[string]interface{}{
				{
					"type":             "modify",
					"action":           "delete",
					"target_construct": "route",
					"target_name":      name,
				},
			},
		}, fmt.Sprintf("Delete the route %s.", name), nil)
	}

	if action == "update" {
		return p.planRouteUpdate(prompt, lowerPrompt, symbols)
	}

	routeName, ok := extractRouteName(prompt, lowerPrompt, nil)
	if !ok {
		return plannerClarification(
			"What should the new route be called?",
			[]ClarificationOption{
				{ID: "named_route", Label: "Add a name", Description: `Say "create route support_route" or "create a route named support_route".`},
			},
		)
	}

	model, modelInferred := extractModel(prompt, lowerPrompt, symbols)
	if model == "" {
		return plannerClarification(
			"Which model should this route use?",
			buildModelOptions(symbols),
		)
	}

	priority, ok := extractPriority(lowerPrompt)
	warnings := []PlannerWarning{}
	if !ok {
		priority = 100
		warnings = append(warnings, PlannerWarning{Code: "default_priority", Message: "Using default route priority 100."})
	}
	if modelInferred {
		warnings = append(warnings, PlannerWarning{Code: "inferred_model", Message: fmt.Sprintf("Using the only known model %s from Builder symbols.", model)})
	}

	var condition map[string]interface{}
	if shouldResolveRouteCondition(lowerPrompt, symbols) {
		var conditionWarnings []PlannerWarning
		var conditionResult *PlannerResult
		condition, conditionWarnings, conditionResult = p.resolveRouteCondition(prompt, lowerPrompt, symbols)
		if conditionResult != nil {
			return *conditionResult
		}
		if condition != nil {
			warnings = append(warnings, conditionWarnings...)
		}
	}

	pluginRefs := []map[string]interface{}{}
	if shouldResolveRoutePlugin(lowerPrompt, symbols) {
		resolvedPluginRefs, pluginWarnings, pluginResult := p.resolveRoutePluginRefs(lowerPrompt, symbols)
		if pluginResult != nil {
			return *pluginResult
		}
		pluginRefs = resolvedPluginRefs
		warnings = append(warnings, pluginWarnings...)
	}

	intent := map[string]interface{}{
		"type":     "route",
		"name":     routeName,
		"priority": priority,
		"models": []map[string]interface{}{
			{"model": model},
		},
		"plugins": pluginRefs,
	}
	if condition != nil {
		intent["condition"] = condition
	}

	return plannerReady(IntentIR{
		Version:   "1.0",
		Operation: OperationGenerate,
		Intents:   []map[string]interface{}{intent},
	}, fmt.Sprintf("Create a route named %s for model %s.", routeName, model), warnings)
}

func (p PreviewPlanner) planRouteUpdate(prompt, lowerPrompt string, symbols *SymbolSnapshot) PlannerResult {
	routeName, ok := extractRouteName(prompt, lowerPrompt, symbols)
	if !ok {
		return plannerClarification(
			"Which existing route should I update?",
			buildStringOptions("route", routeNames(symbols)),
		)
	}

	changes := map[string]interface{}{}
	warnings := []PlannerWarning{}
	summary := []string{}

	if priority, ok := extractPriority(lowerPrompt); ok {
		changes["priority"] = priority
		summary = append(summary, fmt.Sprintf("set priority to %d", priority))
	}

	if shouldResolveRouteModel(lowerPrompt, symbols) {
		model, inferred := extractModel(prompt, lowerPrompt, symbols)
		if model == "" {
			return plannerClarification(
				"Which model should this route use?",
				buildModelOptions(symbols),
			)
		}
		changes["models"] = []map[string]interface{}{{"model": model}}
		summary = append(summary, fmt.Sprintf("use model %s", model))
		if inferred {
			warnings = append(warnings, PlannerWarning{
				Code:    "inferred_model",
				Message: fmt.Sprintf("Using the only known model %s from Builder symbols.", model),
			})
		}
	}

	if hasClearRouteConditionPrompt(lowerPrompt) {
		changes["condition"] = nil
		summary = append(summary, "clear the route condition")
	} else if shouldResolveRouteCondition(lowerPrompt, symbols) {
		condition, conditionWarnings, conditionResult := p.resolveRouteCondition(prompt, lowerPrompt, symbols)
		if conditionResult != nil {
			return *conditionResult
		}
		if condition != nil {
			changes["condition"] = condition
			summary = append(summary, fmt.Sprintf(`reference signal %s("%s")`, condition["signal_type"], condition["signal_name"]))
			warnings = append(warnings, conditionWarnings...)
		}
	}

	if shouldResolveRoutePlugin(lowerPrompt, symbols) {
		pluginRefs, pluginWarnings, pluginResult := p.resolveRoutePluginRefs(lowerPrompt, symbols)
		if pluginResult != nil {
			return *pluginResult
		}
		if len(pluginRefs) > 0 {
			changes["plugins"] = pluginRefs
			pluginNames := make([]string, 0, len(pluginRefs))
			for _, pluginRef := range pluginRefs {
				if name, ok := pluginRef["name"].(string); ok && name != "" {
					pluginNames = append(pluginNames, name)
				}
			}
			if len(pluginNames) > 0 {
				summary = append(summary, fmt.Sprintf("attach plugin %s", strings.Join(pluginNames, ", ")))
			}
			warnings = append(warnings, pluginWarnings...)
		}
	}

	if len(changes) == 0 {
		return plannerClarification(
			fmt.Sprintf("What should I change on route %s?", routeName),
			[]ClarificationOption{
				{ID: "change_model", Label: "Change model", Description: "Switch the route to a different model."},
				{ID: "set_condition", Label: "Set condition", Description: "Add or replace the route signal condition."},
				{ID: "attach_plugin", Label: "Attach plugin", Description: "Add an existing plugin reference to the route."},
				{ID: "set_priority", Label: "Set priority", Description: "Change the route priority value."},
			},
		)
	}

	explanation := fmt.Sprintf("Update route %s.", routeName)
	if len(summary) > 0 {
		explanation = fmt.Sprintf("Update route %s to %s.", routeName, strings.Join(summary, " and "))
	}

	return plannerReady(IntentIR{
		Version:   "1.0",
		Operation: OperationModify,
		Intents: []map[string]interface{}{
			{
				"type":             "modify",
				"action":           "update",
				"target_construct": "route",
				"target_name":      routeName,
				"changes":          changes,
			},
		},
	}, explanation, warnings)
}

func routeNames(symbols *SymbolSnapshot) []string {
	if symbols == nil {
		return nil
	}
	return symbols.Routes
}

func shouldResolveRouteModel(lowerPrompt string, symbols *SymbolSnapshot) bool {
	if strings.Contains(lowerPrompt, "model") {
		return true
	}
	return containsExistingCandidate(lowerPrompt, symbolsGetModels(symbols))
}

func shouldResolveRouteCondition(lowerPrompt string, symbols *SymbolSnapshot) bool {
	if containsAny(lowerPrompt, "signal", "when", "if", "condition") {
		return true
	}
	if symbols == nil {
		return false
	}
	for _, signal := range symbols.Signals {
		if strings.Contains(lowerPrompt, strings.ToLower(signal.Name)) {
			return true
		}
	}
	return false
}

func hasClearRouteConditionPrompt(lowerPrompt string) bool {
	return containsAny(lowerPrompt,
		"without condition",
		"clear condition",
		"remove condition",
		"no condition",
		"default route",
	)
}

func shouldResolveRoutePlugin(lowerPrompt string, symbols *SymbolSnapshot) bool {
	if containsAny(lowerPrompt, "plugin", "plugins", "protect", "guard", "cache", "memory", "response") {
		return true
	}
	if symbols == nil {
		return false
	}
	return containsExistingCandidate(lowerPrompt, symbols.Plugins)
}

func (p PreviewPlanner) resolveRouteCondition(prompt, lowerPrompt string, symbols *SymbolSnapshot) (map[string]interface{}, []PlannerWarning, *PlannerResult) {
	if symbols == nil || len(symbols.Signals) == 0 {
		result := plannerClarification(
			"Which existing signal should this route reference?",
			buildSignalOptions(symbols),
		)
		return nil, nil, &result
	}

	signalName, signalType, ok := matchExistingSignal(prompt, lowerPrompt, symbols)
	if !ok {
		result := plannerClarification(
			"Which existing signal should this route reference?",
			buildSignalOptions(symbols),
		)
		return nil, nil, &result
	}

	warnings := []PlannerWarning{}
	if !containsExplicitSignalMention(lowerPrompt, signalName) && len(symbols.Signals) == 1 {
		warnings = append(warnings, PlannerWarning{
			Code:    "inferred_signal",
			Message: fmt.Sprintf("Using the only known signal %s from Builder symbols.", signalName),
		})
	}

	return map[string]interface{}{
		"op":          "SIGNAL_REF",
		"signal_type": signalType,
		"signal_name": signalName,
	}, warnings, nil
}

func (p PreviewPlanner) resolveRoutePluginRefs(lowerPrompt string, symbols *SymbolSnapshot) ([]map[string]interface{}, []PlannerWarning, *PlannerResult) {
	if symbols == nil || len(symbols.Plugins) == 0 {
		result := plannerClarification(
			"Which existing plugin should this route use?",
			buildStringOptions("plugin", nil),
		)
		return nil, nil, &result
	}

	pluginName, explicit := matchExplicitValue(lowerPrompt, symbols.Plugins)
	if !explicit {
		if len(symbols.Plugins) != 1 {
			result := plannerClarification(
				"Which existing plugin should this route use?",
				buildStringOptions("plugin", symbols.Plugins),
			)
			return nil, nil, &result
		}
		pluginName = symbols.Plugins[0]
	}

	warnings := []PlannerWarning{}
	if !explicit {
		warnings = append(warnings, PlannerWarning{
			Code:    "inferred_plugin",
			Message: fmt.Sprintf("Using the only known plugin %s from Builder symbols.", pluginName),
		})
	}

	return []map[string]interface{}{{"name": pluginName}}, warnings, nil
}

func symbolsGetModels(symbols *SymbolSnapshot) []string {
	if symbols == nil {
		return nil
	}
	return symbols.Models
}

func containsExistingCandidate(lowerPrompt string, candidates []string) bool {
	for _, candidate := range candidates {
		if strings.Contains(lowerPrompt, strings.ToLower(candidate)) {
			return true
		}
	}
	return false
}

func containsExplicitSignalMention(lowerPrompt, signalName string) bool {
	return strings.Contains(lowerPrompt, strings.ToLower(signalName))
}

func plannerReady(intentIR IntentIR, explanation string, warnings []PlannerWarning) PlannerResult {
	return PlannerResult{
		Status:      PlannerStatusReady,
		IntentIR:    &intentIR,
		Explanation: explanation,
		Warnings:    warnings,
	}
}

func plannerClarification(question string, options []ClarificationOption) PlannerResult {
	return PlannerResult{
		Status: PlannerStatusNeedsClarification,
		Clarification: &Clarification{
			Question: question,
			Options:  options,
		},
	}
}

func detectAction(lowerPrompt string, modeHint OperationMode, hasBaseDSL bool) string {
	switch {
	case containsAny(lowerPrompt, "delete", "remove"):
		return "delete"
	case containsAny(lowerPrompt, "update", "modify", "change", "rename"):
		return "update"
	case containsAny(lowerPrompt, "add", "create", "new", "declare", "define"):
		return "add"
	case modeHint == OperationModify || hasBaseDSL:
		return "update"
	default:
		return "add"
	}
}

func containsAny(lowerPrompt string, values ...string) bool {
	for _, value := range values {
		if strings.Contains(lowerPrompt, value) {
			return true
		}
	}
	return false
}

func matchManifestType(lowerPrompt string, entries []TypeSchemaEntry) (string, bool) {
	for _, entry := range entries {
		if strings.Contains(lowerPrompt, entry.TypeName) {
			return entry.TypeName, true
		}
	}
	return "", false
}

func extractExplicitName(prompt string) string {
	match := namePattern.FindStringSubmatch(prompt)
	if len(match) == 2 {
		return sanitizeIdentifier(match[1])
	}
	return ""
}

func sanitizeIdentifier(value string) string {
	value = strings.ToLower(strings.TrimSpace(value))
	var b strings.Builder
	lastUnderscore := false
	for _, r := range value {
		switch {
		case r >= 'a' && r <= 'z', r >= '0' && r <= '9', r == '-':
			b.WriteRune(r)
			lastUnderscore = false
		default:
			if !lastUnderscore {
				b.WriteRune('_')
				lastUnderscore = true
			}
		}
	}
	result := strings.Trim(b.String(), "_")
	if result == "" {
		return "draft"
	}
	return result
}

func extractStringList(prompt string) []string {
	values := make([]string, 0)
	for _, match := range quotedStringPattern.FindAllStringSubmatch(prompt, -1) {
		value := firstNonEmpty(match[1], match[2])
		if value != "" && !slices.Contains(values, value) {
			values = append(values, value)
		}
	}
	if len(values) > 0 {
		return values
	}

	lowerPrompt := strings.ToLower(prompt)
	keywordsIndex := strings.Index(lowerPrompt, "keywords")
	if keywordsIndex == -1 {
		return nil
	}
	rest := prompt[keywordsIndex+len("keywords"):]
	rest = strings.TrimLeft(rest, ": ")
	parts := strings.Split(rest, ",")
	for _, part := range parts {
		value := strings.TrimSpace(part)
		if value == "" {
			continue
		}
		value = strings.Trim(value, ".")
		if value != "" && !slices.Contains(values, value) {
			values = append(values, value)
		}
	}
	return values
}

func firstQuotedString(prompt string) string {
	values := extractStringList(prompt)
	if len(values) == 0 {
		return ""
	}
	return values[0]
}

func extractNumberAfterWord(lowerPrompt, word string) (float64, bool) {
	index := strings.Index(lowerPrompt, word)
	if index == -1 {
		return 0, false
	}
	match := numberPattern.FindString(lowerPrompt[index+len(word):])
	if match == "" {
		return 0, false
	}
	value, err := strconv.ParseFloat(match, 64)
	if err != nil {
		return 0, false
	}
	return value, true
}

func extractPromptGuardEnabled(baseDSL string) (bool, bool) {
	match := globalPromptGuard.FindStringSubmatch(baseDSL)
	if len(match) != 2 {
		return false, false
	}
	enabledMatch := boolFieldPattern.FindStringSubmatch(match[1])
	if len(enabledMatch) != 2 {
		return false, false
	}
	return enabledMatch[1] == "true", true
}

type simpleRoute struct {
	Name        string
	Description string
	Priority    int
	SignalType  string
	SignalName  string
	Model       string
}

func extractSingleSimpleRoute(baseDSL string) (simpleRoute, bool) {
	matches := routeHeaderPattern.FindAllStringSubmatchIndex(baseDSL, -1)
	if len(matches) != 1 {
		return simpleRoute{}, false
	}

	match := routeHeaderPattern.FindStringSubmatch(baseDSL[matches[0][0]:matches[0][1]])
	if len(match) < 2 {
		return simpleRoute{}, false
	}

	block, ok := extractBlockFromIndex(baseDSL, matches[0][0])
	if !ok || strings.Contains(block, "ALGORITHM") || strings.Contains(block, "PLUGIN") {
		return simpleRoute{}, false
	}

	priority, ok := extractPriority(strings.ToLower(block))
	if !ok {
		return simpleRoute{}, false
	}

	whenMatch := routeWhenPattern.FindStringSubmatch(block)
	if len(whenMatch) != 3 {
		return simpleRoute{}, false
	}

	modelMatches := routeModelLine.FindAllStringSubmatch(block, -1)
	if len(modelMatches) != 1 || strings.TrimSpace(modelMatches[0][2]) != "" {
		return simpleRoute{}, false
	}

	return simpleRoute{
		Name:        match[1],
		Description: firstNonEmpty(match[2]),
		Priority:    priority,
		SignalType:  whenMatch[1],
		SignalName:  whenMatch[2],
		Model:       modelMatches[0][1],
	}, true
}

func extractBlockFromIndex(src string, start int) (string, bool) {
	braceCount := 0
	foundOpen := false
	for index := start; index < len(src); index++ {
		switch src[index] {
		case '{':
			braceCount++
			foundOpen = true
		case '}':
			braceCount--
			if foundOpen && braceCount == 0 {
				return src[start : index+1], true
			}
		}
	}
	return "", false
}

func hasThresholdDiagnostic(diagnostics []DiagnosticSnapshot) bool {
	for _, diagnostic := range diagnostics {
		lowerMessage := strings.ToLower(diagnostic.Message)
		if diagnostic.Level == "constraint" && strings.Contains(lowerMessage, "threshold") {
			return true
		}
	}
	return false
}

func hasUndefinedSignalDiagnostic(diagnostics []DiagnosticSnapshot) bool {
	for _, diagnostic := range diagnostics {
		lowerMessage := strings.ToLower(diagnostic.Message)
		if strings.Contains(lowerMessage, "signal") && strings.Contains(lowerMessage, "not defined") {
			return true
		}
	}
	return false
}

func extractPriority(lowerPrompt string) (int, bool) {
	match := priorityPattern.FindStringSubmatch(lowerPrompt)
	if len(match) != 2 {
		return 0, false
	}
	value, err := strconv.Atoi(match[1])
	if err != nil {
		return 0, false
	}
	return value, true
}

func extractHostPort(prompt string) (string, int, bool) {
	match := hostPortPattern.FindStringSubmatch(prompt)
	if len(match) != 3 {
		return "", 0, false
	}
	port, err := strconv.Atoi(match[2])
	if err != nil {
		return "", 0, false
	}
	return match[1], port, true
}

func extractModel(prompt, lowerPrompt string, symbols *SymbolSnapshot) (string, bool) {
	match := modelPattern.FindStringSubmatch(prompt)
	if len(match) == 2 {
		return strings.TrimSpace(match[1]), false
	}
	if symbols == nil || len(symbols.Models) == 0 {
		return "", false
	}

	matchedModel, ok := matchExistingValue(lowerPrompt, symbols.Models)
	if ok {
		return matchedModel, false
	}
	if len(symbols.Models) == 1 {
		return symbols.Models[0], true
	}
	return "", false
}

func extractRouteName(prompt, lowerPrompt string, symbols *SymbolSnapshot) (string, bool) {
	if symbols != nil {
		if name, ok := matchExistingName(lowerPrompt, symbols.Routes); ok {
			return name, true
		}
	}

	tokens := strings.Fields(prompt)
	for i := 0; i < len(tokens)-1; i++ {
		if strings.EqualFold(tokens[i], "route") {
			candidate := sanitizeIdentifier(strings.Trim(tokens[i+1], `"'.,`))
			if candidate != "" && candidate != "for" {
				return candidate, true
			}
		}
	}

	if explicit := extractExplicitName(prompt); explicit != "" {
		return explicit, true
	}
	return "", false
}

func matchExistingSignal(prompt, lowerPrompt string, symbols *SymbolSnapshot) (string, string, bool) {
	if symbols == nil {
		return "", "", false
	}
	bestScore := 0
	bestName := ""
	bestType := ""
	for _, signal := range symbols.Signals {
		if strings.Contains(lowerPrompt, strings.ToLower(signal.Name)) {
			score := len(signal.Name)
			if score > bestScore {
				bestScore = score
				bestName = sanitizeIdentifier(signal.Name)
				bestType = signal.Type
			}
		}
	}
	if bestName != "" {
		return bestName, bestType, true
	}

	if explicit := extractExplicitName(prompt); explicit != "" {
		for _, signal := range symbols.Signals {
			if strings.EqualFold(signal.Name, explicit) {
				return sanitizeIdentifier(signal.Name), signal.Type, true
			}
		}
	}
	if len(symbols.Signals) == 1 {
		return sanitizeIdentifier(symbols.Signals[0].Name), symbols.Signals[0].Type, true
	}
	return "", "", false
}

func matchExistingBackend(prompt, lowerPrompt string, symbols *SymbolSnapshot) (string, string, bool) {
	if symbols == nil {
		return "", "", false
	}
	bestScore := 0
	bestName := ""
	bestType := ""
	for _, backend := range symbols.Backends {
		if strings.Contains(lowerPrompt, strings.ToLower(backend.Name)) {
			score := len(backend.Name)
			if score > bestScore {
				bestScore = score
				bestName = sanitizeIdentifier(backend.Name)
				bestType = backend.Type
			}
		}
	}
	if bestName != "" {
		return bestName, bestType, true
	}

	if explicit := extractExplicitName(prompt); explicit != "" {
		for _, backend := range symbols.Backends {
			if strings.EqualFold(backend.Name, explicit) {
				return sanitizeIdentifier(backend.Name), backend.Type, true
			}
		}
	}
	if len(symbols.Backends) == 1 {
		return sanitizeIdentifier(symbols.Backends[0].Name), symbols.Backends[0].Type, true
	}
	return "", "", false
}

func matchExistingName(lowerPrompt string, candidates []string) (string, bool) {
	value, ok := matchExistingValue(lowerPrompt, candidates)
	if !ok {
		return "", false
	}
	return sanitizeIdentifier(value), true
}

func matchExistingValue(lowerPrompt string, candidates []string) (string, bool) {
	best := ""
	bestScore := 0
	for _, candidate := range candidates {
		if strings.Contains(lowerPrompt, strings.ToLower(candidate)) {
			score := len(candidate)
			if score > bestScore {
				best = candidate
				bestScore = score
			}
		}
	}
	if best != "" {
		return best, true
	}
	if len(candidates) == 1 {
		return candidates[0], true
	}
	return "", false
}

func matchExplicitValue(lowerPrompt string, candidates []string) (string, bool) {
	best := ""
	bestScore := 0
	for _, candidate := range candidates {
		if strings.Contains(lowerPrompt, strings.ToLower(candidate)) {
			score := len(candidate)
			if score > bestScore {
				best = candidate
				bestScore = score
			}
		}
	}
	if best == "" {
		return "", false
	}
	return best, true
}

func buildSignalOptions(symbols *SymbolSnapshot) []ClarificationOption {
	options := []ClarificationOption{}
	if symbols != nil {
		for _, signal := range symbols.Signals {
			options = append(options, ClarificationOption{
				ID:          sanitizeIdentifier(signal.Name),
				Label:       signal.Name,
				Description: fmt.Sprintf("%s signal", signal.Type),
			})
		}
	}
	if len(options) == 0 {
		return []ClarificationOption{
			{ID: "name_signal", Label: "Name a signal", Description: "Specify the existing signal name to delete."},
		}
	}
	return options
}

func buildBackendOptions(symbols *SymbolSnapshot) []ClarificationOption {
	options := []ClarificationOption{}
	if symbols != nil {
		for _, backend := range symbols.Backends {
			options = append(options, ClarificationOption{
				ID:          sanitizeIdentifier(backend.Name),
				Label:       backend.Name,
				Description: fmt.Sprintf("%s backend", backend.Type),
			})
		}
	}
	if len(options) == 0 {
		return []ClarificationOption{
			{ID: "name_backend", Label: "Name a backend", Description: "Specify the existing backend name to delete."},
		}
	}
	return options
}

func buildStringOptions(kind string, values []string) []ClarificationOption {
	if len(values) == 0 {
		return []ClarificationOption{
			{ID: "name_" + kind, Label: "Add a name", Description: fmt.Sprintf("Specify the existing %s name to delete.", kind)},
		}
	}
	options := make([]ClarificationOption, 0, len(values))
	for _, value := range values {
		options = append(options, ClarificationOption{
			ID:          sanitizeIdentifier(value),
			Label:       value,
			Description: fmt.Sprintf("Existing %s", kind),
		})
	}
	return options
}

func buildModelOptions(symbols *SymbolSnapshot) []ClarificationOption {
	if symbols == nil || len(symbols.Models) == 0 {
		return []ClarificationOption{
			{ID: "add_model", Label: "Specify model", Description: `Say "model <name>" to choose the route target model.`},
		}
	}
	options := make([]ClarificationOption, 0, len(symbols.Models))
	for _, model := range symbols.Models {
		options = append(options, ClarificationOption{
			ID:          sanitizeIdentifier(model),
			Label:       model,
			Description: "Known Builder model symbol",
		})
	}
	return options
}

func buildTypeOptions(kind string, entries []TypeSchemaEntry) []ClarificationOption {
	if len(entries) == 0 {
		return []ClarificationOption{
			{ID: "open_dsl", Label: "Open in DSL", Description: fmt.Sprintf("Repair the %s type manually in the DSL editor.", kind)},
		}
	}

	options := make([]ClarificationOption, 0, len(entries))
	for _, entry := range entries {
		options = append(options, ClarificationOption{
			ID:          sanitizeIdentifier(entry.TypeName),
			Label:       entry.TypeName,
			Description: entry.Description,
		})
	}
	return options
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value != "" {
			return value
		}
	}
	return ""
}
