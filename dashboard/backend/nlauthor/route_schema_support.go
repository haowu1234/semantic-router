package nlauthor

import "strings"

func plannerSupportsConstruct(support PlannerSupport, construct ConstructKind) bool {
	for _, candidate := range support.Constructs {
		if candidate == construct {
			return true
		}
	}
	return false
}

func routeSchemaEntries(typeName string, support PlannerSupport) []TypeSchemaEntry {
	if !plannerSupportsConstruct(support, ConstructRoute) {
		return []TypeSchemaEntry{}
	}
	if trimmed := strings.TrimSpace(typeName); trimmed != "" && trimmed != string(ConstructRoute) {
		return []TypeSchemaEntry{}
	}

	return []TypeSchemaEntry{
		{
			TypeName:    string(ConstructRoute),
			Description: "Create or update a route with models, optional condition, optional plugins, and optional algorithm settings.",
			Fields: []FieldSchema{
				{Key: "name", Label: "Name", Type: FieldTypeString, Required: true, Description: "Stable route identifier."},
				{Key: "description", Label: "Description", Type: FieldTypeString, Description: "Optional route description shown in Builder and reviews."},
				{Key: "priority", Label: "Priority", Type: FieldTypeNumber, Description: "Route priority. Lower values evaluate earlier unless project conventions differ."},
				{Key: "models", Label: "Models", Type: FieldTypeJSON, Required: true, Description: "Ordered model refs; each item needs at least a model name."},
				{Key: "condition", Label: "Condition", Type: FieldTypeJSON, Description: "Optional signal condition tree using SIGNAL_REF / AND / OR / NOT nodes."},
				{Key: "plugins", Label: "Plugins", Type: FieldTypeJSON, Description: "Optional plugin refs attached to the route."},
				{Key: "algorithm", Label: "Algorithm", Type: FieldTypeJSON, Description: "Optional route-local algorithm block with algo_type and params."},
			},
		},
	}
}
