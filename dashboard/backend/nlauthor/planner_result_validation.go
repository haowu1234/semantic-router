package nlauthor

import (
	"fmt"
	"reflect"
	"sort"
	"strings"
)

func normalizePlannerResult(result PlannerResult, manifest SchemaManifest) PlannerResult {
	if err := validatePlannerResult(result, manifest); err != nil {
		warnings := append([]PlannerWarning{}, result.Warnings...)
		warnings = append(warnings, PlannerWarning{
			Code:    "invalid_planner_result",
			Message: "Planner output did not match the backend NL schema contract.",
		})

		return PlannerResult{
			Status:      PlannerStatusError,
			Explanation: result.Explanation,
			Warnings:    warnings,
			Error:       fmt.Sprintf("Planner returned invalid structured output: %v", err),
		}
	}

	return result
}

func validatePlannerResult(result PlannerResult, manifest SchemaManifest) error {
	switch result.Status {
	case PlannerStatusReady:
		return validateReadyPlannerResult(result, manifest)
	case PlannerStatusNeedsClarification:
		return validateClarificationResult(result)
	case PlannerStatusUnsupported, PlannerStatusError:
		return nil
	default:
		return fmt.Errorf("unsupported planner status %q", result.Status)
	}
}

func validateReadyPlannerResult(result PlannerResult, manifest SchemaManifest) error {
	if result.IntentIR == nil {
		return fmt.Errorf("ready result is missing intentIr")
	}
	if strings.TrimSpace(string(result.IntentIR.Operation)) == "" {
		return fmt.Errorf("ready result is missing intentIr.operation")
	}
	if len(result.IntentIR.Intents) == 0 {
		return fmt.Errorf("ready result is missing intentIr.intents")
	}

	index := newSchemaIndex(manifest)
	for idx, intent := range result.IntentIR.Intents {
		if err := validateIntent(intent, index); err != nil {
			return fmt.Errorf("intent %d: %w", idx, err)
		}
	}
	return nil
}

func validateClarificationResult(result PlannerResult) error {
	if result.Clarification == nil {
		return fmt.Errorf("clarification result is missing clarification payload")
	}
	if strings.TrimSpace(result.Clarification.Question) == "" {
		return fmt.Errorf("clarification result is missing question")
	}
	if len(result.Clarification.Options) == 0 {
		return fmt.Errorf("clarification result has no options")
	}
	for idx, option := range result.Clarification.Options {
		if strings.TrimSpace(option.ID) == "" {
			return fmt.Errorf("clarification option %d is missing id", idx)
		}
		if strings.TrimSpace(option.Label) == "" {
			return fmt.Errorf("clarification option %d is missing label", idx)
		}
	}
	return nil
}

func validateIntent(intent map[string]interface{}, index schemaIndex) error {
	intentType, ok := requiredString(intent, "type")
	if !ok {
		return fmt.Errorf("missing type")
	}

	switch intentType {
	case "signal":
		return validateTypedFieldsIntent(intent, "signal_type", index.signals, "signal")
	case "plugin_template":
		return validateTypedFieldsIntent(intent, "plugin_type", index.plugins, "plugin")
	case "backend":
		return validateTypedFieldsIntent(intent, "backend_type", index.backends, "backend")
	case "route":
		return validateRouteIntent(intent, index)
	case "global":
		return validateGlobalIntent(intent)
	case "modify":
		return validateModifyIntent(intent, index)
	default:
		return fmt.Errorf("unsupported intent type %q", intentType)
	}
}

func validateTypedFieldsIntent(intent map[string]interface{}, typeKey string, allowed map[string]TypeSchemaEntry, label string) error {
	typeName, ok := requiredString(intent, typeKey)
	if !ok {
		return fmt.Errorf("missing %s", typeKey)
	}
	schema, ok := allowed[typeName]
	if !ok {
		return fmt.Errorf("unsupported %s type %q", label, typeName)
	}
	if _, hasName := requiredString(intent, "name"); !hasName {
		return fmt.Errorf("missing name")
	}
	fields, hasFields := requiredObject(intent, "fields")
	if !hasFields {
		return fmt.Errorf("missing fields")
	}
	return validateFieldMap(fields, schema, true)
}

func validateRouteIntent(intent map[string]interface{}, index schemaIndex) error {
	if _, ok := requiredString(intent, "name"); !ok {
		return fmt.Errorf("missing name")
	}
	models, ok := requiredArray(intent, "models")
	if !ok || len(models) == 0 {
		return fmt.Errorf("route requires at least one model")
	}
	for idx, modelEntry := range models {
		model, ok := modelEntry.(map[string]interface{})
		if !ok {
			return fmt.Errorf("model %d must be an object", idx)
		}
		if _, ok := requiredString(model, "model"); !ok {
			return fmt.Errorf("model %d is missing model", idx)
		}
	}

	if algorithm, ok := optionalObject(intent, "algorithm"); ok {
		algoType, ok := requiredString(algorithm, "algo_type")
		if !ok {
			return fmt.Errorf("route algorithm is missing algo_type")
		}
		schema, ok := index.algorithms[algoType]
		if !ok {
			return fmt.Errorf("unsupported algorithm type %q", algoType)
		}
		if params, ok := optionalObject(algorithm, "params"); ok {
			if err := validateFieldMap(params, schema, true); err != nil {
				return fmt.Errorf("route algorithm params: %w", err)
			}
		}
	}

	if condition, ok := optionalObject(intent, "condition"); ok {
		if err := validateConditionNode(condition); err != nil {
			return fmt.Errorf("route condition: %w", err)
		}
	}

	return nil
}

func validateGlobalIntent(intent map[string]interface{}) error {
	if _, ok := requiredObject(intent, "fields"); !ok {
		return fmt.Errorf("missing fields")
	}
	return nil
}

func validateModifyIntent(intent map[string]interface{}, index schemaIndex) error {
	action, ok := requiredString(intent, "action")
	if !ok {
		return fmt.Errorf("missing action")
	}
	targetConstruct, ok := requiredString(intent, "target_construct")
	if !ok {
		return fmt.Errorf("missing target_construct")
	}
	if _, ok := requiredString(intent, "target_name"); !ok {
		return fmt.Errorf("missing target_name")
	}

	if action == "delete" {
		return nil
	}

	if action != "add" && action != "update" {
		return fmt.Errorf("unsupported modify action %q", action)
	}

	switch targetConstruct {
	case "signal":
		return validateModifyTypedFields(intent, action, "target_signal_type", index.signals, "signal")
	case "plugin":
		return validateModifyTypedFields(intent, action, "target_plugin_type", index.plugins, "plugin")
	case "backend":
		return validateModifyTypedFields(intent, action, "target_backend_type", index.backends, "backend")
	case "route":
		changes, ok := requiredObject(intent, "changes")
		if !ok {
			return fmt.Errorf("missing changes")
		}
		return validateModifyRouteChanges(changes, index)
	case "global":
		if _, ok := requiredObject(intent, "changes"); !ok {
			return fmt.Errorf("missing changes")
		}
		return nil
	default:
		return fmt.Errorf("unsupported modify target_construct %q", targetConstruct)
	}
}

func validateModifyTypedFields(intent map[string]interface{}, action string, typeKey string, allowed map[string]TypeSchemaEntry, label string) error {
	typeName, ok := requiredString(intent, typeKey)
	if !ok {
		return fmt.Errorf("missing %s", typeKey)
	}
	schema, ok := allowed[typeName]
	if !ok {
		return fmt.Errorf("unsupported %s type %q", label, typeName)
	}
	changes, ok := requiredObject(intent, "changes")
	if !ok {
		return fmt.Errorf("missing changes")
	}
	return validateFieldMap(changes, schema, action == "add")
}

func validateModifyRouteChanges(changes map[string]interface{}, index schemaIndex) error {
	if len(changes) == 0 {
		return fmt.Errorf("missing supported changes")
	}

	for key := range changes {
		if !containsString([]string{"description", "priority", "condition", "models", "algorithm", "plugins"}, key) {
			return fmt.Errorf("unsupported route change field %q", key)
		}
	}

	if value, ok := changes["description"]; ok && value != nil {
		if _, ok := value.(string); !ok {
			return fmt.Errorf("route description must be a string or null")
		}
	}

	if value, ok := changes["priority"]; ok && value != nil && !isNumberValue(value) {
		return fmt.Errorf("route priority must be a number or null")
	}

	if value, ok := changes["models"]; ok {
		models, ok := asSlice(value)
		if !ok {
			return fmt.Errorf("route models must be an array")
		}
		if len(models) == 0 {
			return fmt.Errorf("route models cannot be empty")
		}
		for idx, modelEntry := range models {
			model, ok := modelEntry.(map[string]interface{})
			if !ok {
				return fmt.Errorf("model %d must be an object", idx)
			}
			if _, ok := requiredString(model, "model"); !ok {
				return fmt.Errorf("model %d is missing model", idx)
			}
		}
	}

	if value, ok := changes["algorithm"]; ok && value != nil {
		algorithm, ok := value.(map[string]interface{})
		if !ok {
			return fmt.Errorf("route algorithm must be an object or null")
		}
		algoType, ok := requiredString(algorithm, "algo_type")
		if !ok {
			return fmt.Errorf("route algorithm is missing algo_type")
		}
		schema, ok := index.algorithms[algoType]
		if !ok {
			return fmt.Errorf("unsupported algorithm type %q", algoType)
		}
		if params, ok := optionalObject(algorithm, "params"); ok {
			if err := validateFieldMap(params, schema, true); err != nil {
				return fmt.Errorf("route algorithm params: %w", err)
			}
		}
	}

	if value, ok := changes["condition"]; ok && value != nil {
		condition, ok := value.(map[string]interface{})
		if !ok {
			return fmt.Errorf("route condition must be an object or null")
		}
		if err := validateConditionNode(condition); err != nil {
			return fmt.Errorf("route condition: %w", err)
		}
	}

	if value, ok := changes["plugins"]; ok {
		plugins, ok := asSlice(value)
		if !ok {
			return fmt.Errorf("route plugins must be an array")
		}
		for idx, pluginEntry := range plugins {
			plugin, ok := pluginEntry.(map[string]interface{})
			if !ok {
				return fmt.Errorf("plugin %d must be an object", idx)
			}
			if _, ok := requiredString(plugin, "name"); !ok {
				return fmt.Errorf("plugin %d is missing name", idx)
			}
			if overrides, ok := plugin["overrides"]; ok && overrides != nil {
				if _, ok := overrides.(map[string]interface{}); !ok {
					return fmt.Errorf("plugin %d overrides must be an object or null", idx)
				}
			}
		}
	}

	return nil
}

func validateConditionNode(node map[string]interface{}) error {
	op, ok := requiredString(node, "op")
	if !ok {
		return fmt.Errorf("missing op")
	}

	switch op {
	case "SIGNAL_REF":
		if _, ok := requiredString(node, "signal_type"); !ok {
			return fmt.Errorf("missing signal_type")
		}
		if _, ok := requiredString(node, "signal_name"); !ok {
			return fmt.Errorf("missing signal_name")
		}
		return nil
	case "NOT":
		operand, ok := requiredObject(node, "operand")
		if !ok {
			return fmt.Errorf("missing operand")
		}
		return validateConditionNode(operand)
	case "AND", "OR":
		operands, ok := requiredArray(node, "operands")
		if !ok || len(operands) == 0 {
			return fmt.Errorf("%s requires operands", op)
		}
		for idx, operand := range operands {
			child, ok := operand.(map[string]interface{})
			if !ok {
				return fmt.Errorf("operand %d must be an object", idx)
			}
			if err := validateConditionNode(child); err != nil {
				return fmt.Errorf("operand %d: %w", idx, err)
			}
		}
		return nil
	default:
		return fmt.Errorf("unsupported op %q", op)
	}
}

func validateFieldMap(fields map[string]interface{}, schema TypeSchemaEntry, requireRequiredFields bool) error {
	if len(fields) == 0 {
		requiredFields := requiredFieldKeys(schema.Fields)
		if requireRequiredFields && len(requiredFields) > 0 {
			return fmt.Errorf("missing required fields: %s", strings.Join(requiredFields, ", "))
		}
		return nil
	}

	fieldIndex := make(map[string]FieldSchema, len(schema.Fields))
	for _, field := range schema.Fields {
		fieldIndex[field.Key] = field
	}

	for key, value := range fields {
		fieldSchema, ok := fieldIndex[key]
		if !ok {
			return fmt.Errorf("field %q is not declared in schema", key)
		}
		if err := validateFieldValue(value, fieldSchema); err != nil {
			return fmt.Errorf("field %q: %w", key, err)
		}
	}

	missing := make([]string, 0)
	if requireRequiredFields {
		for _, field := range schema.Fields {
			if !field.Required {
				continue
			}
			if value, ok := fields[field.Key]; !ok || isEmptyRequiredValue(value) {
				missing = append(missing, field.Key)
			}
		}
	}
	if len(missing) > 0 {
		sort.Strings(missing)
		return fmt.Errorf("missing required fields: %s", strings.Join(missing, ", "))
	}

	return nil
}

func validateFieldValue(value interface{}, schema FieldSchema) error {
	switch schema.Type {
	case FieldTypeString, FieldTypeSelect:
		stringValue, ok := value.(string)
		if !ok {
			return fmt.Errorf("expected string")
		}
		if schema.Type == FieldTypeSelect && len(schema.Options) > 0 && !containsString(schema.Options, stringValue) {
			return fmt.Errorf("expected one of %s", strings.Join(schema.Options, ", "))
		}
		return nil
	case FieldTypeNumber:
		if !isNumberValue(value) {
			return fmt.Errorf("expected number")
		}
		return nil
	case FieldTypeBoolean:
		if _, ok := value.(bool); !ok {
			return fmt.Errorf("expected boolean")
		}
		return nil
	case FieldTypeStringArray:
		return validateArrayValue(value, func(item interface{}) bool {
			_, ok := item.(string)
			return ok
		}, "string[]")
	case FieldTypeNumberArray:
		return validateArrayValue(value, isNumberValue, "number[]")
	case FieldTypeJSON:
		return nil
	default:
		return fmt.Errorf("unsupported field type %q", schema.Type)
	}
}

func validateArrayValue(value interface{}, accepts func(interface{}) bool, label string) error {
	items, ok := asSlice(value)
	if !ok {
		return fmt.Errorf("expected %s", label)
	}
	for idx, item := range items {
		if !accepts(item) {
			return fmt.Errorf("item %d must be %s", idx, label)
		}
	}
	return nil
}

func requiredString(m map[string]interface{}, key string) (string, bool) {
	value, ok := m[key]
	if !ok {
		return "", false
	}
	stringValue, ok := value.(string)
	if !ok || strings.TrimSpace(stringValue) == "" {
		return "", false
	}
	return stringValue, true
}

func requiredObject(m map[string]interface{}, key string) (map[string]interface{}, bool) {
	value, ok := m[key]
	if !ok || value == nil {
		return nil, false
	}
	objectValue, ok := value.(map[string]interface{})
	if !ok || objectValue == nil {
		return nil, false
	}
	return objectValue, true
}

func optionalObject(m map[string]interface{}, key string) (map[string]interface{}, bool) {
	value, ok := m[key]
	if !ok || value == nil {
		return nil, false
	}
	objectValue, ok := value.(map[string]interface{})
	if !ok || objectValue == nil {
		return nil, false
	}
	return objectValue, true
}

func requiredArray(m map[string]interface{}, key string) ([]interface{}, bool) {
	value, ok := m[key]
	if !ok || value == nil {
		return nil, false
	}
	return asSlice(value)
}

func isNumberValue(value interface{}) bool {
	switch value.(type) {
	case int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64, float32, float64:
		return true
	default:
		return false
	}
}

func asSlice(value interface{}) ([]interface{}, bool) {
	rv := reflect.ValueOf(value)
	if !rv.IsValid() {
		return nil, false
	}
	if rv.Kind() != reflect.Slice && rv.Kind() != reflect.Array {
		return nil, false
	}

	items := make([]interface{}, rv.Len())
	for i := 0; i < rv.Len(); i++ {
		items[i] = rv.Index(i).Interface()
	}
	return items, true
}

func containsString(values []string, candidate string) bool {
	for _, value := range values {
		if value == candidate {
			return true
		}
	}
	return false
}

func isEmptyRequiredValue(value interface{}) bool {
	switch typed := value.(type) {
	case string:
		return strings.TrimSpace(typed) == ""
	case []interface{}:
		return len(typed) == 0
	case []string:
		return len(typed) == 0
	default:
		return value == nil
	}
}

func requiredFieldKeys(fields []FieldSchema) []string {
	keys := make([]string, 0)
	for _, field := range fields {
		if field.Required {
			keys = append(keys, field.Key)
		}
	}
	sort.Strings(keys)
	return keys
}

type schemaIndex struct {
	signals    map[string]TypeSchemaEntry
	plugins    map[string]TypeSchemaEntry
	algorithms map[string]TypeSchemaEntry
	backends   map[string]TypeSchemaEntry
}

func newSchemaIndex(manifest SchemaManifest) schemaIndex {
	return schemaIndex{
		signals:    indexTypeSchemas(manifest.Signals),
		plugins:    indexTypeSchemas(manifest.Plugins),
		algorithms: indexTypeSchemas(manifest.Algorithms),
		backends:   indexTypeSchemas(manifest.Backends),
	}
}

func indexTypeSchemas(entries []TypeSchemaEntry) map[string]TypeSchemaEntry {
	index := make(map[string]TypeSchemaEntry, len(entries))
	for _, entry := range entries {
		index[entry.TypeName] = entry
	}
	return index
}
