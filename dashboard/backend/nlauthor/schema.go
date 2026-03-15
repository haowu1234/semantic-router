package nlauthor

import (
	_ "embed"
	"encoding/json"
	"fmt"
)

//go:embed catalog/schema_manifest.json
var defaultSchemaManifestJSON []byte

var defaultSchemaManifest = mustLoadDefaultSchemaManifest()

// DefaultSchemaManifest returns the current backend-owned NL authoring manifest.
func DefaultSchemaManifest() SchemaManifest {
	return defaultSchemaManifest
}

func mustLoadDefaultSchemaManifest() SchemaManifest {
	var manifest SchemaManifest
	if err := json.Unmarshal(defaultSchemaManifestJSON, &manifest); err != nil {
		panic(fmt.Sprintf("nlauthor: invalid embedded schema manifest: %v", err))
	}
	if manifest.Version == "" {
		manifest.Version = SchemaVersion
	}
	return manifest
}
