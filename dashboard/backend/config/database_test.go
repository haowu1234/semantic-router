package config

import "testing"

func TestDatabaseConfigResolvedDefaultsToSQLite(t *testing.T) {
	t.Parallel()

	resolved := (DatabaseConfig{}).Resolved(DefaultAuthDBPath)
	if resolved.Driver != DatabaseDriverSQLite {
		t.Fatalf("driver = %q, want %q", resolved.Driver, DatabaseDriverSQLite)
	}
	if resolved.Path != DefaultAuthDBPath {
		t.Fatalf("path = %q, want %q", resolved.Path, DefaultAuthDBPath)
	}
}

func TestDatabaseConfigResolvedInfersPostgresFromURL(t *testing.T) {
	t.Parallel()

	resolved := (DatabaseConfig{URL: "postgres://user:pass@localhost:5432/router"}).Resolved(DefaultAuthDBPath)
	if resolved.Driver != DatabaseDriverPostgres {
		t.Fatalf("driver = %q, want %q", resolved.Driver, DatabaseDriverPostgres)
	}
	if resolved.Path != "" {
		t.Fatalf("path = %q, want empty for postgres URL", resolved.Path)
	}
}

func TestConfigEffectiveAuthDBPrefersLegacyPath(t *testing.T) {
	t.Parallel()

	cfg := &Config{
		AuthDBPath: "custom-auth.db",
		AuthDB: DatabaseConfig{
			Driver: DatabaseDriverSQLite,
			Path:   "ignored.db",
		},
	}

	resolved := cfg.EffectiveAuthDB()
	if resolved.Path != "custom-auth.db" {
		t.Fatalf("path = %q, want %q", resolved.Path, "custom-auth.db")
	}
}
