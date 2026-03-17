package config

import "strings"

const (
	DatabaseDriverSQLite   = "sqlite3"
	DatabaseDriverPostgres = "postgres"

	DefaultAuthDBPath       = "./data/auth.db"
	DefaultEvaluationDBPath = "./data/evaluations.db"
)

// DatabaseConfig describes a dashboard persistence target.
type DatabaseConfig struct {
	Driver string
	URL    string
	Path   string
}

// Resolved returns a normalized config with defaults applied.
func (cfg DatabaseConfig) Resolved(defaultPath string) DatabaseConfig {
	driver := strings.ToLower(strings.TrimSpace(cfg.Driver))
	url := strings.TrimSpace(cfg.URL)
	path := strings.TrimSpace(cfg.Path)

	switch driver {
	case "":
		if strings.HasPrefix(strings.ToLower(url), "postgres://") || strings.HasPrefix(strings.ToLower(url), "postgresql://") {
			driver = DatabaseDriverPostgres
		} else {
			driver = DatabaseDriverSQLite
		}
	case "sqlite":
		driver = DatabaseDriverSQLite
	case "postgresql":
		driver = DatabaseDriverPostgres
	}

	if driver == DatabaseDriverSQLite && path == "" {
		path = defaultPath
	}

	return DatabaseConfig{
		Driver: driver,
		URL:    url,
		Path:   path,
	}
}

func (cfg *Config) EffectiveAuthDB() DatabaseConfig {
	if cfg == nil {
		return DatabaseConfig{}.Resolved(DefaultAuthDBPath)
	}

	resolved := cfg.AuthDB.Resolved(DefaultAuthDBPath)
	if strings.TrimSpace(cfg.AuthDBPath) != "" {
		resolved.Path = strings.TrimSpace(cfg.AuthDBPath)
	}
	if resolved.Driver == DatabaseDriverSQLite && resolved.Path == "" {
		resolved.Path = DefaultAuthDBPath
	}
	return resolved
}

func (cfg *Config) EffectiveEvaluationDB() DatabaseConfig {
	if cfg == nil {
		return DatabaseConfig{}.Resolved(DefaultEvaluationDBPath)
	}

	resolved := cfg.EvaluationDB.Resolved(DefaultEvaluationDBPath)
	if strings.TrimSpace(cfg.EvaluationDBPath) != "" {
		resolved.Path = strings.TrimSpace(cfg.EvaluationDBPath)
	}
	if resolved.Driver == DatabaseDriverSQLite && resolved.Path == "" {
		resolved.Path = DefaultEvaluationDBPath
	}
	return resolved
}
