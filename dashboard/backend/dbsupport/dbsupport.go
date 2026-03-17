package dbsupport

import (
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	_ "github.com/lib/pq"
	_ "github.com/mattn/go-sqlite3"

	backendconfig "github.com/vllm-project/semantic-router/dashboard/backend/config"
)

type OpenOptions struct {
	SQLiteDSNSuffix      string
	SQLiteMaxOpenConns   int
	PostgresMaxOpenConns int
	ConnMaxLifetime      time.Duration
}

func Open(dbCfg backendconfig.DatabaseConfig, defaultPath string, opts OpenOptions) (*sql.DB, backendconfig.DatabaseConfig, error) {
	resolved := dbCfg.Resolved(defaultPath)

	switch resolved.Driver {
	case backendconfig.DatabaseDriverSQLite:
		if err := ensureSQLiteDir(resolved.Path); err != nil {
			return nil, backendconfig.DatabaseConfig{}, err
		}

		db, err := sql.Open("sqlite3", sqliteDSN(resolved.Path, opts.SQLiteDSNSuffix))
		if err != nil {
			return nil, backendconfig.DatabaseConfig{}, fmt.Errorf("open sqlite database: %w", err)
		}
		if opts.SQLiteMaxOpenConns > 0 {
			db.SetMaxOpenConns(opts.SQLiteMaxOpenConns)
		}
		if opts.ConnMaxLifetime > 0 {
			db.SetConnMaxLifetime(opts.ConnMaxLifetime)
		}
		if err := db.Ping(); err != nil {
			_ = db.Close()
			return nil, backendconfig.DatabaseConfig{}, fmt.Errorf("ping sqlite database: %w", err)
		}
		return db, resolved, nil

	case backendconfig.DatabaseDriverPostgres:
		if resolved.URL == "" {
			return nil, backendconfig.DatabaseConfig{}, fmt.Errorf("database URL is required for postgres")
		}

		db, err := sql.Open("postgres", resolved.URL)
		if err != nil {
			return nil, backendconfig.DatabaseConfig{}, fmt.Errorf("open postgres database: %w", err)
		}
		if opts.PostgresMaxOpenConns > 0 {
			db.SetMaxOpenConns(opts.PostgresMaxOpenConns)
		}
		if opts.ConnMaxLifetime > 0 {
			db.SetConnMaxLifetime(opts.ConnMaxLifetime)
		}
		if err := db.Ping(); err != nil {
			_ = db.Close()
			return nil, backendconfig.DatabaseConfig{}, fmt.Errorf("ping postgres database: %w", err)
		}
		return db, resolved, nil
	}

	return nil, backendconfig.DatabaseConfig{}, fmt.Errorf("unsupported database driver %q", resolved.Driver)
}

func Rebind(driver string, query string) string {
	if driver != backendconfig.DatabaseDriverPostgres {
		return query
	}

	var builder strings.Builder
	builder.Grow(len(query) + 8)
	index := 1
	for i := 0; i < len(query); i++ {
		if query[i] != '?' {
			builder.WriteByte(query[i])
			continue
		}
		builder.WriteByte('$')
		builder.WriteString(strconv.Itoa(index))
		index++
	}
	return builder.String()
}

func ensureSQLiteDir(path string) error {
	dir := filepath.Dir(path)
	if dir == "." || dir == "" || dir == "/" {
		return nil
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("create sqlite database directory: %w", err)
	}
	return nil
}

func sqliteDSN(path string, suffix string) string {
	if suffix == "" {
		return path
	}
	if strings.Contains(path, "?") && strings.HasPrefix(suffix, "?") {
		return path + "&" + strings.TrimPrefix(suffix, "?")
	}
	return path + suffix
}
