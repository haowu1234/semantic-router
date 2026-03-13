package extproc

import (
	"fmt"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessionaffinity"
)

func createSessionAffinityRuntime(cfg *config.RouterConfig) (*sessionaffinity.Manager, error) {
	if cfg == nil || !cfg.IntelligentRouting.SessionAffinity.Enabled {
		return nil, nil
	}

	affinityCfg := cfg.IntelligentRouting.SessionAffinity
	managerCfg := sessionaffinity.Config{
		Enabled:                   affinityCfg.Enabled,
		RequireTrustedUser:        affinityCfg.RequireTrustedUser,
		TTL:                       time.Duration(affinityCfg.TTLSeconds) * time.Second,
		ImmediateUpgradeThreshold: affinityCfg.EffectiveImmediateUpgradeThreshold(),
		ReleaseAfterPendingTurns:  affinityCfg.EffectiveReleaseAfterPendingTurns(),
		NegativeFeedbackSignals:   make(map[string]struct{}),
	}
	for _, signal := range affinityCfg.EffectiveNegativeFeedbackSignals() {
		managerCfg.NegativeFeedbackSignals[signal] = struct{}{}
	}

	storeBackend := strings.ToLower(affinityCfg.EffectiveStoreBackend())
	var store sessionaffinity.Store
	switch storeBackend {
	case "memory":
		store = sessionaffinity.NewMemoryStore()
	case "redis":
		redisStore, err := sessionaffinity.NewRedisStore(sessionaffinity.RedisConfig{
			Address:          affinityCfg.Redis.Address,
			Password:         affinityCfg.Redis.Password,
			DB:               affinityCfg.Redis.DB,
			KeyPrefix:        affinityCfg.Redis.KeyPrefix,
			ClusterMode:      affinityCfg.Redis.ClusterMode,
			ClusterAddresses: affinityCfg.Redis.ClusterAddresses,
			PoolSize:         affinityCfg.Redis.PoolSize,
			MinIdleConns:     affinityCfg.Redis.MinIdleConns,
			MaxRetries:       affinityCfg.Redis.MaxRetries,
			DialTimeout:      affinityCfg.Redis.DialTimeout,
			ReadTimeout:      affinityCfg.Redis.ReadTimeout,
			WriteTimeout:     affinityCfg.Redis.WriteTimeout,
			TLSEnabled:       affinityCfg.Redis.TLSEnabled,
			TLSCertPath:      affinityCfg.Redis.TLSCertPath,
			TLSKeyPath:       affinityCfg.Redis.TLSKeyPath,
			TLSCAPath:        affinityCfg.Redis.TLSCAPath,
		})
		if err != nil {
			return nil, fmt.Errorf("create session affinity redis store: %w", err)
		}
		store = redisStore
	default:
		return nil, fmt.Errorf("unsupported session_affinity.store_backend %q", affinityCfg.StoreBackend)
	}

	logging.Infof("SessionAffinity: enabled backend=%s require_trusted_user=%v ttl_seconds=%d",
		storeBackend, affinityCfg.RequireTrustedUser, affinityCfg.TTLSeconds)
	return sessionaffinity.NewManager(managerCfg, store), nil
}
