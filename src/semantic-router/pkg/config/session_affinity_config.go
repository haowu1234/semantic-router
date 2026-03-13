package config

// SessionAffinityConfig defines the global multi-turn session-affinity contract.
// This lives at the IntelligentRouting layer because it arbitrates across
// selector families instead of belonging to one selector implementation.
type SessionAffinityConfig struct {
	Enabled                   bool                       `yaml:"enabled,omitempty"`
	RequireTrustedUser        bool                       `yaml:"require_trusted_user,omitempty"`
	SessionIDHeader           string                     `yaml:"session_id_header,omitempty"`
	StoreBackend              string                     `yaml:"store_backend,omitempty"`
	TTLSeconds                int                        `yaml:"ttl_seconds,omitempty"`
	ImmediateUpgradeThreshold float64                    `yaml:"immediate_upgrade_threshold,omitempty"`
	ReleaseAfterPendingTurns  int                        `yaml:"release_after_pending_turns,omitempty"`
	NegativeFeedbackSignals   []string                   `yaml:"negative_feedback_signals,omitempty"`
	Redis                     SessionAffinityRedisConfig `yaml:"redis,omitempty"`
}

// SessionAffinityRedisConfig configures Redis-backed affinity state.
type SessionAffinityRedisConfig struct {
	Address          string   `yaml:"address,omitempty"`
	Password         string   `yaml:"password,omitempty"`
	DB               int      `yaml:"db,omitempty"`
	KeyPrefix        string   `yaml:"key_prefix,omitempty"`
	ClusterMode      bool     `yaml:"cluster_mode,omitempty"`
	ClusterAddresses []string `yaml:"cluster_addresses,omitempty"`
	PoolSize         int      `yaml:"pool_size,omitempty"`
	MinIdleConns     int      `yaml:"min_idle_conns,omitempty"`
	MaxRetries       int      `yaml:"max_retries,omitempty"`
	DialTimeout      int      `yaml:"dial_timeout,omitempty"`
	ReadTimeout      int      `yaml:"read_timeout,omitempty"`
	WriteTimeout     int      `yaml:"write_timeout,omitempty"`
	TLSEnabled       bool     `yaml:"tls_enabled,omitempty"`
	TLSCertPath      string   `yaml:"tls_cert_path,omitempty"`
	TLSKeyPath       string   `yaml:"tls_key_path,omitempty"`
	TLSCAPath        string   `yaml:"tls_ca_path,omitempty"`
}

func (c SessionAffinityConfig) EffectiveSessionIDHeader() string {
	if c.SessionIDHeader == "" {
		return "x-vsr-session-id"
	}
	return c.SessionIDHeader
}

func (c SessionAffinityConfig) EffectiveStoreBackend() string {
	if c.StoreBackend == "" {
		return "memory"
	}
	return c.StoreBackend
}

func (c SessionAffinityConfig) EffectiveImmediateUpgradeThreshold() float64 {
	if c.ImmediateUpgradeThreshold <= 0 {
		return 0.15
	}
	return c.ImmediateUpgradeThreshold
}

func (c SessionAffinityConfig) EffectiveReleaseAfterPendingTurns() int {
	if c.ReleaseAfterPendingTurns <= 0 {
		return 2
	}
	return c.ReleaseAfterPendingTurns
}

func (c SessionAffinityConfig) EffectiveNegativeFeedbackSignals() []string {
	if len(c.NegativeFeedbackSignals) == 0 {
		return []string{"wrong_answer", "want_different"}
	}
	return c.NegativeFeedbackSignals
}
