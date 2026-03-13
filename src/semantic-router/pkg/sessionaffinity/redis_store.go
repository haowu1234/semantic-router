package sessionaffinity

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/redis/go-redis/v9"
)

type RedisConfig struct {
	Address          string
	Password         string
	DB               int
	KeyPrefix        string
	ClusterMode      bool
	ClusterAddresses []string
	PoolSize         int
	MinIdleConns     int
	MaxRetries       int
	DialTimeout      int
	ReadTimeout      int
	WriteTimeout     int
	TLSEnabled       bool
	TLSCertPath      string
	TLSKeyPath       string
	TLSCAPath        string
}

// RedisStore keeps affinity state in Redis for multi-router deployments.
type RedisStore struct {
	client    redis.UniversalClient
	keyPrefix string
}

func NewRedisStore(cfg RedisConfig) (*RedisStore, error) {
	keyPrefix := cfg.KeyPrefix
	if keyPrefix == "" {
		keyPrefix = "sr:session-affinity:"
	}
	if !strings.HasSuffix(keyPrefix, ":") {
		keyPrefix += ":"
	}

	client, err := newRedisUniversalClient(cfg)
	if err != nil {
		return nil, err
	}

	return &RedisStore{
		client:    client,
		keyPrefix: keyPrefix,
	}, nil
}

func (s *RedisStore) Get(key string) (*State, error) {
	data, err := s.client.Get(context.Background(), s.prefixedKey(key)).Bytes()
	if err != nil {
		if err == redis.Nil {
			return nil, ErrNotFound
		}
		return nil, fmt.Errorf("get affinity state: %w", err)
	}

	var state State
	if err := json.Unmarshal(data, &state); err != nil {
		return nil, fmt.Errorf("decode affinity state: %w", err)
	}
	return &state, nil
}

func (s *RedisStore) Put(state *State, ttl time.Duration) error {
	if state == nil || state.Key == "" {
		return nil
	}
	data, err := json.Marshal(state)
	if err != nil {
		return fmt.Errorf("encode affinity state: %w", err)
	}
	if err := s.client.Set(context.Background(), s.prefixedKey(state.Key), data, ttl).Err(); err != nil {
		return fmt.Errorf("put affinity state: %w", err)
	}
	return nil
}

func (s *RedisStore) Delete(key string) error {
	if err := s.client.Del(context.Background(), s.prefixedKey(key)).Err(); err != nil {
		return fmt.Errorf("delete affinity state: %w", err)
	}
	return nil
}

func (s *RedisStore) Close() error {
	return s.client.Close()
}

func (s *RedisStore) prefixedKey(key string) string {
	return s.keyPrefix + key
}

func newRedisUniversalClient(cfg RedisConfig) (redis.UniversalClient, error) {
	tlsConfig, err := buildRedisTLSConfig(cfg)
	if err != nil {
		return nil, err
	}

	addrs := cfg.ClusterAddresses
	if len(addrs) == 0 && cfg.Address != "" {
		addrs = []string{cfg.Address}
	}
	if len(addrs) == 0 {
		return nil, fmt.Errorf("redis address is required")
	}

	options := &redis.UniversalOptions{
		Addrs:        addrs,
		Password:     cfg.Password,
		DB:           cfg.DB,
		PoolSize:     cfg.PoolSize,
		MinIdleConns: cfg.MinIdleConns,
		MaxRetries:   cfg.MaxRetries,
		TLSConfig:    tlsConfig,
	}
	if cfg.DialTimeout > 0 {
		options.DialTimeout = time.Duration(cfg.DialTimeout) * time.Second
	}
	if cfg.ReadTimeout > 0 {
		options.ReadTimeout = time.Duration(cfg.ReadTimeout) * time.Second
	}
	if cfg.WriteTimeout > 0 {
		options.WriteTimeout = time.Duration(cfg.WriteTimeout) * time.Second
	}

	return redis.NewUniversalClient(options), nil
}

func buildRedisTLSConfig(cfg RedisConfig) (*tls.Config, error) {
	if !cfg.TLSEnabled {
		return nil, nil
	}

	tlsConfig := &tls.Config{MinVersion: tls.VersionTLS12}

	if cfg.TLSCAPath != "" {
		caPEM, err := os.ReadFile(cfg.TLSCAPath)
		if err != nil {
			return nil, fmt.Errorf("read redis CA: %w", err)
		}
		pool := x509.NewCertPool()
		if !pool.AppendCertsFromPEM(caPEM) {
			return nil, fmt.Errorf("parse redis CA bundle")
		}
		tlsConfig.RootCAs = pool
	}

	if cfg.TLSCertPath != "" && cfg.TLSKeyPath != "" {
		cert, err := tls.LoadX509KeyPair(cfg.TLSCertPath, cfg.TLSKeyPath)
		if err != nil {
			return nil, fmt.Errorf("load redis TLS keypair: %w", err)
		}
		tlsConfig.Certificates = []tls.Certificate{cert}
	}

	return tlsConfig, nil
}
