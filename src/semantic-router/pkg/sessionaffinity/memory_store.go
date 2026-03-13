package sessionaffinity

import (
	"sync"
	"time"
)

type memoryEntry struct {
	state     State
	expiresAt time.Time
}

// MemoryStore keeps affinity state in-process for single-router deployments.
type MemoryStore struct {
	mu     sync.RWMutex
	states map[string]memoryEntry
}

func NewMemoryStore() *MemoryStore {
	return &MemoryStore{
		states: make(map[string]memoryEntry),
	}
}

func (s *MemoryStore) Get(key string) (*State, error) {
	s.mu.RLock()
	entry, ok := s.states[key]
	s.mu.RUnlock()
	if !ok {
		return nil, ErrNotFound
	}
	if !entry.expiresAt.IsZero() && time.Now().After(entry.expiresAt) {
		s.mu.Lock()
		delete(s.states, key)
		s.mu.Unlock()
		return nil, ErrNotFound
	}
	state := entry.state
	return &state, nil
}

func (s *MemoryStore) Put(state *State, ttl time.Duration) error {
	if state == nil || state.Key == "" {
		return nil
	}
	entry := memoryEntry{state: *state}
	if ttl > 0 {
		entry.expiresAt = time.Now().Add(ttl)
	}
	s.mu.Lock()
	s.states[state.Key] = entry
	s.mu.Unlock()
	return nil
}

func (s *MemoryStore) Delete(key string) error {
	s.mu.Lock()
	delete(s.states, key)
	s.mu.Unlock()
	return nil
}

func (s *MemoryStore) Close() error {
	s.mu.Lock()
	s.states = make(map[string]memoryEntry)
	s.mu.Unlock()
	return nil
}
