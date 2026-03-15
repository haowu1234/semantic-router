package nlauthor

import (
	"errors"
	"sync"
	"time"
)

var errSessionNotFound = errors.New("nl session not found")

// Session is the backend-owned NL authoring session state.
type Session struct {
	ID            string
	SchemaVersion string
	OwnerUserID   string
	Context       SessionContext
	CreatedAt     time.Time
	UpdatedAt     time.Time
	ExpiresAt     time.Time
}

// SessionStore persists short-lived NL planning sessions.
type SessionStore interface {
	Create(session Session) error
	Get(id string, now time.Time) (Session, error)
	Update(session Session, now time.Time) error
}

// InMemorySessionStore keeps preview sessions in process memory with TTL-based eviction.
type InMemorySessionStore struct {
	mu       sync.Mutex
	sessions map[string]Session
}

func NewInMemorySessionStore() *InMemorySessionStore {
	return &InMemorySessionStore{
		sessions: make(map[string]Session),
	}
}

func (s *InMemorySessionStore) Create(session Session) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.purgeExpiredLocked(time.Now().UTC())
	s.sessions[session.ID] = session
	return nil
}

func (s *InMemorySessionStore) Get(id string, now time.Time) (Session, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.purgeExpiredLocked(now)
	session, ok := s.sessions[id]
	if !ok {
		return Session{}, errSessionNotFound
	}
	return session, nil
}

func (s *InMemorySessionStore) Update(session Session, now time.Time) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.purgeExpiredLocked(now)
	if _, ok := s.sessions[session.ID]; !ok {
		return errSessionNotFound
	}

	s.sessions[session.ID] = session
	return nil
}

func (s *InMemorySessionStore) purgeExpiredLocked(now time.Time) {
	for id, session := range s.sessions {
		if !session.ExpiresAt.IsZero() && !session.ExpiresAt.After(now) {
			delete(s.sessions, id)
		}
	}
}
